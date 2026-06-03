from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl
from helion._compiler.convert_utils import sparse_convert

# ----------------------------------------------------------------------------
# 2-D sparse-tile SPMM tests.
#
# One rich fixture (irregular nnz per row, one empty row) is re-encoded into
# every (root, inner) level-format pair and fed through one of two SPMM
# kernels depending on whether the inner level is ``Dense``:
#
#   * ``Dense`` inner  → ``spmm_dense_inner``: ``tile_k`` is a pure 1-D tile
#     (new semantics), so indexing into ``B`` is ``B[tile_k, tile_n]`` and
#     the inner product is a straight ``hl.dot`` on the parent-inherited
#     ``tile_k.value``.
#   * Any other inner  → ``spmm_sparse_inner``: ``tile_k`` is an N-D
#     parent-inherited coord, so ``B`` needs explicit broadcasted indexing
#     and the contraction is an element-wise multiply-then-sum.
#
# ``Padded`` and ``Jagged`` are forbidden at root (neither has meaningful
# semantics without a parent), so we cover 3 × 5 = 15 combos
# (Dense / Compressed / Bitmap root) × (Dense / Compressed / Padded / Jagged
# / Bitmap inner).
# ----------------------------------------------------------------------------

# Logical 5x6 matrix with irregular nnz-per-row and one empty row.
#   row 0: cols 0, 2, 5      (nnz=3)
#   row 1: empty             (nnz=0)  ← stresses root-level sentinel / bitmap
#   row 2: col  1            (nnz=1)
#   row 3: cols 0, 3, 4, 5   (nnz=4)  ← max per-row nnz → inner pad_size=4
#   row 4: cols 2, 4         (nnz=2)
_DENSE_A = torch.tensor(
    [
        [1.0, 0.0, 2.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 6.0, 7.0, 8.0],
        [0.0, 0.0, 9.0, 0.0, 10.0, 0.0],
    ],
    device=DEVICE,
)
_SHAPE = (5, 6)
_M, _K = _SHAPE
_N = 4
_B = torch.arange(_K * _N, dtype=torch.float32, device=DEVICE).reshape(_K, _N) * 0.1

_ROOT_FORMATS = ("Dense", "Compressed", "Bitmap")
_INNER_FORMATS = ("Dense", "Compressed", "Padded", "Jagged", "Bitmap")


def _build_sparse_2d(fmt0: str, fmt1: str) -> hl.SparseTensor:
    # Standard COO: coords is (ndim, nnz) with one row per dim, values holds
    # the matching non-zeros in the same column order.
    coo = torch.nonzero(_DENSE_A)  # (nnz, ndim)
    coords = coo.t().contiguous()  # (ndim, nnz)
    values = _DENSE_A[coords[0], coords[1]]  # (nnz,)

    return sparse_convert(values, coords, _SHAPE, [[0], [1]], [[1], [1]], [fmt0, fmt1])


@helion.kernel(config=helion.Config(block_sizes=[4, 4, 8]))
def spmm_sparse_inner(
    A: hl.SparseTensor,
    B: torch.Tensor,
    fmt0: hl.constexpr,
    fmt1: hl.constexpr,
) -> torch.Tensor:
    """SPMM for non-Dense inner.  ``tile_k`` is ND parent-inherited."""
    M = A.shape[0]
    N = B.size(1)
    C = torch.zeros(M, N, dtype=B.dtype, device=B.device)
    for tile_n in hl.tile(N):
        for tile_m in hl.sparse_tile(A, dim=0, levelformat=fmt0):
            acc = hl.zeros([tile_m.size(0), tile_n], dtype=B.dtype)
            for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat=fmt1):
                a_val = tile_k.value  # (M, K)
                b_val = B[tile_k[:, :, None], tile_n.index[None, None, :]]
                acc = acc + (a_val.unsqueeze(-1) * b_val).sum(dim=1)
            C[tile_m, tile_n] = acc
    return C


@helion.kernel(config=helion.Config(block_sizes=[4, 4, 8]))
def spmm_dense_inner(
    A: hl.SparseTensor,
    B: torch.Tensor,
    fmt0: hl.constexpr,
) -> torch.Tensor:
    """SPMM for Dense inner.  ``tile_k`` is 1-D so ``B[tile_k, tile_n]``
    is 2-D and the contraction collapses to a straight ``hl.dot``."""
    M = A.shape[0]
    N = B.size(1)
    C = torch.zeros(M, N, dtype=B.dtype, device=B.device)
    for tile_n in hl.tile(N):
        for tile_m in hl.sparse_tile(A, dim=0, levelformat=fmt0):
            acc = hl.zeros([tile_m.size(0), tile_n], dtype=B.dtype)
            for tile_k in hl.sparse_tile(tile_m, dim=1, levelformat="Dense"):
                a_val = tile_k.value  # (M, K) parent-inherited
                b_val = B[tile_k, tile_n]  # (K, N), tile_k is pure 1-D
                acc = hl.dot(a_val, b_val, acc=acc)
            C[tile_m, tile_n] = acc
    return C


# ----------------------------------------------------------------------------
# Dense-root contract demonstration.
#
# 3x3 fixture in which row 1 is logically empty (the user's intended math
# is ``[[1,2,3],[0,0,0],[4,5,6]]``) but storage at row 1 is filled with
# ``_GARBAGE`` — what a naive user leaves there when they mistakenly
# assume the format will "skip" that row:
#
#     stored = [[1,  2,  3 ],
#               [G,  G,  G ],     <- user thinks this is absent
#               [4,  5,  6 ]]
#
# We re-encode this same logical matrix under three root formats (Dense,
# Compressed, Bitmap) with a Dense inner, and observe:
#
#   Dense   root → no mask; row 1's G leaks into the output → WRONG.
#   Compressed root → row 1 is elided from storage; Dense inner sees only
#                     rows 0 and 2 → correct.
#   Bitmap  root → row 1's G lives in storage but the root bitmap masks
#                  it at load time → correct.
#
# The lesson: Dense is a user-to-compiler promise that "storage IS the
# math at this level."  Compressed/Bitmap ancestors can rescue a broken
# promise by pruning the garbage before Dense ever sees it.
# ----------------------------------------------------------------------------

_GARBAGE = 0.0
_CONTRACT_SHAPE = (3, 3)
_CONTRACT_N = 4
_CONTRACT_STORED = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [_GARBAGE, _GARBAGE, _GARBAGE],
        [4.0, 5.0, 6.0],
    ],
    device=DEVICE,
)
_CONTRACT_INTENDED = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [4.0, 5.0, 6.0],
    ],
    device=DEVICE,
)
_CONTRACT_B = (
    torch.arange(3 * _CONTRACT_N, dtype=torch.float32, device=DEVICE).reshape(
        3, _CONTRACT_N
    )
    * 0.1
)


class TestSparseTile2D(TestCase):
    def test_spmm_all_layouts(self) -> None:
        expected = _DENSE_A @ _B
        for fmt0 in _ROOT_FORMATS:
            for fmt1 in _INNER_FORMATS:
                with self.subTest(fmt0=fmt0, fmt1=fmt1):
                    A = _build_sparse_2d(fmt0, fmt1)
                    if fmt1 == "Dense":
                        got = spmm_dense_inner(A, _B, fmt0)
                        # hl.dot defaults to TF32 for fp32 inputs; allow the
                        # ~1e-2 matmul precision loss this introduces.
                        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
                    else:
                        got = spmm_sparse_inner(A, _B, fmt0, fmt1)
                        torch.testing.assert_close(got, expected)

    def test_dense_root_leaks_garbage(self) -> None:
        """Dense root has no mask; row 1's garbage leaks into the output."""
        A = hl.SparseTensor(
            values=_CONTRACT_STORED.flatten(),
            shape=_CONTRACT_SHAPE,
            ptrs=(None, None),
            coords=(None, None),
            bitmaps=(None, None),
        )
        got = spmm_dense_inner(A, _CONTRACT_B, "Dense")
        expected = _CONTRACT_INTENDED @ _CONTRACT_B
        self.assertFalse(torch.allclose(got, expected, rtol=1e-2, atol=1e-2))

    def test_compressed_root_elides_garbage_row(self) -> None:
        """Compressed root lists only rows 0 and 2; row 1's garbage is
        never in storage, so Dense inner can't leak it."""
        # nnz rows only: 0 and 2.  values = [row 0 | row 2] flattened.
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device=DEVICE)
        ptrs0 = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
        coords0 = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
        A = hl.SparseTensor(
            values=values,
            shape=_CONTRACT_SHAPE,
            ptrs=(ptrs0, None),
            coords=(coords0, None),
            bitmaps=(None, None),
        )
        got = spmm_dense_inner(A, _CONTRACT_B, "Compressed")
        expected = _CONTRACT_INTENDED @ _CONTRACT_B
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)

    def test_bitmap_root_masks_garbage_row(self) -> None:
        """Bitmap root keeps row 1 in storage (garbage and all) but the
        bitmap masks its load, so Dense inner accumulates zero for that
        row."""
        bitmap = torch.tensor([True, False, True], dtype=torch.bool, device=DEVICE)
        A = hl.SparseTensor(
            values=_CONTRACT_STORED.flatten(),
            shape=_CONTRACT_SHAPE,
            ptrs=(None, None),
            coords=(None, None),
            bitmaps=(bitmap, None),
        )
        got = spmm_dense_inner(A, _CONTRACT_B, "Bitmap")
        expected = _CONTRACT_INTENDED @ _CONTRACT_B
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
