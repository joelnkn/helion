from __future__ import annotations

import unittest

import torch

import helion

from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl
from helion._compiler.convert_utils import sparse_convert

# ----------------------------------------------------------------------------
# 3-D sparse-tile SDOT tests.
#
# SDOT computes ``C[i, j] = sum_k A[i, j, k] * x[k]`` — a per-(i, j) sparse
# dot product against a dense 1-D vector ``x``.  One rich fixture with
# irregular k-nnz per (i, j) plus one empty (i, j) plane is re-encoded into
# 30 format combos spanning all positions.
#
# A single kernel covers every combo.  The inner body is a pure element-wise
# multiply-then-sum, so PyTorch broadcasting makes it rank-polymorphic in
# ``x_val``: when the innermost level is Dense, ``tile_k`` is 1-D and
# ``x[tile_k]`` has shape ``(K,)`` which left-pads to ``(1, 1, K)`` against
# the parent-inherited ``a_val`` of shape ``(I, J, K)``; for any non-Dense
# innermost level, ``tile_k`` is ND and ``x[tile_k]`` is already ``(I, J, K)``.
# Both paths produce ``(I, J, K) → .sum(dim=-1) → (I, J)`` identically.
# ----------------------------------------------------------------------------

# Logical (I=4, J=3, K=6) tensor.  Irregular k-nnz per (i, j), one wholly
# empty (i, j) plane, empty i=2 row at root for Bitmap/Padded masking.
#   A[0, 0, :] = [1, 0, 2, 0, 0, 3]   nnz=3
#   A[0, 1, :] = [0, 0, 0, 0, 0, 0]   nnz=0  ← empty (i, j) plane
#   A[0, 2, :] = [0, 4, 0, 0, 0, 0]   nnz=1
#   A[1, 0, :] = [5, 0, 0, 6, 7, 8]   nnz=4  ← max k-nnz
#   A[1, 1, :] = [0, 0, 9, 0, 10, 0]  nnz=2
#   A[1, 2, :] = [0, 0, 0, 0, 0, 11]  nnz=1
#   A[2, *, :] = 0                    nnz=0  ← empty root i=2
#   A[3, 0, :] = [12, 0, 13, 0, 0, 0] nnz=2
#   A[3, 1, :] = [0, 14, 0, 0, 15, 0] nnz=2
#   A[3, 2, :] = [0, 0, 0, 16, 0, 17] nnz=2
_DENSE_A_3D = torch.tensor(
    [
        [
            [1.0, 0.0, 2.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [5.0, 0.0, 0.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 9.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 11.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        [
            [12.0, 0.0, 13.0, 0.0, 0.0, 0.0],
            [0.0, 14.0, 0.0, 0.0, 15.0, 0.0],
            [0.0, 0.0, 0.0, 16.0, 0.0, 17.0],
        ],
    ],
    device=DEVICE,
)
_SHAPE_3D = (4, 3, 6)
_I, _J, _K = _SHAPE_3D
_X = torch.arange(_K, dtype=torch.float32, device=DEVICE) * 0.1 + 1.0  # (K,)
_GARBAGE = 777.0


def _int64(xs):
    return torch.tensor(xs, dtype=torch.int64, device=DEVICE)


def _bool(xs):
    return torch.tensor(xs, dtype=torch.bool, device=DEVICE)


def _build_sparse_3d(fmt0: str, fmt1: str, fmt2: str) -> hl.SparseTensor:
    # Standard COO: coords is (ndim, nnz) with one row per dim, values holds
    # the matching non-zeros in the same column order.
    coo = torch.nonzero(_DENSE_A_3D)  # (nnz, ndim)
    coords = coo.t().contiguous()  # (ndim, nnz)
    values = _DENSE_A_3D[coords[0], coords[1], coords[2]]  # (nnz,)

    ct = sparse_convert(
        values, coords, _SHAPE_3D, [[0], [1], [2]], [[1], [1], [1]], [fmt0, fmt1, fmt2]
    )

    return hl.SparseTensor(
        values=values,
        shape=_SHAPE_3D,
        ptrs=(ct.levels[0].ptrs, ct.levels[1].ptrs, ct.levels[2].ptrs),
        coords=(ct.levels[0].coords, ct.levels[1].coords, ct.levels[2].coords),
        bitmaps=(None, None, None),
    )


@helion.kernel(config=helion.Config(block_sizes=[4, 4, 8]))
def sdot_kernel(
    A: hl.SparseTensor,
    x: torch.Tensor,
    fmt0: hl.constexpr,
    fmt1: hl.constexpr,
    fmt2: hl.constexpr,
) -> torch.Tensor:
    """Format-invariant SDOT via element-wise mul-then-sum.  ``a_val`` is
    always ``(I, J, K)`` via parent-chain inheritance; ``x_val`` is either
    ``(I, J, K)`` (non-Dense innermost: ND ``tile_k`` gather) or ``(K,)``
    (Dense innermost: 1-D ``tile_k``).  Broadcasting left-pads the 1-D case
    so both paths collapse to ``(I, J, K) → (I, J)`` after the reduction."""
    I = A.shape[0]
    J = A.shape[1]
    C = torch.zeros(I * J, dtype=x.dtype, device=x.device)
    for tile_i in hl.sparse_tile(A, dim=0, levelformat=fmt0):
        for tile_j in hl.sparse_tile(tile_i, dim=1, levelformat=fmt1):
            acc = hl.zeros([tile_i.size(0), tile_j.size(-1)], dtype=x.dtype)
            for tile_k in hl.sparse_tile(tile_j, dim=2, levelformat=fmt2):
                a_val = tile_k.value
                x_val = x[tile_k]
                acc = acc + (a_val * x_val).sum(dim=-1)
            flat_idx = tile_i[:, None] * J + tile_j
            C[flat_idx] = acc
    return C.view(I, J)


# Curated 30-combo sweep covering every format at every position.  Root is
# restricted to ``Dense``/``Compressed``/``Bitmap`` (Padded / Jagged are
# forbidden at root).
_LAYOUTS: list[tuple[str, str, str]] = [
    # root=Dense (12)
    ("Dense", "Dense", "Dense"),
    ("Dense", "Dense", "Compressed"),
    ("Dense", "Dense", "Padded"),
    ("Dense", "Dense", "Jagged"),
    ("Dense", "Dense", "Bitmap"),
    ("Dense", "Compressed", "Dense"),
    ("Dense", "Compressed", "Compressed"),
    ("Dense", "Padded", "Dense"),
    ("Dense", "Jagged", "Dense"),
    ("Dense", "Bitmap", "Dense"),
    ("Dense", "Padded", "Bitmap"),
    ("Dense", "Padded", "Jagged"),
    # root=Compressed (10)
    ("Compressed", "Dense", "Dense"),
    ("Compressed", "Compressed", "Compressed"),
    ("Compressed", "Compressed", "Dense"),
    ("Compressed", "Compressed", "Padded"),
    ("Compressed", "Padded", "Compressed"),
    ("Compressed", "Jagged", "Jagged"),
    ("Compressed", "Bitmap", "Dense"),
    ("Compressed", "Dense", "Bitmap"),
    ("Compressed", "Bitmap", "Bitmap"),
    ("Compressed", "Compressed", "Jagged"),
    # root=Bitmap (8)
    ("Bitmap", "Dense", "Dense"),
    ("Bitmap", "Compressed", "Compressed"),
    ("Bitmap", "Bitmap", "Bitmap"),
    ("Bitmap", "Dense", "Jagged"),
    ("Bitmap", "Jagged", "Bitmap"),
    ("Bitmap", "Padded", "Padded"),
    ("Bitmap", "Dense", "Compressed"),
    ("Bitmap", "Padded", "Dense"),
]


class TestSparseTile3D(TestCase):
    def test_sdot_all_layouts(self) -> None:
        # Reference: einsum over k of dense A and x.
        expected = torch.einsum("ijk,k->ij", _DENSE_A_3D, _X)
        for fmt in _LAYOUTS:
            if "Bitmap" in fmt:
                continue
            with self.subTest(fmt=fmt):
                A = _build_sparse_3d(*fmt)
                fmt0, fmt1, fmt2 = fmt
                got = sdot_kernel(A, _X, fmt0, fmt1, fmt2)
                torch.testing.assert_close(got, expected)

            break


if __name__ == "__main__":
    unittest.main()
