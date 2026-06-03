from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Protocol, Sequence, TypeAlias, Union

import torch

import helion.language as hl

# Per-level indexing representation. The four ``csr_*`` variants each map to a
# dedicated ``LevelRep`` class. ``"csr"`` is accepted as a backward-compatible
# alias for ``"csr_compressed"`` only at the public API surface; internally a
# normalized value is always stored on ``LevelSpec``.
Encoding: TypeAlias = Literal[
    "coo",
    "csr_compressed",
    "csr_dense",
    "csr_jagged",
    "csr_padded",
]
EncodingInput: TypeAlias = Literal[
    "coo",
    "csr",
    "csr_compressed",
    "csr_dense",
    "csr_jagged",
    "csr_padded",
]

_CSR_ENCODINGS: frozenset[str] = frozenset(
    {"csr_compressed", "csr_dense", "csr_jagged", "csr_padded"}
)


@dataclass(frozen=True)
class FormatSpec:
    levels: tuple[LevelSpec, ...]

    def validate_against_shape(self, shape: tuple[int, ...]) -> None:
        seen: list[int] = []
        ndim = len(shape)

        for level in self.levels:
            for d in level.dims:
                if d < 0 or d >= ndim:
                    raise ValueError(f"dimension {d} out of range")
                seen.append(d)

        if sorted(seen) != list(range(ndim)):
            raise ValueError(
                f"level dims must partition all tensor dimensions exactly once; got {seen}"
            )


def _normalize_encoding(enc: EncodingInput) -> Encoding:
    """Map the public ``"csr"`` alias to ``"csr_compressed"``; pass others through."""
    if enc == "csr":
        return "csr_compressed"
    return enc  # type: ignore[return-value]


def _encoding_to_jagged_padded(enc: Encoding) -> tuple[bool, bool]:
    if enc == "csr_jagged":
        return True, False
    if enc == "csr_padded":
        return False, True
    return False, False


_LAYOUT_ALIASES: dict[str, tuple[Encoding, bool, bool]] = {
    "dense": ("csr_dense", False, False),
    "compressed": ("csr_compressed", False, False),
    "jagged": ("csr_jagged", True, False),
    "padded": ("csr_padded", False, True),
    "coo": ("coo", False, False),
}


def _parse_level_layout(
    layout: str, n_dims: int, level_i: int
) -> tuple[Encoding, bool, bool]:
    key = layout.strip().lower()
    if key not in _LAYOUT_ALIASES:
        raise ValueError(
            f"level {level_i}: unknown level_layout {layout!r}; "
            "expected one of: COO, Compressed, Dense, Jagged, Padded (any casing)"
        )
    enc, jg, pd = _LAYOUT_ALIASES[key]
    if n_dims == 1:
        if enc == "coo":
            raise ValueError(
                f"level {level_i}: single-dim (CSR) level cannot use layout 'COO'; "
                "use Dense, Compressed, Jagged, or Padded"
            )
    elif enc != "coo":
        raise ValueError(
            f"level {level_i}: multi-dim level must use layout 'COO', got {layout!r}"
        )
    return enc, jg, pd


def format_spec_from_level_lists(
    level_order: Sequence[Sequence[int]],
    shape: Sequence[int],
    block_size: Sequence[Sequence[int]],
    level_layout: Sequence[str],
    *,
    level_encoding: Sequence[EncodingInput] | None = None,
) -> FormatSpec:
    """
    Build a FormatSpec from parallel nested lists (one entry per tree level).

    Encoding selection (in order of precedence):

      1. An explicit value in ``level_encoding`` (``"csr"`` is normalized to
         ``"csr_compressed"``).
      2. Otherwise, ``level_layout[i]`` names the storage for this level:
         ``"Dense"``, ``"Compressed"``, ``"Jagged"``, or ``"Padded"`` for a
         single-dim (CSR) level; ``"COO"`` for a multi-dim COO level.
         Matching is case-insensitive (e.g. ``"compressed"`` is allowed).
    """
    n = len(level_order)
    if not (len(block_size) == n and len(level_layout) == n):
        raise ValueError(
            "level_order, block_size, and level_layout must have the same length"
        )
    if level_encoding is not None and len(level_encoding) != n:
        raise ValueError("level_encoding must match level_order length")

    levels: list[LevelSpec] = []
    for i, (dims, bs, layout) in enumerate(
        zip(level_order, block_size, level_layout, strict=True)
    ):
        dt = tuple(int(d) for d in dims)
        bt = tuple(int(b) for b in bs)
        sh = tuple(shape[d] for d in dims)
        if len(dt) != len(bt):
            raise ValueError(
                f"level {i}: dims and block_size must have the same length"
            )
        if level_encoding is not None:
            enc = _normalize_encoding(level_encoding[i])
            jg, pd = _encoding_to_jagged_padded(enc)
        else:
            enc, jg, pd = _parse_level_layout(layout, len(dt), i)
        levels.append(
            LevelSpec(
                dims=dt, shape=sh, block_size=bt, jagged=jg, padded=pd, encoding=enc
            )
        )

    return FormatSpec(levels=tuple(levels))


@dataclass(frozen=True)
class LevelSpec:
    dims: tuple[int, ...]
    shape: tuple[int, ...]
    block_size: tuple[int, ...]
    jagged: bool = False
    padded: bool = False
    encoding: Encoding = "coo"

    def __post_init__(self) -> None:
        if len(self.dims) != len(self.block_size):
            raise ValueError("dims and block_size must have same length")
        if len(self.dims) != len(self.shape):
            raise ValueError("dims and shape must have same length")
        if self.encoding in _CSR_ENCODINGS and len(self.dims) != 1:
            raise ValueError(
                f"{self.encoding} encoding requires exactly one dimension in dims"
            )


def _sort_coo_by_level_order(
    values: torch.Tensor,
    coords: torch.Tensor,
    level_order: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reorder COO columns so they are sorted lexicographically by coordinate rows in
    tree order: all dims of ``level_order[0]``, then ``level_order[1]``, etc.
    Within one level, dim order follows each inner sequence (any fixed order is valid).
    """
    nnz = int(values.shape[0])
    if nnz == 0:
        return values, coords

    keys: list[torch.Tensor] = []
    for dims in level_order:
        for d in dims:
            keys.append(coords[d])

    perm = torch.arange(nnz, device=coords.device, dtype=torch.long)
    for key in reversed(keys):
        perm = perm[torch.argsort(key[perm], stable=True)]

    return values[perm], coords[:, perm]


def _csr_parent_coord_rows(
    level_order: Sequence[Sequence[int]], level_index: int, d0: int
) -> tuple[int, ...]:
    """
    Logical dim indices for parent fibers: dims listed in ``level_order[0:level_index]``,
    excluding the CSR axis ``d0``, in first-seen order.
    """
    seen: set[int] = set()
    rows: list[int] = []
    for lev in range(level_index):
        for d in level_order[lev]:
            di = int(d)
            if di == d0 or di in seen:
                continue
            seen.add(di)
            rows.append(di)
    return tuple(rows)


def _assert_coo_layout(
    values: torch.Tensor,
    coords: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if values.dim() != 1:
        raise ValueError(f"values must be 1D (nnz,), got shape {tuple(values.shape)}")
    if coords.dim() != 2:
        raise ValueError(f"coords must be 2D (k, nnz), got shape {tuple(coords.shape)}")
    k, nnz = coords.shape
    if k != len(shape):
        raise ValueError(f"coords row count {k} must match len(shape)={len(shape)}")
    if values.shape[0] != nnz:
        raise ValueError(
            f"values length {values.shape[0]} must match coords columns {nnz}"
        )
    return coords.to(device=values.device, dtype=torch.long)


# -----------------------------
# Per-level representation
# -----------------------------
#
# Single dataclass keyed on ``spec.encoding``. ``ptrs`` and ``coords`` are
# present or ``None`` per the encoding's semantics:
#
#   encoding         | ptrs        | coords
#   -----------------|-------------|--------------------------------------
#   csr_compressed   | 1D (P+1,)   | 1D (nnz_at_level,)
#   csr_dense        | None        | None
#   csr_padded       | None        | 2D (n_parents, max_row_len), -1 pad
#   csr_jagged       | 1D (P+1,)   | None  (coords implicit 0..run_len-1)


@dataclass(frozen=True)
class LevelRep:
    spec: LevelSpec
    ptrs: torch.Tensor | None = None
    coords: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self._check()

    def _check(self) -> None:
        """Validate ``ptrs`` / ``coords`` presence + shape against ``spec.encoding``."""
        enc = self.spec.encoding
        has_ptrs = self.ptrs is not None
        has_coords = self.coords is not None

        expected: dict[str, tuple[bool, bool]] = {
            "csr_compressed": (True, True),
            "csr_dense": (False, False),
            "csr_padded": (False, True),
            "csr_jagged": (True, False),
        }
        if enc not in expected:
            raise ValueError(f"LevelRep: unsupported encoding {enc!r}")

        want_ptrs, want_coords = expected[enc]
        if has_ptrs != want_ptrs:
            raise ValueError(
                f"LevelRep[{enc}]: ptrs must be "
                f"{'a Tensor' if want_ptrs else 'None'}, got {type(self.ptrs).__name__}"
            )
        if has_coords != want_coords:
            raise ValueError(
                f"LevelRep[{enc}]: coords must be "
                f"{'a Tensor' if want_coords else 'None'}, got {type(self.coords).__name__}"
            )

        if self.ptrs is not None and self.ptrs.dim() != 1:
            raise ValueError(
                f"LevelRep[{enc}]: ptrs must be 1D, got shape {tuple(self.ptrs.shape)}"
            )
        if self.coords is not None:
            want_dim = 2 if enc == "csr_padded" else 1
            if self.coords.dim() != want_dim:
                raise ValueError(
                    f"LevelRep[{enc}]: coords must be {want_dim}D, "
                    f"got shape {tuple(self.coords.shape)}"
                )


# -----------------------------
# Bottom-up build
# -----------------------------
#
# Each ``build_<enc>`` takes the build state assembled so far (levels closer to
# the leaf + the current flat values) and the spec for the new level being
# added above. It may rewrite ``state.levels`` and ``state.values`` in place to
# satisfy the new top's invariants -- e.g. promoting a stack of padded reps to
# dense rows when a Dense level is laid above them -- and returns the new top
# ``LevelRep``. One O(n) scan-down per O(n) layer => O(n^2) total.


@dataclass
class BuildState:
    """Mutable carrier for the bottom-up build.

    ``coo`` is the original lex-sorted COO coords (k, nnz) and never changes.
    ``levels`` accumulates in bottom-up order; the driver reverses it once the
    loop completes. ``values`` may be replaced by a build step that forces a
    denser layout below.
    """

    coo: torch.Tensor
    dense_shape: tuple[int, ...]
    level_order: Sequence[Sequence[int]]
    values: torch.Tensor
    levels: list[LevelRep]
    fill_value: float

    def __post_init__(self):
        self.device = self.coo.device


def build_compressed(state: BuildState, level_index: int, spec: LevelSpec) -> LevelRep:
    """Build a ``csr_compressed`` top: ptrs + coords over the level below."""
    unq_parent, _csr_axis, _inverse, _counts, _d0 = _csr_setup(
        state.coo, level_index, spec, state.level_order
    )

    if unq_parent.numel() == 0:
        device = state.coo.device
        return LevelRep(
            spec,
            torch.zeros(1, device=device, dtype=torch.long),
            torch.zeros(0, device=device, dtype=torch.long),
        )

    level_coord = unq_parent[-1]
    if unq_parent.shape[0] == 1:
        ptrs = torch.tensor(
            [0, unq_parent.size(1)], device=state.device, dtype=torch.long
        )
    else:
        _, _, parent_counts = torch.unique_consecutive(
            unq_parent[:-1, :], dim=1, return_inverse=True, return_counts=True
        )
        ptrs = torch.zeros(
            parent_counts.numel() + 1, device=state.device, dtype=torch.long
        )
        ptrs[1:] = parent_counts.cumsum(0)

    return LevelRep(spec, ptrs, level_coord)


def _tile_size(rep: LevelRep) -> int:
    """Per-fiber fan-out for dense/padded levels (variable-fan-out raises)."""
    enc = rep.spec.encoding
    if enc == "csr_dense":
        return int(rep.spec.shape[0])
    if enc == "csr_padded":
        assert rep.coords is not None
        return int(rep.coords.shape[1])
    raise ValueError(f"_tile_size: variable fan-out encoding {enc!r}")


def _widen_padded(
    state: BuildState,
    idx: int,
    dense_pos: torch.Tensor,
    total_slots: int,
    tile: int,
) -> None:
    """Widen ``state.levels[idx]`` (a padded rep) so its coords have
    ``total_slots * tile`` parent rows, scattering existing rows to their new
    positions via the same ``dense_pos[old // tile] * tile + (old % tile)``
    mapping used for ptrs / values."""
    rep = state.levels[idx]
    assert rep.coords is not None
    old_coords = rep.coords
    max_rl = old_coords.shape[1]
    n_old = old_coords.shape[0]
    new_n = total_slots * tile
    new_coords = torch.full(
        (new_n, max_rl), -1, dtype=old_coords.dtype, device=state.device
    )
    orig_idx = torch.arange(n_old, device=state.device)
    new_idx = dense_pos[orig_idx // tile] * tile + (orig_idx % tile)
    new_coords[new_idx] = old_coords
    state.levels[idx] = LevelRep(rep.spec, None, new_coords)


def _scan_down_widen(
    state: BuildState, dense_pos: torch.Tensor, total_slots: int
) -> tuple[int, int]:
    """Walk down ``state.levels`` through dense/padded layers, widening passed
    padded coords along the way. Returns ``(target_idx, tile)`` where
    ``target_idx`` is the first ptr-bearing layer (or ``-1`` if values reached)
    and ``tile`` is the accumulated per-fiber fan-out from the new top down
    to that target."""
    tile = 1
    idx = len(state.levels) - 1
    while idx >= 0:
        rep = state.levels[idx]
        enc = rep.spec.encoding
        if enc in ("csr_compressed", "csr_jagged"):
            break
        if enc == "csr_padded":
            _widen_padded(state, idx, dense_pos, total_slots, tile)
        tile *= _tile_size(rep)
        idx -= 1
    return idx, tile


def _scatter_to_target(
    state: BuildState,
    target_idx: int,
    tile: int,
    dense_pos: torch.Tensor,
    n_fibers: int,
    total_slots: int,
    fill_value: float,
) -> None:
    """At the bottom of the scan, rewrite the target ptr tensor (inserting
    empty fibers at missing slots) or ``state.values`` (filling missing slots
    with ``fill_value``). Empty fiber insertion is a duplicated ptr boundary."""
    orig_n = n_fibers * tile
    target_n = total_slots * tile
    orig_idx = torch.arange(orig_n, device=state.device)
    new_pos = dense_pos[orig_idx // tile] * tile + (orig_idx % tile)

    if target_idx < 0:
        new_values = torch.full(
            (target_n,),
            fill_value,
            dtype=state.values.dtype,
            device=state.values.device,
        )
        new_values[new_pos] = state.values
        state.values = new_values
    else:
        tgt = state.levels[target_idx]
        assert tgt.ptrs is not None
        orig_lens = tgt.ptrs[1:] - tgt.ptrs[:-1]
        new_lens = torch.zeros(target_n, dtype=tgt.ptrs.dtype, device=state.device)
        new_lens[new_pos] = orig_lens
        new_ptrs = torch.zeros(target_n + 1, dtype=tgt.ptrs.dtype, device=state.device)
        new_ptrs[1:] = new_lens.cumsum(0)
        state.levels[target_idx] = LevelRep(tgt.spec, new_ptrs, tgt.coords)


def build_dense(state: BuildState, level_index: int, spec: LevelSpec) -> LevelRep:
    """Build a ``csr_dense`` top.

    Strategy: compute what the compressed coords would be at this level
    (``_csr_setup``), then for each missing slot in ``[0, n_outer * extent)``
    insert ``tile`` empty entries below. The scan-down widens passed padded
    coords too (see ``_widen_padded``) and finally hits a ptr-bearing level
    or ``state.values``.
    """
    unq_parent, _csr_axis, _inverse, _counts, d0 = _csr_setup(
        state.coo, level_index, spec, state.level_order
    )
    dense_extent = int(state.dense_shape[d0])

    n_fibers = int(unq_parent.shape[1])
    if n_fibers == 0:
        return LevelRep(spec, None, None)

    if unq_parent.shape[0] > 1:
        outer = unq_parent[:-1, :]
        _, outer_inv = torch.unique_consecutive(outer, dim=1, return_inverse=True)
        n_outer = int(outer_inv.max().item()) + 1
        dense_pos = outer_inv.long() * dense_extent + unq_parent[-1, :].long()
    else:
        n_outer = 1
        dense_pos = unq_parent[0, :].long()

    total_slots = n_outer * dense_extent

    target_idx, tile = _scan_down_widen(state, dense_pos, total_slots)
    _scatter_to_target(
        state,
        target_idx,
        tile,
        dense_pos,
        n_fibers,
        total_slots,
        fill_value=state.fill_value,
    )

    return LevelRep(spec, None, None)


def build_padded(state: BuildState, level_index: int, spec: LevelSpec) -> LevelRep:
    """Build a ``csr_padded`` top.

    Each parent fiber is padded to ``max_row_len`` (= max distinct-d0 count
    over parents). Pads are ``-1`` in ``coords`` (shape
    ``(n_parents, max_row_len)``). Same scan-down as ``build_dense``; only the
    coordinate scheme differs (local position within the parent rather than
    the d0 coord itself).
    """
    unq_parent, _csr_axis, _inverse, _counts, _d0 = _csr_setup(
        state.coo, level_index, spec, state.level_order
    )

    n_fibers = int(unq_parent.shape[1])
    if n_fibers == 0:
        return LevelRep(
            spec,
            None,
            torch.full((0, 1), -1, dtype=torch.long, device=state.device),
        )

    if unq_parent.shape[0] > 1:
        outer = unq_parent[:-1, :]
        _, outer_inv, counts_pp = torch.unique_consecutive(
            outer, dim=1, return_inverse=True, return_counts=True
        )
        n_parents = int(counts_pp.numel())
    else:
        outer_inv = torch.zeros(n_fibers, dtype=torch.long, device=state.device)
        counts_pp = torch.tensor([n_fibers], dtype=torch.long, device=state.device)
        n_parents = 1

    max_row_len = int(counts_pp.max().item())
    group_starts = torch.cumsum(counts_pp, dim=0) - counts_pp
    local_idx = torch.arange(n_fibers, device=state.device) - group_starts[outer_inv]
    dense_pos = outer_inv.long() * max_row_len + local_idx
    total_slots = n_parents * max_row_len

    target_idx, tile = _scan_down_widen(state, dense_pos, total_slots)
    _scatter_to_target(
        state,
        target_idx,
        tile,
        dense_pos,
        n_fibers,
        total_slots,
        fill_value=state.fill_value,
    )

    coords_2d = torch.full(
        (n_parents, max_row_len), -1, dtype=torch.long, device=state.device
    )
    coords_2d.view(-1)[dense_pos] = unq_parent[-1, :].long()

    return LevelRep(spec, None, coords_2d)


def build_jagged(state: BuildState, level_index: int, spec: LevelSpec) -> LevelRep:
    """Build a ``csr_jagged`` top.

    Each parent fiber fills ``[0, last_d0 + 1)``; coords are implicit (the
    in-row position is the d0 coord itself). Same scan-down as
    ``build_dense`` / ``build_padded`` -- widens passed padded layers, then
    modifies the first ptr-bearing level (or values) below.

    Jagged is itself ptr-bearing, so upper builders treat it as a scan-stopper
    (handled by the existing branch in ``_scan_down_widen``).
    """
    unq_parent, _csr_axis, _inverse, _counts, _d0 = _csr_setup(
        state.coo, level_index, spec, state.level_order
    )

    n_fibers = int(unq_parent.shape[1])
    if n_fibers == 0:
        return LevelRep(
            spec, torch.zeros(1, dtype=torch.long, device=state.device), None
        )

    if unq_parent.shape[0] > 1:
        outer = unq_parent[:-1, :]
        _, outer_inv, counts_pp = torch.unique_consecutive(
            outer, dim=1, return_inverse=True, return_counts=True
        )
        n_parents = int(counts_pp.numel())
    else:
        outer_inv = torch.zeros(n_fibers, dtype=torch.long, device=state.device)
        counts_pp = torch.tensor([n_fibers], dtype=torch.long, device=state.device)
        n_parents = 1

    d0_coord = unq_parent[-1, :].long()

    # widths[p] = last_d0 + 1. d0 coords are sorted within each parent's run
    # (lex sort on outer + d0), so the max is at the parent's last fiber.
    group_ends = torch.cumsum(counts_pp, 0) - 1
    widths = d0_coord[group_ends] + 1
    total_slots = int(widths.sum().item())

    ptrs = torch.zeros(n_parents + 1, dtype=torch.long, device=state.device)
    ptrs[1:] = torch.cumsum(widths, 0)
    ptr_start = ptrs[:-1]

    dense_pos = ptr_start[outer_inv] + d0_coord

    target_idx, tile = _scan_down_widen(state, dense_pos, total_slots)
    _scatter_to_target(
        state,
        target_idx,
        tile,
        dense_pos,
        n_fibers,
        total_slots,
        fill_value=state.fill_value,
    )

    return LevelRep(spec, ptrs, None)


_BUILDERS = {
    "csr_compressed": build_compressed,
    "csr_dense": build_dense,
    "csr_padded": build_padded,
    "csr_jagged": build_jagged,
}


def _csr_setup(
    coords: torch.Tensor,
    level_index: int,
    level: LevelSpec,
    level_order: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Group lex-sorted COO columns into CSR fibers at this level.

    Returns ``(unq_parent, csr_axis, inverse, counts, d0)`` where ``d0`` is the
    represented dim. Fibers are ``unique_consecutive`` runs on parent coords from
    outer levels (if any), else consecutive runs of equal ``coords[d0]``.
    """
    d0 = int(level.dims[0])
    csr_axis = coords[d0]
    parent_rows = _csr_parent_coord_rows(level_order, level_index, d0)

    if csr_axis.numel() == 0:
        device, dtype = coords.device, coords.dtype
        unq_parent = torch.zeros((len(parent_rows), 0), device=device, dtype=dtype)
        inverse = torch.zeros((0,), device=device, dtype=torch.long)
        counts = torch.zeros((0,), device=device, dtype=torch.long)
        return unq_parent, csr_axis, inverse, counts, d0

    if parent_rows:
        rows = parent_rows + (d0,)
        csr_axis = coords[list(rows), :]
        unq_parent, inverse, counts = torch.unique_consecutive(
            csr_axis, dim=1, return_inverse=True, return_counts=True
        )
    else:
        unq_parent, inverse, counts = torch.unique_consecutive(
            csr_axis.unsqueeze(0), dim=1, return_inverse=True, return_counts=True
        )

    return unq_parent, csr_axis, inverse, counts, d0


def sparse_convert(
    values: torch.Tensor,
    coords: torch.Tensor,
    shape: tuple[int, ...],
    level_order: Sequence[Sequence[int]],
    block_size: Sequence[Sequence[int]],
    level_layout: Sequence[str],
    *,
    fill_value: float = 0.0,
    level_encoding: Sequence[EncodingInput] | None = None,
) -> hl.SparseTensor:
    spec = format_spec_from_level_lists(
        level_order, shape, block_size, level_layout, level_encoding=level_encoding
    )
    spec.validate_against_shape(shape)
    coords = _assert_coo_layout(values, coords, shape)
    values, coords = _sort_coo_by_level_order(values, coords, level_order)
    dense_shape = tuple(int(s) for s in shape)

    state = BuildState(
        coo=coords,
        dense_shape=dense_shape,
        level_order=level_order,
        values=values,
        levels=[],
        fill_value=fill_value,
    )

    # Bottom-up: leaf-most level first, root last. Each builder may rewrite
    # state.levels and state.values to align with the new top's layout.
    for i in range(len(spec.levels) - 1, -1, -1):
        level = spec.levels[i]
        builder = _BUILDERS.get(level.encoding)
        if builder is None:
            raise ValueError(f"unknown encoding {level.encoding!r}")
        state.levels.append(builder(state, i, level))

    levels = tuple(reversed(state.levels))
    ptrs = tuple(level.ptrs for level in levels)
    coords = tuple(level.coords for level in levels)
    bitmaps = tuple(None for _ in levels)
    return hl.SparseTensor(
        values=state.values,
        shape=dense_shape,
        ptrs=ptrs,
        coords=coords,
        bitmaps=bitmaps,
    )
