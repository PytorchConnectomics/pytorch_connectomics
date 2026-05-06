"""Algorithm kernels used by segmentation decoders."""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import fastremap
import numpy as np
from scipy import ndimage

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)


try:
    import edt

    EDT_AVAILABLE = True
except ImportError:
    EDT_AVAILABLE = False


try:
    import cupy as _cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    _cp = None


def compute_edt(
    foreground_mask: np.ndarray,
    use_fast_edt: bool = True,
    edt_parallel: int = 4,
    edt_anisotropy: Optional[Tuple[float, ...]] = None,
    edt_downsample_factor: int = 1,
) -> np.ndarray:
    """Compute Euclidean Distance Transform on a foreground mask."""
    resolved_anisotropy = edt_anisotropy

    def _edt_on_mask(mask: np.ndarray, anisotropy: Optional[Tuple[float, ...]]) -> np.ndarray:
        if use_fast_edt and EDT_AVAILABLE:
            ani = anisotropy if anisotropy is not None else tuple(1.0 for _ in range(mask.ndim))
            return edt.edt(
                mask.astype(np.uint8),
                anisotropy=ani,
                black_border=True,
                parallel=edt_parallel,
            )

        if use_fast_edt and not EDT_AVAILABLE:
            warnings.warn(
                "Fast edt library not available. Using scipy.ndimage (slower). "
                "Install edt for 10-50x speedup: pip install edt",
                UserWarning,
            )

        if anisotropy is not None:
            return ndimage.distance_transform_edt(mask, sampling=anisotropy)
        return ndimage.distance_transform_edt(mask)

    if edt_downsample_factor > 1:
        zoom_factors = tuple(1.0 / edt_downsample_factor for _ in range(foreground_mask.ndim))
        foreground_ds = ndimage.zoom(
            foreground_mask.astype(np.float32), zoom_factors, order=0
        ).astype(bool)

        if resolved_anisotropy is not None:
            ds_anisotropy = tuple(a / edt_downsample_factor for a in resolved_anisotropy)
        else:
            ds_anisotropy = None

        distance_ds = _edt_on_mask(foreground_ds, ds_anisotropy)
        distance_fg = ndimage.zoom(
            distance_ds,
            tuple(edt_downsample_factor for _ in range(distance_ds.ndim)),
            order=1,
        )
        distance_fg *= edt_downsample_factor

        if distance_fg.shape != foreground_mask.shape:
            slices = tuple(
                slice(0, min(s1, s2)) for s1, s2 in zip(distance_fg.shape, foreground_mask.shape)
            )
            corrected = np.zeros(foreground_mask.shape, dtype=distance_fg.dtype)
            corrected[slices] = distance_fg[slices]
            distance_fg = corrected

        return distance_fg

    return _edt_on_mask(foreground_mask, resolved_anisotropy)


@jit(nopython=True, cache=True)
def connected_components_affinity_3d_numba(
    hard_aff: np.ndarray, edge_offset: int = 0
) -> np.ndarray:
    """Numba-accelerated connected components from 3D affinities."""
    visited = np.zeros(hard_aff.shape[1:], dtype=np.uint8)
    seg = np.zeros(hard_aff.shape[1:], dtype=np.uint32)
    cur_id = 1
    max_stack = visited.shape[0] * visited.shape[1] * visited.shape[2]
    stack_x = np.zeros(max_stack, dtype=np.int32)
    stack_y = np.zeros(max_stack, dtype=np.int32)
    stack_z = np.zeros(max_stack, dtype=np.int32)

    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            for k in range(visited.shape[2]):
                if hard_aff[:, i, j, k].any() and not visited[i, j, k]:
                    stack_size = 0

                    stack_x[stack_size] = i
                    stack_y[stack_size] = j
                    stack_z[stack_size] = k
                    stack_size += 1
                    visited[i, j, k] = True

                    while stack_size > 0:
                        stack_size -= 1
                        x = stack_x[stack_size]
                        y = stack_y[stack_size]
                        z = stack_z[stack_size]

                        seg[x, y, z] = cur_id

                        if (
                            x + 1 < visited.shape[0]
                            and hard_aff[0, x + edge_offset, y, z]
                            and not visited[x + 1, y, z]
                        ):
                            stack_x[stack_size] = x + 1
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x + 1, y, z] = True

                        if (
                            y + 1 < visited.shape[1]
                            and hard_aff[1, x, y + edge_offset, z]
                            and not visited[x, y + 1, z]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y + 1
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x, y + 1, z] = True

                        if (
                            z + 1 < visited.shape[2]
                            and hard_aff[2, x, y, z + edge_offset]
                            and not visited[x, y, z + 1]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z + 1
                            stack_size += 1
                            visited[x, y, z + 1] = True

                        if (
                            x - 1 >= 0
                            and hard_aff[0, x - 1 + edge_offset, y, z]
                            and not visited[x - 1, y, z]
                        ):
                            stack_x[stack_size] = x - 1
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x - 1, y, z] = True

                        if (
                            y - 1 >= 0
                            and hard_aff[1, x, y - 1 + edge_offset, z]
                            and not visited[x, y - 1, z]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y - 1
                            stack_z[stack_size] = z
                            stack_size += 1
                            visited[x, y - 1, z] = True

                        if (
                            z - 1 >= 0
                            and hard_aff[2, x, y, z - 1 + edge_offset]
                            and not visited[x, y, z - 1]
                        ):
                            stack_x[stack_size] = x
                            stack_y[stack_size] = y
                            stack_z[stack_size] = z - 1
                            stack_size += 1
                            visited[x, y, z - 1] = True

                    cur_id += 1

    return seg


def _affinity_foreground_mask(hard_aff: np.ndarray, edge_offset: int) -> np.ndarray:
    """Voxels labeled by the serial DFS: seeds + endpoints of in-bounds edges.

    The serial kernel seeds any voxel with ``hard_aff[:, v].any()`` true (even when
    every implied edge points out of bounds → singleton component) and additionally
    floods through valid edges. Both classes must be foreground here so the
    parallel sweep gives them a unique propagated label.
    """
    X, Y, Z = hard_aff.shape[1], hard_aff.shape[2], hard_aff.shape[3]
    fg = hard_aff[:3].any(axis=0).copy()
    if edge_offset == 0:
        ax0 = hard_aff[0, : X - 1, :, :]
        ax1 = hard_aff[1, :, : Y - 1, :]
        ax2 = hard_aff[2, :, :, : Z - 1]
    else:
        ax0 = hard_aff[0, 1:X, :, :]
        ax1 = hard_aff[1, :, 1:Y, :]
        ax2 = hard_aff[2, :, :, 1:Z]
    fg[: X - 1, :, :] |= ax0
    fg[1:X, :, :] |= ax0
    fg[:, : Y - 1, :] |= ax1
    fg[:, 1:Y, :] |= ax1
    fg[:, :, : Z - 1] |= ax2
    fg[:, :, 1:Z] |= ax2
    return fg


@jit(nopython=True, parallel=True, cache=True)
def _propagate_affinity_sweep(seg, hard_aff, edge_offset):
    """One outer iteration of axis-aligned forward+backward min-label sweeps.

    Each prange iterates over an axis-orthogonal column; serial scan along the
    axis owned by the column makes the work race-free without atomics.
    """
    X = seg.shape[0]
    Y = seg.shape[1]
    Z = seg.shape[2]
    changes = 0

    for jk in prange(Y * Z):
        j = jk // Z
        k = jk - j * Z
        for i in range(X - 1):
            if hard_aff[0, i + edge_offset, j, k]:
                a = seg[i, j, k]
                b = seg[i + 1, j, k]
                if a < b:
                    seg[i + 1, j, k] = a
                    changes += 1
                elif b < a:
                    seg[i, j, k] = b
                    changes += 1
        for i in range(X - 2, -1, -1):
            if hard_aff[0, i + edge_offset, j, k]:
                a = seg[i, j, k]
                b = seg[i + 1, j, k]
                if a < b:
                    seg[i + 1, j, k] = a
                    changes += 1
                elif b < a:
                    seg[i, j, k] = b
                    changes += 1

    for ik in prange(X * Z):
        i = ik // Z
        k = ik - i * Z
        for j in range(Y - 1):
            if hard_aff[1, i, j + edge_offset, k]:
                a = seg[i, j, k]
                b = seg[i, j + 1, k]
                if a < b:
                    seg[i, j + 1, k] = a
                    changes += 1
                elif b < a:
                    seg[i, j, k] = b
                    changes += 1
        for j in range(Y - 2, -1, -1):
            if hard_aff[1, i, j + edge_offset, k]:
                a = seg[i, j, k]
                b = seg[i, j + 1, k]
                if a < b:
                    seg[i, j + 1, k] = a
                    changes += 1
                elif b < a:
                    seg[i, j, k] = b
                    changes += 1

    for ij in prange(X * Y):
        i = ij // Y
        j = ij - i * Y
        for k in range(Z - 1):
            if hard_aff[2, i, j, k + edge_offset]:
                a = seg[i, j, k]
                b = seg[i, j, k + 1]
                if a < b:
                    seg[i, j, k + 1] = a
                    changes += 1
                elif b < a:
                    seg[i, j, k] = b
                    changes += 1
        for k in range(Z - 2, -1, -1):
            if hard_aff[2, i, j, k + edge_offset]:
                a = seg[i, j, k]
                b = seg[i, j, k + 1]
                if a < b:
                    seg[i, j, k + 1] = a
                    changes += 1
                elif b < a:
                    seg[i, j, k] = b
                    changes += 1

    return changes


@jit(nopython=True, cache=True)
def _compact_affinity_labels(flat_in: np.ndarray) -> np.ndarray:
    """Map sparse uint64 labels in scan order to dense uint32 1..K (0 stays 0)."""
    out = np.zeros(flat_in.size, dtype=np.uint32)
    remap = {np.uint64(0): np.uint32(0)}
    next_id = np.uint32(0)
    for i in range(flat_in.size):
        v = flat_in[i]
        if v in remap:
            out[i] = remap[v]
        else:
            next_id += np.uint32(1)
            remap[v] = next_id
            out[i] = next_id
    return out


def connected_components_affinity_3d_numba_parallel(
    hard_aff: np.ndarray, edge_offset: int = 0, max_iters: int = 256
) -> np.ndarray:
    """Parallel-numba CC over a 3D affinity graph.

    Bit-exact match with :func:`connected_components_affinity_3d_numba` because
    the affinity graph is undirected (each edge is stored once at the
    lower-coordinate voxel for ``edge_offset=0`` or higher for ``edge_offset=1``)
    and both kernels label components by the scan-order rank of each component's
    lex-smallest member.
    """
    edge_offset = int(edge_offset)
    if edge_offset not in (0, 1):
        raise ValueError(f"edge_offset must be 0 or 1, got {edge_offset}")
    if hard_aff.ndim != 4:
        raise ValueError(f"hard_aff must be 4D, got {hard_aff.ndim}D")
    if hard_aff.shape[0] < 3:
        raise ValueError(f"hard_aff must have >=3 channels, got {hard_aff.shape[0]}")

    hard_aff = np.ascontiguousarray(hard_aff[:3])
    X, Y, Z = hard_aff.shape[1], hard_aff.shape[2], hard_aff.shape[3]

    fg = _affinity_foreground_mask(hard_aff, edge_offset)
    if not fg.any():
        return np.zeros((X, Y, Z), dtype=np.uint32)

    fg_flat = fg.reshape(-1)
    n_fg = int(fg_flat.sum())
    labels_flat = np.zeros(fg_flat.shape, dtype=np.uint64)
    labels_flat[fg_flat] = np.arange(1, n_fg + 1, dtype=np.uint64)
    seg = labels_flat.reshape((X, Y, Z))

    converged = False
    for _ in range(max_iters):
        if _propagate_affinity_sweep(seg, hard_aff, edge_offset) == 0:
            converged = True
            break
    if not converged:
        warnings.warn(
            f"connected_components_affinity_3d_numba_parallel: did not converge in "
            f"{max_iters} sweeps; result may be incorrect for unusually long components.",
            RuntimeWarning,
        )

    return _compact_affinity_labels(seg.reshape(-1)).reshape((X, Y, Z))


_AFFINITY_CC_SWEEP_KERNELS: dict = {}

_LABEL_CTYPE = {
    "uint32": "unsigned int",
    "uint64": "unsigned long long",
}


def _build_affinity_cc_sweep_kernels(label_key: str):
    """Per-axis CUDA RawKernels that do forward+backward min sweep within a column.

    One thread per axis-orthogonal column scans the column serially in both
    directions; one outer iteration thus fully propagates labels along each
    axis line, matching the numba parallel sweep (O(turns) iterations rather
    than O(diameter)).

    ``label_key`` selects the seg array C type (``"uint32"`` or ``"uint64"``);
    kernels are cached per type.
    """
    cached = _AFFINITY_CC_SWEEP_KERNELS.get(label_key)
    if cached is not None:
        return cached
    cp = _cp

    src = r"""
extern "C" __global__
void sweep_axis0(LABEL_T* seg, const unsigned char* aff,
                 int X, int Y, int Z, int edge_offset, int* changed) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= Y || k >= Z) return;
    long long YZ = (long long)Y * Z;
    int local = 0;
    for (int i = 0; i < X - 1; ++i) {
        long long aff_idx = (long long)(i + edge_offset) * YZ + (long long)j * Z + k;
        if (aff[aff_idx]) {
            long long idx_lo = (long long)i * YZ + (long long)j * Z + k;
            long long idx_hi = (long long)(i + 1) * YZ + (long long)j * Z + k;
            LABEL_T a = seg[idx_lo];
            LABEL_T b = seg[idx_hi];
            if (a < b) { seg[idx_hi] = a; local = 1; }
            else if (b < a) { seg[idx_lo] = b; local = 1; }
        }
    }
    for (int i = X - 2; i >= 0; --i) {
        long long aff_idx = (long long)(i + edge_offset) * YZ + (long long)j * Z + k;
        if (aff[aff_idx]) {
            long long idx_lo = (long long)i * YZ + (long long)j * Z + k;
            long long idx_hi = (long long)(i + 1) * YZ + (long long)j * Z + k;
            LABEL_T a = seg[idx_lo];
            LABEL_T b = seg[idx_hi];
            if (a < b) { seg[idx_hi] = a; local = 1; }
            else if (b < a) { seg[idx_lo] = b; local = 1; }
        }
    }
    if (local) *changed = 1;
}

extern "C" __global__
void sweep_axis1(LABEL_T* seg, const unsigned char* aff,
                 int X, int Y, int Z, int edge_offset, int* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= X || k >= Z) return;
    long long YZ = (long long)Y * Z;
    int local = 0;
    for (int j = 0; j < Y - 1; ++j) {
        long long aff_idx = (long long)i * YZ + (long long)(j + edge_offset) * Z + k;
        if (aff[aff_idx]) {
            long long idx_lo = (long long)i * YZ + (long long)j * Z + k;
            long long idx_hi = (long long)i * YZ + (long long)(j + 1) * Z + k;
            LABEL_T a = seg[idx_lo];
            LABEL_T b = seg[idx_hi];
            if (a < b) { seg[idx_hi] = a; local = 1; }
            else if (b < a) { seg[idx_lo] = b; local = 1; }
        }
    }
    for (int j = Y - 2; j >= 0; --j) {
        long long aff_idx = (long long)i * YZ + (long long)(j + edge_offset) * Z + k;
        if (aff[aff_idx]) {
            long long idx_lo = (long long)i * YZ + (long long)j * Z + k;
            long long idx_hi = (long long)i * YZ + (long long)(j + 1) * Z + k;
            LABEL_T a = seg[idx_lo];
            LABEL_T b = seg[idx_hi];
            if (a < b) { seg[idx_hi] = a; local = 1; }
            else if (b < a) { seg[idx_lo] = b; local = 1; }
        }
    }
    if (local) *changed = 1;
}

extern "C" __global__
void sweep_axis2(LABEL_T* seg, const unsigned char* aff,
                 int X, int Y, int Z, int edge_offset, int* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= X || j >= Y) return;
    long long YZ = (long long)Y * Z;
    long long base = (long long)i * YZ + (long long)j * Z;
    int local = 0;
    for (int k = 0; k < Z - 1; ++k) {
        long long aff_idx = base + (k + edge_offset);
        if (aff[aff_idx]) {
            LABEL_T a = seg[base + k];
            LABEL_T b = seg[base + k + 1];
            if (a < b) { seg[base + k + 1] = a; local = 1; }
            else if (b < a) { seg[base + k] = b; local = 1; }
        }
    }
    for (int k = Z - 2; k >= 0; --k) {
        long long aff_idx = base + (k + edge_offset);
        if (aff[aff_idx]) {
            LABEL_T a = seg[base + k];
            LABEL_T b = seg[base + k + 1];
            if (a < b) { seg[base + k + 1] = a; local = 1; }
            else if (b < a) { seg[base + k] = b; local = 1; }
        }
    }
    if (local) *changed = 1;
}
"""
    src = src.replace("LABEL_T", _LABEL_CTYPE[label_key])
    module = cp.RawModule(code=src)
    kernels = (
        module.get_function("sweep_axis0"),
        module.get_function("sweep_axis1"),
        module.get_function("sweep_axis2"),
    )
    _AFFINITY_CC_SWEEP_KERNELS[label_key] = kernels
    return kernels


def connected_components_affinity_3d_cupy(
    hard_aff: np.ndarray,
    edge_offset: int = 0,
    max_iters: int = 4096,
    use_uint32: bool = True,
) -> np.ndarray:
    """GPU-parallel CC over a 3D affinity graph using cupy.

    Same propagation strategy as ``..._numba_parallel`` (axis-aligned min sweeps)
    but using a custom RawKernel per axis — each thread does forward+backward
    serial scan along its column, so one outer iteration propagates fully along
    each axis line.

    Bit-exact with the serial kernel for ``edge_offset=0``; same partition
    (relabeling-invariant) for ``edge_offset=1``.

    Memory: one label buffer (seg) plus the bool affinity array on GPU; a
    device-side change flag replaces a snapshot copy so peak GPU usage is
    ``≈ 9 × X*Y*Z`` bytes with ``use_uint32=True`` (default). Falls back to
    uint64 if the foreground voxel count could exceed 2³² − 1; use chunked
    decoding for volumes that exceed device memory either way.
    """
    if not CUPY_AVAILABLE:
        raise ImportError(
            "cupy is required for backend='cupy'. Install a CUDA-matching wheel, e.g. "
            "`pip install cupy-cuda12x`."
        )

    edge_offset = int(edge_offset)
    if edge_offset not in (0, 1):
        raise ValueError(f"edge_offset must be 0 or 1, got {edge_offset}")
    if hard_aff.ndim != 4:
        raise ValueError(f"hard_aff must be 4D, got {hard_aff.ndim}D")
    if hard_aff.shape[0] < 3:
        raise ValueError(f"hard_aff must have >=3 channels, got {hard_aff.shape[0]}")

    cp = _cp
    if isinstance(hard_aff, cp.ndarray):
        hard_aff_gpu = cp.ascontiguousarray(hard_aff[:3])
    else:
        hard_aff_gpu = cp.asarray(np.ascontiguousarray(hard_aff[:3]))

    X, Y, Z = hard_aff_gpu.shape[1], hard_aff_gpu.shape[2], hard_aff_gpu.shape[3]

    fg = hard_aff_gpu.any(axis=0).copy()
    if edge_offset == 0:
        m0 = hard_aff_gpu[0, : X - 1, :, :]
        m1 = hard_aff_gpu[1, :, : Y - 1, :]
        m2 = hard_aff_gpu[2, :, :, : Z - 1]
    else:
        m0 = hard_aff_gpu[0, 1:X, :, :]
        m1 = hard_aff_gpu[1, :, 1:Y, :]
        m2 = hard_aff_gpu[2, :, :, 1:Z]
    fg[: X - 1, :, :] |= m0
    fg[1:X, :, :] |= m0
    fg[:, : Y - 1, :] |= m1
    fg[:, 1:Y, :] |= m1
    fg[:, :, : Z - 1] |= m2
    fg[:, :, 1:Z] |= m2

    if not bool(fg.any()):
        return np.zeros((X, Y, Z), dtype=np.uint32)

    fg_flat = fg.reshape(-1)
    # Cheap upper bound on the foreground count without an extra reduction
    # (avoids an extra V-sized pass and host sync just for the dtype check).
    n_fg_upper = int(fg_flat.size)
    if use_uint32 and n_fg_upper < (1 << 32):
        label_dtype = cp.uint32
        label_key = "uint32"
    else:
        label_dtype = cp.uint64
        label_key = "uint64"
    # Scan-order labels via cumsum + mask-multiply: foreground voxels get
    # rank 1..n_fg, background stays 0. No boolean-scatter scratch.
    labels_flat = cp.cumsum(fg_flat, dtype=label_dtype)
    labels_flat *= fg_flat
    seg = labels_flat.reshape((X, Y, Z))

    sweep0, sweep1, sweep2 = _build_affinity_cc_sweep_kernels(label_key)
    aff0 = cp.ascontiguousarray(hard_aff_gpu[0]).view(cp.uint8)
    aff1 = cp.ascontiguousarray(hard_aff_gpu[1]).view(cp.uint8)
    aff2 = cp.ascontiguousarray(hard_aff_gpu[2]).view(cp.uint8)

    block = (16, 16, 1)
    grid0 = ((Y + block[0] - 1) // block[0], (Z + block[1] - 1) // block[1], 1)
    grid1 = ((X + block[0] - 1) // block[0], (Z + block[1] - 1) // block[1], 1)
    grid2 = ((X + block[0] - 1) // block[0], (Y + block[1] - 1) // block[1], 1)

    changed = cp.zeros(1, dtype=cp.int32)
    converged = False
    for _ in range(max_iters):
        changed.fill(0)
        sweep0(grid0, block, (seg, aff0, X, Y, Z, edge_offset, changed))
        sweep1(grid1, block, (seg, aff1, X, Y, Z, edge_offset, changed))
        sweep2(grid2, block, (seg, aff2, X, Y, Z, edge_offset, changed))
        if int(changed[0]) == 0:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"connected_components_affinity_3d_cupy: did not converge in "
            f"{max_iters} sweeps; result may be incorrect for unusually long components.",
            RuntimeWarning,
        )

    # Skip cp.unique (O(V log V) sort + V-sized inverse). Transfer sparse
    # root labels to host and let fastremap.renumber compact them in O(V)
    # via its C++ hash; the wrapper's later fastremap.refit only does a
    # dtype refit, so total compaction cost is paid once.
    seg_cpu = cp.asnumpy(seg)
    seg_cpu, _ = fastremap.renumber(seg_cpu, preserve_zero=True, in_place=True)
    return seg_cpu


__all__ = [
    "CUPY_AVAILABLE",
    "NUMBA_AVAILABLE",
    "compute_edt",
    "connected_components_affinity_3d_cupy",
    "connected_components_affinity_3d_numba",
    "connected_components_affinity_3d_numba_parallel",
]
