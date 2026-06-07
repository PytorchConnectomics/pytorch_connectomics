"""Affinity-prediction QC: scan stats, derive Z cuts, build keep/drop mask.

The prediction is treated as ``(C, *spatial)`` with the final spatial axis
acting as the "Z" axis where slab-level outliers manifest. The mask layout is
``(X, Y, Z) uint8`` with 0=drop, 1=keep, consumed by
``decoding.affinity_mask_path``.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

CH_NAMES = ("X", "Y", "Z")


@dataclass
class AffinityQCParams:
    z_stride: int = 10
    k_edge: int = 20
    refine_window: int = 30
    drift_thresh: float = 0.05
    border_width: int = 32
    bg_thresh: int = 30
    n_z_border: int = 8


@dataclass
class AffinityQCReport:
    """Structured output of affinity QC scans.

    Streaming finalize populates ``z_idx``/``means``/``stds`` at raw stride 1;
    post-save scan populates them at stride ``z_stride``. ``interior_mean`` and
    cutoff math agree between paths.
    """

    low_z: int
    high_z: int
    interior_mean: np.ndarray
    nan_count: int
    inf_count: int
    sampled_voxels: int
    total_voxels: int
    g_mean: np.ndarray  # (C,)
    g_std: np.ndarray   # (C,)
    g_min: np.ndarray   # (C,)
    g_max: np.ndarray   # (C,)
    z_idx: np.ndarray   # (S,) for post-save, (Z,) for streaming
    means: np.ndarray   # (S, C) for post-save, (Z, C) for streaming
    stds: np.ndarray    # (S, C) for post-save, (Z, C) for streaming
    border_rows: list = field(default_factory=list)


class AffinityQCAccumulator:
    """Per-Z streaming stats for the chunked inference path.

    ``update`` is called with each chunk slab (typically ``(C, Zslab, Y, X)``
    from ``inference/chunked.py`` stitching). ``finalize`` derives the Z cuts
    from the accumulated per-slice means using the same `drift_thresh` /
    `k_edge` rules as :func:`scan_prediction`.
    """

    def __init__(self, channel_count: int, z_extent: int, params: "AffinityQCParams"):
        C, Z = int(channel_count), int(z_extent)
        if C <= 0 or Z <= 0:
            raise ValueError(f"AffinityQCAccumulator: bad shape C={C} Z={Z}")
        self.C = C
        self.Z = Z
        self.params = params
        self._sum = np.zeros((Z, C), dtype=np.float64)
        self._sq = np.zeros((Z, C), dtype=np.float64)
        self._mn = np.full((Z, C), np.inf, dtype=np.float32)
        self._mx = np.full((Z, C), -np.inf, dtype=np.float32)
        self._n = np.zeros(Z, dtype=np.int64)
        self._nan = 0
        self._inf = 0

    def update(self, slab: np.ndarray, *, z_offset: int, z_axis: int = -1) -> None:
        """Fold one chunk slab into the running per-Z streaming statistics."""
        if slab.ndim < 2 or slab.shape[0] != self.C:
            raise ValueError(
                f"slab.shape={slab.shape} incompatible with accumulator C={self.C}"
            )
        # Normalize to (C, Zslab, rest) with Z as the second axis for the loop.
        if z_axis < 0:
            z_axis = slab.ndim + z_axis
        if z_axis == 0:
            raise ValueError("z_axis cannot be the channel axis (0)")
        if z_axis != 1:
            slab = np.moveaxis(slab, z_axis, 1)
        slab = np.asarray(slab, dtype=np.float32)
        self._nan += int(np.isnan(slab).sum())
        self._inf += int(np.isinf(slab).sum())
        Zslab = slab.shape[1]
        if z_offset < 0 or z_offset + Zslab > self.Z:
            raise ValueError(
                f"slab z_offset={z_offset} Zslab={Zslab} out of bounds for Z={self.Z}"
            )
        for i in range(Zslab):
            sl = slab[:, i].reshape(self.C, -1)
            zi = z_offset + i
            self._sum[zi] += sl.sum(axis=1)
            self._sq[zi] += (sl.astype(np.float64) ** 2).sum(axis=1)
            self._mn[zi] = np.minimum(self._mn[zi], sl.min(axis=1))
            self._mx[zi] = np.maximum(self._mx[zi], sl.max(axis=1))
            self._n[zi] += sl.shape[1]

    def finalize(self, *, img=None) -> "AffinityQCReport":
        n = self._n
        if (n == 0).all():
            raise ValueError(
                "AffinityQCAccumulator.finalize: no slabs were folded in. "
                "In distributed runs, finish_streaming_qc must only be called on rank 0."
            )
        if (n == 0).any():
            missing = int((n == 0).sum())
            raise ValueError(
                f"AffinityQCAccumulator.finalize: {missing} Z slices were never updated"
            )
        means_raw = (self._sum / n[:, None]).astype(np.float32)        # (Z, C)
        var_raw = self._sq / n[:, None] - (self._sum / n[:, None]) ** 2
        stds_raw = np.sqrt(np.maximum(var_raw, 0)).astype(np.float32)  # (Z, C)
        z_idx_raw = np.arange(self.Z, dtype=np.int64)
        z_stride = self.params.z_stride
        ke = self.params.k_edge
        sampled = means_raw[::z_stride]
        interior = sampled[ke:-ke] if len(sampled) > 2 * ke + 1 else sampled
        interior_mean = interior.mean(axis=0)
        cutoff = interior_mean - self.params.drift_thresh

        head_end = min(self.params.refine_window, self.Z)
        low_z = head_end
        for z in range(head_end):
            if bool((means_raw[z] >= cutoff).all()):
                low_z = z
                break

        tail_start = max(0, self.Z - self.params.refine_window)
        last_ok = -1
        for z in range(tail_start, self.Z):
            if bool((means_raw[z] >= cutoff).all()):
                last_ok = z
        high_z = last_ok + 1 if last_ok >= 0 else tail_start

        # Global mean/std from per-Z sums.
        total_n = int(n.sum())
        sum_all = self._sum.sum(axis=0)
        sq_all = self._sq.sum(axis=0)
        g_mean = (sum_all / max(total_n, 1)).astype(np.float32)
        g_var = sq_all / max(total_n, 1) - (sum_all / max(total_n, 1)) ** 2
        g_std = np.sqrt(np.maximum(g_var, 0)).astype(np.float32)
        g_min = self._mn.min(axis=0)
        g_max = self._mx.max(axis=0)
        # XY-border check needs random access to the prediction; the streaming
        # path skips it (the post-save report can still be generated later).
        border_rows = ["(skipped — streaming mode does not retain spatial pred)"]
        return AffinityQCReport(
            low_z=int(low_z),
            high_z=int(high_z),
            interior_mean=interior_mean.astype(np.float32),
            nan_count=int(self._nan),
            inf_count=int(self._inf),
            sampled_voxels=total_n,
            total_voxels=total_n,
            g_mean=g_mean,
            g_std=g_std,
            g_min=g_min,
            g_max=g_max,
            z_idx=z_idx_raw,
            means=means_raw,
            stds=stds_raw,
            border_rows=border_rows,
        )


def _per_z_scan(pred, z_stride: int) -> dict:
    C = pred.shape[0]
    spatial = pred.shape[1:]
    Z = spatial[-1]
    # Chunk along Z if pred is a chunked dataset (h5py/zarr).
    cz = getattr(pred, "chunks", None)
    cz = cz[-1] if cz is not None else 32
    block_z = max(cz, z_stride)

    z_idx = np.arange(0, Z, z_stride, dtype=np.int64)
    means = np.zeros((len(z_idx), C), dtype=np.float32)
    stds = np.zeros((len(z_idx), C), dtype=np.float32)
    g_sum = np.zeros(C, dtype=np.float64)
    g_sq = np.zeros(C, dtype=np.float64)
    g_min = np.full(C, np.inf, dtype=np.float32)
    g_max = np.full(C, -np.inf, dtype=np.float32)
    g_n = 0
    nan_count = 0
    inf_count = 0
    for z0 in range(0, Z, block_z):
        z1 = min(z0 + block_z, Z)
        sel = [(i, z) for i, z in enumerate(z_idx) if z0 <= z < z1]
        if not sel:
            continue
        # Read native dtype (typically float16) — keep the slab compact and
        # only widen one z-plane at a time below. Halves peak RAM on large
        # (C, X, Y, block_z) reads.
        block = np.asarray(pred[..., z0:z1])
        nan_count += int(np.isnan(block).sum())
        inf_count += int(np.isinf(block).sum())
        for i, z in sel:
            sl = block[..., z - z0].astype(np.float32, copy=False).reshape(C, -1)
            means[i] = sl.mean(axis=1)
            stds[i] = sl.std(axis=1)
            g_sum += sl.sum(axis=1, dtype=np.float64)
            g_sq += np.square(sl, dtype=np.float64).sum(axis=1)
            g_min = np.minimum(g_min, sl.min(axis=1))
            g_max = np.maximum(g_max, sl.max(axis=1))
            g_n += sl.shape[1]
        del block
    return {
        "z_idx": z_idx, "means": means, "stds": stds,
        "g_sum": g_sum, "g_sq": g_sq, "g_min": g_min, "g_max": g_max,
        "g_n": g_n, "nan": nan_count, "inf": inf_count,
    }


def _refine_z_cuts(pred, interior_mean: np.ndarray,
                   refine_window: int, drift_thresh: float):
    C = pred.shape[0]
    Z = pred.shape[-1]
    cutoff = interior_mean - drift_thresh

    head_end = min(refine_window, Z)
    low_z = head_end
    head_rows = []
    if head_end > 0:
        # Read each Z-plane individually; refine_window is small (~30) so the
        # extra h5 calls are negligible vs holding (C, X, Y, refine_window)
        # widened to float32 in RAM.
        for z in range(head_end):
            m = np.asarray(pred[..., z]).astype(np.float32, copy=False) \
                .reshape(C, -1).mean(axis=1)
            ok = bool((m >= cutoff).all())
            head_rows.append((z, m.copy(), ok))
            if ok and low_z == head_end:
                low_z = z

    tail_start = max(0, Z - refine_window)
    high_z = tail_start
    tail_rows = []
    if tail_start < Z:
        last_ok = -1
        for z in range(tail_start, Z):
            m = np.asarray(pred[..., z]).astype(np.float32, copy=False) \
                .reshape(C, -1).mean(axis=1)
            ok = bool((m >= cutoff).all())
            tail_rows.append((z, m.copy(), ok))
            if ok:
                last_ok = z
        high_z = last_ok + 1 if last_ok >= 0 else tail_start

    return low_z, high_z, head_rows, tail_rows


def _xy_border_rows(pred, img, n_z: int, border: int, bg_thresh: int) -> list[str]:
    """One human-readable row per sampled z with border vs interior aff stats."""
    if img is None:
        return ["(skipped — no image provided)"]
    C = pred.shape[0]
    X, Y, Z = pred.shape[1], pred.shape[2], pred.shape[-1]
    if tuple(img.shape[:3]) != (X, Y, Z):
        return [f"WARN: img XYZ {tuple(img.shape[:3])} != pred XYZ {(X, Y, Z)}; skipped."]

    margin = max(int(0.02 * Z), 1)
    zs = np.linspace(margin, Z - margin - 1, n_z, dtype=np.int64)
    border_mask = np.zeros((X, Y), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    interior_mask = ~border_mask

    rows: list[str] = []
    for z in zs:
        img_xy = np.asarray(img[:, :, z, 0]) if img.ndim == 4 else np.asarray(img[:, :, z])
        bg_border = (img_xy <= bg_thresh) & border_mask
        bg_inter = (img_xy <= bg_thresh) & interior_mask
        n_bg = int(bg_border.sum())
        if n_bg < 100:
            rows.append(f"  z={z:5d}: bg-border voxels={n_bg} (too few)")
            continue
        pred_cxy = np.asarray(pred[..., z]).astype(np.float32)
        parts = [f"z={z:5d} bg_border_n={n_bg:>9,d} bg_int_n={int(bg_inter.sum()):>9,d}"]
        for c, name in enumerate(CH_NAMES[:C]):
            v_b = pred_cxy[c][bg_border]
            v_i = pred_cxy[c][bg_inter] if bg_inter.any() else np.zeros(1, dtype=np.float32)
            parts.append(
                f"ch{c}({name}): border μ={v_b.mean():.3f}/q95={np.quantile(v_b, 0.95):.3f}"
                f"/p>0.5={float((v_b > 0.5).mean()):.1%} vs int μ={v_i.mean():.3f}"
            )
        rows.append("  " + " | ".join(parts))
    return rows


def scan_prediction(
    pred,
    img,
    params: AffinityQCParams,
) -> AffinityQCReport:
    """Compute QC stats and resolve (low_z, high_z) cuts.

    ``pred`` is array-like with shape ``(C, *spatial)``; the last spatial
    axis is treated as Z. ``img`` is array-like ``(X, Y, Z[, 1])`` or
    ``None`` to skip the XY-border check.
    """
    C = pred.shape[0]
    Z = pred.shape[-1]
    total = int(np.prod(pred.shape[1:]))

    scan = _per_z_scan(pred, params.z_stride)
    g_n = max(scan["g_n"], 1)
    g_mean = scan["g_sum"] / g_n
    g_var = scan["g_sq"] / g_n - g_mean ** 2
    g_std = np.sqrt(np.maximum(g_var, 0))

    means = scan["means"]
    if len(means) > 2 * params.k_edge + 1:
        interior = means[params.k_edge:-params.k_edge]
    else:
        interior = means
    interior_mean = interior.mean(axis=0)

    low_z, high_z, _, _ = _refine_z_cuts(
        pred, interior_mean, params.refine_window, params.drift_thresh
    )
    border_rows = _xy_border_rows(
        pred, img, params.n_z_border, params.border_width, params.bg_thresh
    )

    return AffinityQCReport(
        low_z=low_z,
        high_z=high_z,
        interior_mean=interior_mean.astype(np.float32),
        nan_count=scan["nan"],
        inf_count=scan["inf"],
        sampled_voxels=scan["g_n"],
        total_voxels=total,
        g_mean=g_mean.astype(np.float32),
        g_std=g_std.astype(np.float32),
        g_min=scan["g_min"],
        g_max=scan["g_max"],
        z_idx=scan["z_idx"],
        means=scan["means"],
        stds=scan["stds"],
        border_rows=border_rows,
    )


def render_markdown_report(
    report: AffinityQCReport,
    params: AffinityQCParams,
    *,
    pred_desc: str,
    img_desc: str,
    mask_path: str,
    image_path: str,
) -> str:
    """Format a QC report as markdown with frontmatter consumed by build_affinity_mask."""
    C = len(report.g_mean)
    lines: list[str] = []
    lines.append("---")
    lines.append(f"img: {image_path}")
    lines.append(f"out: {mask_path}")
    lines.append(f"low_z: {report.low_z}")
    lines.append(f"high_z: {report.high_z}")
    lines.append(f"bg_thresh: {params.bg_thresh}")
    lines.append(f"border_width: {params.border_width}")
    lines.append("---")
    lines.append("")
    lines.append("# Affinity check report")
    lines.append(f"- pred: {pred_desc}")
    if img_desc:
        lines.append(f"- img:  {img_desc}")
    lines.append("")
    lines.append("## Volume health")
    lines.append(
        f"- Sampled {report.sampled_voxels:,} / {report.total_voxels:,} voxels per channel "
        f"(~{report.sampled_voxels / max(report.total_voxels, 1):.1%})."
    )
    lines.append(
        "- Per-Z report arrays are raw stride-1 for streaming reports and "
        "z_stride-sampled for post-save reports; derived cuts use the same "
        "sampled interior baseline."
    )
    lines.append(f"- NaN={report.nan_count}, Inf={report.inf_count}")
    lines.append("```")
    for c in range(C):
        name = CH_NAMES[c] if c < len(CH_NAMES) else f"c{c}"
        lines.append(
            f"  ch{c}({name}-aff): mean={report.g_mean[c]:.4f} std={report.g_std[c]:.4f} "
            f"min={report.g_min[c]:.4f} max={report.g_max[c]:.4f}"
        )
    lines.append("```")
    spread = float(report.g_mean.max() - report.g_mean.min())
    lines.append(
        f"- Channel-mean spread (max-min): {spread:.4f} "
        f"({'OK' if spread < 0.05 else 'imbalanced'})."
    )
    lines.append("")
    lines.append(f"## Derived Z cuts (drift_thresh={params.drift_thresh})")
    lines.append(
        f"- low_z={report.low_z}, high_z={report.high_z} "
        f"(keep z ∈ [{report.low_z}, {report.high_z}))."
    )
    lines.append("")
    lines.append(
        f"## XY-border + intensity (border={params.border_width}px, bg≤{params.bg_thresh}, "
        f"n_z={params.n_z_border})"
    )
    lines.append("```")
    for row in report.border_rows:
        lines.append(row)
    lines.append("```")
    return "\n".join(lines) + "\n"


def build_affinity_mask(
    img_path: str,
    out_path: str,
    *,
    low_z: int,
    high_z: int,
    bg_thresh: int,
    border_width: int,
    chunk_z: int = 256,
) -> None:
    """Write an h5 keep/drop mask, broadcast across the channel dimension.

    The mask is ``(X, Y, Z) uint8`` (0=drop, 1=keep). Two rules combined:
      - Z gate: sections outside ``[low_z, high_z)`` zeroed.
      - XY border + intensity: within the outer XY ring of width ``border_width``,
        voxels whose image intensity is ≤ ``bg_thresh`` are zeroed.
    """
    import h5py
    import zarr

    img = zarr.open(img_path, mode="r")
    if img.ndim == 4:
        def read_block(z0: int, z1: int) -> np.ndarray:
            return np.asarray(img[:, :, z0:z1, 0])
    elif img.ndim == 3:
        def read_block(z0: int, z1: int) -> np.ndarray:
            return np.asarray(img[:, :, z0:z1])
    else:
        raise ValueError(f"img must be 3D or 4D, got shape {img.shape}")

    X, Y, Z = img.shape[:3]
    if not (0 <= low_z <= high_z <= Z):
        raise ValueError(
            f"invalid z range [{low_z}, {high_z}) for Z={Z}"
        )

    border = np.zeros((X, Y), dtype=bool)
    if border_width > 0:
        border[:border_width, :] = True
        border[-border_width:, :] = True
        border[:, :border_width] = True
        border[:, -border_width:] = True

    chunks = (min(256, X), min(256, Y), min(256, Z))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.time()
    n_keep = 0
    n_total = 0
    with h5py.File(out_path, "w") as f:
        ds = f.create_dataset(
            "main",
            shape=(X, Y, Z),
            dtype=np.uint8,
            chunks=chunks,
            compression="gzip",
            compression_opts=4,
        )
        ds.attrs["bg_thresh"] = bg_thresh
        ds.attrs["border_width"] = border_width
        ds.attrs["low_z"] = low_z
        ds.attrs["high_z"] = high_z
        ds.attrs["axis_order"] = "XYZ"
        ds.attrs["source_image"] = img_path

        for z0 in range(0, Z, chunk_z):
            z1 = min(z0 + chunk_z, Z)
            block = np.ones((X, Y, z1 - z0), dtype=np.uint8)
            if border_width > 0:
                img_block = read_block(z0, z1)
                bg_border = (img_block <= bg_thresh) & border[:, :, None]
                block[bg_border] = 0
            for i, z in enumerate(range(z0, z1)):
                if z < low_z or z >= high_z:
                    block[:, :, i] = 0
            ds[:, :, z0:z1] = block
            n_keep += int(block.sum())
            n_total += int(block.size)
    logger.info(
        "Built affinity mask %s in %.1fs (kept %d/%d voxels, %.4f%%)",
        out_path, time.time() - t0, n_keep, n_total,
        100.0 * n_keep / max(n_total, 1),
    )


def apply_affinity_mask_from_spec(predictions: np.ndarray, spec: dict) -> tuple[int, int]:
    """Apply an affinity mask in-place using only the spec, skipping the full mask load.

    Equivalent to building a mask via :func:`build_affinity_mask` with the same
    parameters and then zeroing every prediction voxel where the mask is 0, but
    without materializing the (X, Y, Z) mask array. ``predictions`` is modified
    in place; the (zero, total) prediction-voxel counts are returned for logging.

    ``spec`` requires keys ``low_z``, ``high_z``, ``border_width``, ``bg_thresh``,
    ``image_path`` (a zarr 3D/4D source for the border-intensity check).
    ``axis_order`` is optional and must be ``"XYZ"`` if present.

    The two rules from :func:`build_affinity_mask` are applied:

    - Z-gate: contiguous slice writes zero everything outside ``[low_z, high_z)``.
    - Border + intensity: only the outer XY ring of width ``border_width`` is
      read from the source image; voxels with intensity <= ``bg_thresh`` are
      zeroed across all channels. Corners are zeroed twice (idempotent).
    """
    if predictions.ndim != 4:
        raise ValueError(
            f"apply_affinity_mask_from_spec expects 4D predictions, got {predictions.shape}"
        )
    axis_order = str(spec.get("axis_order", "XYZ"))
    if axis_order != "XYZ":
        raise ValueError(f"unsupported axis_order={axis_order!r}; only 'XYZ' is supported")

    _C, X, Y, Z = predictions.shape
    low_z = int(spec["low_z"])
    high_z = int(spec["high_z"])
    border_width = int(spec["border_width"])
    bg_thresh = int(spec["bg_thresh"])
    image_path = spec.get("image_path") or spec.get("source_image") or ""
    if not (0 <= low_z <= high_z <= Z):
        raise ValueError(f"invalid z range [{low_z}, {high_z}) for Z={Z}")

    n_total_pred = int(predictions.size)
    n_zero_voxels = 0  # counts in mask-volume coords (X*Y*Z), not predictions

    # Z-gate: contiguous slice writes.
    if low_z > 0:
        predictions[:, :, :, :low_z] = 0
        n_zero_voxels += X * Y * low_z
    if high_z < Z:
        predictions[:, :, :, high_z:] = 0
        n_zero_voxels += X * Y * (Z - high_z)

    # Border + intensity.
    if border_width > 0 and bg_thresh > 0 and image_path and high_z > low_z:
        bw = border_width
        if bw * 2 > X or bw * 2 > Y:
            raise ValueError(
                f"border_width={bw} too large for shape (X={X}, Y={Y})"
            )
        import zarr

        img = zarr.open(image_path, mode="r")
        if img.ndim == 4:
            def read_strip(xs: slice, ys: slice) -> np.ndarray:
                return np.asarray(img[xs, ys, low_z:high_z, 0])
        elif img.ndim == 3:
            def read_strip(xs: slice, ys: slice) -> np.ndarray:
                return np.asarray(img[xs, ys, low_z:high_z])
        else:
            raise ValueError(f"image must be 3D or 4D, got shape {img.shape}")

        strips = [
            (slice(0, bw), slice(None, None)),                       # left X edge
            (slice(X - bw, X), slice(None, None)),                   # right X edge
            (slice(bw, X - bw), slice(0, bw)),                       # left Y edge (no corner dup)
            (slice(bw, X - bw), slice(Y - bw, Y)),                   # right Y edge (no corner dup)
        ]
        for sx, sy in strips:
            img_strip = read_strip(sx, sy)
            zero_strip = img_strip <= bg_thresh
            if not zero_strip.any():
                continue
            # pred_view is a basic-slice view of predictions, so the fancy-index
            # assignment below writes through to predictions.
            pred_view = predictions[:, sx, sy, low_z:high_z]
            pred_view[:, zero_strip] = 0
            n_zero_voxels += int(zero_strip.sum())

    return n_zero_voxels, X * Y * Z


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _resolve_save_root(cfg: Any) -> str:
    decoding_cfg = _cfg_get(cfg, "decoding")
    inference_cfg = _cfg_get(cfg, "inference")
    root = _cfg_get(decoding_cfg, "save_path", "") or _cfg_get(inference_cfg, "save_path", "")
    if not root:
        raise ValueError(
            "affinity_qc: cannot auto-resolve mask_path/report_path because both "
            "decoding.save_path and inference.save_path are empty. Set "
            "decoding.affinity_qc.mask_path and .report_path explicitly."
        )
    return root


def _resolve_qc_paths(cfg: Any, qc_cfg: Any) -> tuple[str, str]:
    image_path = _cfg_get(qc_cfg, "image_path", "") or ""
    mask_path = _cfg_get(qc_cfg, "mask_path", "") or ""
    report_path = _cfg_get(qc_cfg, "report_path", "") or ""
    if not mask_path or not report_path:
        save_root = _resolve_save_root(cfg)
        mask_path = mask_path or os.path.join(save_root, "affinity_mask.h5")
        report_path = report_path or os.path.join(save_root, "affinity_qc_report.md")
    return image_path, mask_path, report_path


def _params_from_cfg(qc_cfg: Any) -> "AffinityQCParams":
    return AffinityQCParams(
        z_stride=int(_cfg_get(qc_cfg, "z_stride", 10)),
        k_edge=int(_cfg_get(qc_cfg, "k_edge", 20)),
        refine_window=int(_cfg_get(qc_cfg, "refine_window", 30)),
        drift_thresh=float(_cfg_get(qc_cfg, "drift_thresh", 0.05)),
        border_width=int(_cfg_get(qc_cfg, "border_width", 32)),
        bg_thresh=int(_cfg_get(qc_cfg, "bg_thresh", 30)),
        n_z_border=int(_cfg_get(qc_cfg, "n_z_border", 8)),
    )


def _wire_mask_path(decoding_cfg: Any, mask_path: str) -> None:
    try:
        decoding_cfg.affinity_mask_path = mask_path
    except Exception:
        if isinstance(decoding_cfg, dict):
            decoding_cfg["affinity_mask_path"] = mask_path


def begin_streaming_qc(
    cfg: Any,
    *,
    channel_count: int,
    z_extent: int,
) -> Optional["AffinityQCAccumulator"]:
    """Return a fresh accumulator if ``affinity_qc`` is enabled in streaming mode.

    Callers in the chunked inference path instantiate this before iterating
    chunks, then call ``acc.update(slab, z_offset=...)`` per slab, then
    :func:`finish_streaming_qc` to materialize the mask and report.
    """
    decoding_cfg = _cfg_get(cfg, "decoding")
    qc_cfg = _cfg_get(decoding_cfg, "affinity_qc")
    if not _cfg_get(qc_cfg, "enabled", False):
        return None
    if _cfg_get(qc_cfg, "mode", "post_save") != "streaming":
        return None
    if _cfg_get(decoding_cfg, "affinity_mask_path", "") or "":
        return None
    return AffinityQCAccumulator(channel_count, z_extent, _params_from_cfg(qc_cfg))


def finish_streaming_qc(
    cfg: Any,
    accumulator: "AffinityQCAccumulator",
    *,
    pred_desc: str = "(streaming chunked inference)",
) -> str:
    """Finalize a streaming accumulator, write mask+report, wire ``affinity_mask_path``."""
    decoding_cfg = _cfg_get(cfg, "decoding")
    qc_cfg = _cfg_get(decoding_cfg, "affinity_qc")
    image_path, mask_path, report_path = _resolve_qc_paths(cfg, qc_cfg)
    if not image_path:
        raise ValueError(
            "streaming affinity_qc requires affinity_qc.image_path "
            "(no in-memory prediction is retained to size the mask)"
        )
    params = accumulator.params

    import zarr

    img = zarr.open(image_path, mode="r")
    img_desc = f"{image_path} shape={tuple(img.shape)} dtype={img.dtype}"

    report = accumulator.finalize(img=img)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(render_markdown_report(
            report, params,
            pred_desc=pred_desc, img_desc=img_desc,
            mask_path=mask_path, image_path=image_path,
        ))
    logger.info(
        "Streaming affinity QC: low_z=%d high_z=%d → %s",
        report.low_z, report.high_z, report_path,
    )

    build_affinity_mask(
        image_path, mask_path,
        low_z=report.low_z, high_z=report.high_z,
        bg_thresh=params.bg_thresh, border_width=params.border_width,
    )

    _wire_mask_path(decoding_cfg, mask_path)
    return mask_path


def run_affinity_qc(cfg: Any, predictions: np.ndarray) -> Optional[str]:
    """Run QC + mask build if ``cfg.decoding.affinity_qc.enabled``.

    Writes ``mask_path`` and ``report_path``, then sets
    ``cfg.decoding.affinity_mask_path = mask_path`` so the downstream mask
    application picks it up. Returns the mask path or None if disabled.
    """
    decoding_cfg = _cfg_get(cfg, "decoding")
    qc_cfg = _cfg_get(decoding_cfg, "affinity_qc")
    if not _cfg_get(qc_cfg, "enabled", False):
        return None
    mode = _cfg_get(qc_cfg, "mode", "post_save")
    if mode == "streaming":
        # The chunked inference path is responsible for running this in-band.
        # If it ran, affinity_mask_path was already wired; if it didn't fire
        # (e.g. decode-only re-run), we silently fall back to post_save below.
        existing = _cfg_get(decoding_cfg, "affinity_mask_path", "") or ""
        if existing:
            return existing
        logger.info(
            "affinity_qc.mode=streaming but no mask was produced by inference; "
            "falling back to post_save scan."
        )
    if predictions.ndim != 4:
        raise ValueError(
            "affinity_qc requires 4D predictions (C, *spatial); "
            f"got shape {predictions.shape}"
        )

    params = _params_from_cfg(qc_cfg)
    image_path, mask_path, report_path = _resolve_qc_paths(cfg, qc_cfg)

    img = None
    img_desc = ""
    if image_path:
        import zarr

        img = zarr.open(image_path, mode="r")
        img_desc = f"{image_path} shape={tuple(img.shape)} dtype={img.dtype}"

    pred_desc = f"in-memory shape={tuple(predictions.shape)} dtype={predictions.dtype}"
    logger.info("Running affinity QC on predictions (shape=%s)", predictions.shape)
    report = scan_prediction(predictions, img, params)

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(render_markdown_report(
            report, params,
            pred_desc=pred_desc, img_desc=img_desc,
            mask_path=mask_path, image_path=image_path,
        ))
    logger.info(
        "Affinity QC: low_z=%d high_z=%d → wrote %s",
        report.low_z, report.high_z, report_path,
    )

    if not image_path:
        # Without the source image we can still build a Z-only mask using the
        # prediction's own shape, but the border rule is a no-op.
        # Fall back: write a dummy mask shape via predictions shape (X,Y,Z).
        C, X, Y, Z = predictions.shape
        if not (0 <= report.low_z <= report.high_z <= Z):
            raise ValueError(
                f"affinity_qc: invalid z range [{report.low_z}, {report.high_z}) "
                f"for Z={Z}"
            )
        import h5py

        os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)
        with h5py.File(mask_path, "w") as f:
            ds = f.create_dataset(
                "main",
                shape=(X, Y, Z),
                dtype=np.uint8,
                chunks=(min(256, X), min(256, Y), min(256, Z)),
                compression="gzip",
                compression_opts=4,
            )
            ds.attrs["low_z"] = report.low_z
            ds.attrs["high_z"] = report.high_z
            ds.attrs["border_width"] = 0
            ds.attrs["bg_thresh"] = params.bg_thresh
            ds.attrs["axis_order"] = "XYZ"
            keep_z = np.zeros(Z, dtype=np.uint8)
            keep_z[report.low_z:report.high_z] = 1
            ds[...] = np.broadcast_to(keep_z, (X, Y, Z)).copy()
    else:
        build_affinity_mask(
            image_path, mask_path,
            low_z=report.low_z, high_z=report.high_z,
            bg_thresh=params.bg_thresh, border_width=params.border_width,
        )

    _wire_mask_path(decoding_cfg, mask_path)
    return mask_path
