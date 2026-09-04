#!/usr/bin/env python3
"""GT-free QC for the cross-sample affinity: compare moe against the in-domain IST run.

There is no ground truth for moe, so the only honest readout is whether the
affinity looks like the affinity the same checkpoint produces on the volume it was
trained on. Two things are checked, both of which a failed transfer breaks:

1. DISTRIBUTION. A working affinity head is strongly bimodal -- most voxels are
   confidently inside a process or confidently on a boundary. A collapsed
   transfer shows up as a unimodal blob with a small dynamic range. Reported as
   per-channel percentiles plus the fraction of mass in the middle of the range.
2. STRUCTURE. A PNG montage of one XY slice: image next to each affinity channel.
   Boundaries have to line up with membranes visible in the image.

    python tutorials/neuron_liconn_moe/qc_affinity.py                # every volume
    python tutorials/neuron_liconn_moe/qc_affinity.py --volume <name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volumes as V  # noqa: E402

REPO = V.REPO
# The in-domain reference: the same checkpoint on the volume it was trained on.
IST = V.TEST_OUT / "val/raw_x1_ch0-1-2.h5"
IST_IMAGE = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/img"


def volume_paths(name: str) -> tuple[Path, str]:
    """Affinity h5 + the image it was predicted from (volumes.py owns the
    legacy layout of the first, already-published volume)."""
    return V.affinity_h5(name), str(V.prepared_image(name))


def stats(path: Path, label: str, n_slices: int = 6) -> np.ndarray:
    """Per-channel percentiles over evenly spaced Z slices."""
    with h5py.File(path, "r") as f:
        arr = f["main"]
        print(f"\n{label}: {arr.shape} {arr.dtype}  {path}")
        zs = np.linspace(0, arr.shape[1] - 1, n_slices).astype(int)
        sub = np.stack([np.asarray(arr[:, z]).astype(np.float32) for z in zs], axis=1)
    for c, name in enumerate("ZYX"):
        v = sub[c].ravel()
        p = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        # "Middle mass": fraction of voxels in the central third of this
        # channel's own [p1, p99] range. Low = bimodal/decisive, high = mushy.
        lo, hi = p[0], p[-1]
        mid = ((v > lo + (hi - lo) / 3) & (v < lo + 2 * (hi - lo) / 3)).mean()
        print(
            f"  ch{c} ({name}) min {v.min():.4f} max {v.max():.4f} | "
            f"p1 {p[0]:.3f} p25 {p[2]:.3f} p50 {p[3]:.3f} p75 {p[4]:.3f} p99 {p[6]:.3f} | "
            f"mid-third mass {mid:.3f}"
        )
    return sub


def read_image_slice(spec: str, z: int, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    if ".zarr/" in spec:
        import zarr

        container, _, key = spec.partition(".zarr/")
        return np.asarray(zarr.open_group(container + ".zarr", mode="r")[key][z, y0:y1, x0:x1])
    with h5py.File(spec, "r") as f:
        return np.asarray(f["main"][z, y0:y1, x0:x1])


def montage(aff_path: Path, image_spec: str, out_png: Path, size: int = 512) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(aff_path, "r") as f:
        arr = f["main"]
        _, nz, ny, nx = arr.shape
        z = nz // 2
        y0 = max(0, ny // 2 - size // 2)
        x0 = max(0, nx // 2 - size // 2)
        box = (y0, y0 + size, x0, x0 + size)
        aff = np.asarray(arr[:, z, box[0] : box[1], box[2] : box[3]]).astype(np.float32)

    img = read_image_slice(image_spec, z, box)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"image  z={z}")
    for c, name in enumerate("ZYX"):
        axes[c + 1].imshow(aff[c], cmap="magma", vmin=aff.min(), vmax=aff.max())
        axes[c + 1].set_title(f"affinity ch{c} ({name})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{out_png.stem}  |  {size}x{size} centre crop")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", action="append", help="repeatable; default = all 8")
    ap.add_argument("--out", type=Path, default=V.OUT_ROOT / "qc")
    args = ap.parse_args()

    if IST.exists():
        stats(IST, "IST final_proofread val (in-domain reference)")
        montage(IST, IST_IMAGE, args.out / "ist_final_proofread_val_affinity.png")
    else:
        print(f"\nreference affinity missing, skipping: {IST}")

    for name in args.volume or list(V.VOLUMES):
        aff, image = volume_paths(name)
        if not Path(aff).exists():
            print(f"\nmissing, skipping: {aff}")
            continue
        stats(aff, f"moe {name} (transferred)")
        montage(aff, image, args.out / f"{name}_affinity.png")


if __name__ == "__main__":
    main()
