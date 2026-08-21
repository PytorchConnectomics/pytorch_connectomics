"""Build a whole-volume contact graph (region adjacency + contact area) from a label volume.

`connectomics.decoding.decoders.segmentation_merge` needs, per pair of touching segments, the number
of shared voxel faces. On a volume that fits in memory the decoder computes this itself; on a
whole-dataset segmentation it does not fit, so this script computes it in Z-slabs and the decoder
reads the result via `contact_path`.

Everything here is ground-truth-free: the only input is the segmentation.

One slab per array task (each slab reads one extra Z row so the face between slabs is counted exactly
once, by the lower slab), then a merge pass:

    sbatch --array=0-22 ... python scripts/build_contact_graph.py --seg <zarr> --slab $SLURM_ARRAY_TASK_ID
    python scripts/build_contact_graph.py --seg <zarr> --merge

usage:
  python scripts/build_contact_graph.py --seg seg.zarr --slab 0 --out contacts/
  python scripts/build_contact_graph.py --seg seg.zarr --merge --out contacts/ --merged contacts.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class _PrecomputedZYX:
    """Expose a CloudVolume precomputed layer as a lazy [z, y, x] array.

    ABISS writes its segmentation as precomputed (XYZ, channel-last), but everything in this
    script indexes [z, y, x]. Rather than transpose a whole-volume array, wrap the reads.
    """

    def __init__(self, cloudpath: str, mip):
        from cloudvolume import CloudVolume
        self._cv = CloudVolume(cloudpath, mip=list(mip), fill_missing=True,
                               bounded=False, progress=False)
        sx, sy, sz = (int(v) for v in self._cv.shape[:3])
        self.shape = (sz, sy, sx)

    def __getitem__(self, key):
        z, y, x = key if isinstance(key, tuple) else (key, slice(None), slice(None))

        def span(s, n):
            if isinstance(s, slice):
                return (0 if s.start is None else max(0, s.start),
                        n if s.stop is None else min(n, s.stop))
            return (s, s + 1)

        z0, z1 = span(z, self.shape[0])
        y0, y1 = span(y, self.shape[1])
        x0, x1 = span(x, self.shape[2])
        block = np.asarray(self._cv[x0:x1, y0:y1, z0:z1])
        if block.ndim == 4:
            block = block[..., 0]
        return np.ascontiguousarray(block.transpose(2, 1, 0))  # xyz -> zyx


def open_volume(path: str, dataset: str, precomputed: bool = False, mip=(9, 9, 20)):
    """Open a zarr group/array, an HDF5 dataset, or a precomputed layer lazily."""
    if precomputed or Path(path, "info").is_file():
        cloudpath = path if "://" in path else f"file://{path}"
        return _PrecomputedZYX(cloudpath, mip)
    if path.endswith(".zarr") or "/.zarray" in path or Path(path, ".zgroup").exists():
        import zarr
        z = zarr.open(path, mode="r")
        return z[dataset] if hasattr(z, "array_keys") and dataset in list(z.array_keys()) else z
    import h5py
    return h5py.File(path, "r")[dataset]


def label_sizes(vol, slab_z: int = 256) -> dict:
    """Voxel count per label, streamed in Z-slabs."""
    from collections import Counter
    acc = Counter()
    for z0 in range(0, vol.shape[0], slab_z):
        sub = np.asarray(vol[z0:min(vol.shape[0], z0 + slab_z)])
        lab, cnt = np.unique(sub, return_counts=True)
        for l, c in zip(lab.tolist(), cnt.tolist()):
            if l:
                acc[l] += c
        log(f"  sizes z {z0}/{vol.shape[0]}")
    return dict(acc)


def slab_contacts(vol, sid: int, slab_z: int, yx: int, keep: np.ndarray) -> tuple:
    """Face-adjacency pairs within one Z-slab, restricted to labels in `keep` (sorted)."""
    nk = len(keep)
    bits = int(np.ceil(np.log2(nk + 2)))
    if 2 * bits >= 64:
        raise ValueError(f"too many labels to pack ({nk}); raise --min-size")
    shp = vol.shape
    z0 = sid * slab_z
    z1 = min(shp[0], z0 + slab_z + 1)
    if z0 >= shp[0]:
        return np.zeros(0, np.uint64), np.zeros(0, np.uint64), np.zeros(0, np.int64)

    def dense(v):
        j = np.searchsorted(keep, v)
        np.clip(j, 0, nk - 1, out=j)
        return np.where(keep[j] == v, j, nk).astype(np.uint64)

    folded = None
    t0 = time.time()
    for y0 in range(0, shp[1], yx):
        y1 = min(shp[1], y0 + yx + 1)
        for x0 in range(0, shp[2], yx):
            x1 = min(shp[2], x0 + yx + 1)
            v = np.asarray(vol[z0:z1, y0:y1, x0:x1])
            if v.max() == 0:
                continue
            for ax in range(3):
                a = np.moveaxis(v, ax, 0)
                lo, hi = a[:-1], a[1:]
                m = (lo != hi) & (lo != 0) & (hi != 0)
                if not m.any():
                    continue
                ia, ib = dense(lo[m]), dense(hi[m])
                ok = (ia < nk) & (ib < nk)
                if not ok.any():
                    continue
                ia, ib = ia[ok], ib[ok]
                key = (np.minimum(ia, ib) << np.uint64(bits)) | np.maximum(ia, ib)
                u, c = np.unique(key, return_counts=True)
                if folded is None:
                    folded = (u, c.astype(np.int64))
                else:
                    uu = np.concatenate([folded[0], u])
                    cc = np.concatenate([folded[1], c.astype(np.int64)])
                    u2, inv = np.unique(uu, return_inverse=True)
                    c2 = np.zeros(len(u2), np.int64)
                    np.add.at(c2, inv, cc)
                    folded = (u2, c2)
            del v
        log(f"  y {y0}/{shp[1]}: {0 if folded is None else len(folded[0]):,} pairs "
            f"[{time.time()-t0:.0f}s]")
    if folded is None:
        return np.zeros(0, np.uint64), np.zeros(0, np.uint64), np.zeros(0, np.int64)
    u, c = folded
    lo = keep[(u >> np.uint64(bits)).astype(np.int64)]
    hi = keep[(u & np.uint64((1 << bits) - 1)).astype(np.int64)]
    return lo, hi, c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg", required=True, help="segmentation zarr or h5")
    p.add_argument("--dataset", default="main")
    p.add_argument("--out", default="contacts", help="directory for per-slab shards")
    p.add_argument("--merged", default="", help="path of the merged .npz (default <out>/merged.npz)")
    p.add_argument("--slab", type=int, default=None, help="slab index to compute")
    p.add_argument("--merge", action="store_true", help="merge existing shards")
    p.add_argument("--sizes", default="", help="cached label sizes .npz (labels, counts)")
    p.add_argument("--slab-z", type=int, default=256)
    p.add_argument("--yx", type=int, default=1024)
    p.add_argument("--min-size", type=int, default=200,
                   help="ignore labels smaller than this (noise floor)")
    p.add_argument("--precomputed", action="store_true",
                   help="--seg is a CloudVolume precomputed layer (ABISS output)")
    p.add_argument("--mip", type=int, nargs=3, default=(9, 9, 20),
                   help="precomputed mip/resolution to read")
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    merged = Path(a.merged) if a.merged else out / "merged.npz"

    if a.merge:
        A, B, N = [], [], []
        for f in sorted(out.glob("slab_*.npz")):
            z = np.load(f)
            A.append(z["a"]); B.append(z["b"]); N.append(z["n"].astype(np.int64))
            log(f"  {f.name}: {len(z['n']):,}")
        if not A:
            raise SystemExit(f"no shards in {out}")
        A = np.concatenate(A); B = np.concatenate(B); N = np.concatenate(N)
        uq = np.unique(np.concatenate([A, B]))
        bits = int(np.ceil(np.log2(len(uq) + 2)))
        ia = np.searchsorted(uq, A).astype(np.uint64)
        ib = np.searchsorted(uq, B).astype(np.uint64)
        key = (ia << np.uint64(bits)) | ib
        u, inv = np.unique(key, return_inverse=True)
        n = np.zeros(len(u), np.int64)
        np.add.at(n, inv, N)
        np.savez_compressed(merged,
                            a=uq[(u >> np.uint64(bits)).astype(np.int64)],
                            b=uq[(u & np.uint64((1 << bits) - 1)).astype(np.int64)],
                            n=n.astype(np.uint64))
        log(f"merged {len(n):,} contacts over {len(uq):,} labels -> {merged}")
        return

    vol = open_volume(a.seg, a.dataset, precomputed=a.precomputed, mip=tuple(a.mip))
    if a.sizes:
        z = np.load(a.sizes)
        lb, ct = z["labels"].astype(np.uint64), z["counts"].astype(np.int64)
    else:
        s = label_sizes(vol, a.slab_z)
        lb = np.array(sorted(s), dtype=np.uint64)
        ct = np.array([s[int(k)] for k in lb.tolist()], dtype=np.int64)
        np.savez_compressed(out / "sizes.npz", labels=lb, counts=ct)
    keep = np.sort(lb[ct >= a.min_size])
    log(f"volume {vol.shape}, {len(keep):,} labels >= {a.min_size} vox")

    if a.slab is None:
        raise SystemExit("pass --slab N (or --merge)")
    lo, hi, c = slab_contacts(vol, a.slab, a.slab_z, a.yx, keep)
    np.savez_compressed(out / f"slab_{a.slab}.npz", a=lo, b=hi, n=c.astype(np.uint64))
    log(f"done slab {a.slab}: {len(c):,} contacts")


if __name__ == "__main__":
    main()
