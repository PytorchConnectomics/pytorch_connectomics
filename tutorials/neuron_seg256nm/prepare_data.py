"""Data preparation for the 256 nm low-res tutorial: contrast equalisation and training cubes.

    python prepare_data.py clahe --src IMAGE.h5 --dst IMAGE_clahe.h5 [--cpus N]
    python prepare_data.py cubes --name m65 --image IMAGE.h5 --seg SEG.h5 --offset 4,515,392 \
        --out DIR [--split split_m65.json | --starts q0:37,3848,8444 ...] [--min-seg 0.9]

`clahe` writes a contrast-equalised copy of a uint8 volume once, so training does not repeat it
on the fly. Per z-section: the non-zero pixels are stretched linearly from their 0.5 and 99.5
percentiles to 1..255, then cv2 CLAHE (clip limit 4, about 128-pixel tiles) is applied. Pixels
that are 0 in the source (outside the tissue) stay 0. Use it for volumes whose grey levels span
a few tens of values (the minnie65 and minnie35 downsamples); leave volumes that were already
equalised (phase 1, H01) alone.

`cubes` cuts 256^3 image/label cubes on one grid. `--offset` is zyx with
image index = seg index + offset. With `--split` the cube positions are read from an earlier
cut's JSON (`train[].name`, `train[].start_image`), so a new image or label version lands on
the same cubes; `--starts` names them one by one (`tag:z,y,x`, image index), for a volume
whose labels cover only part of the region (H01's are a diagonal band); with neither the cubes
sit on the quarter points of the region the two volumes
share (`<name>_q0..q7`, or `q0..q3` on the mid-plane when the region is thinner than two cubes
in z) and on its centre (`<name>_ctr`). Images are written as uint8, labels as uint32 relabelled
per cube with 0 as background. A cube with under 90% non-zero image voxels, or under `--min-seg`
(default 0.9) non-zero label voxels, is an error.

All arrays, shapes and offsets are zyx.
"""
import argparse
import json
import os
from multiprocessing import Pool

import h5py
import numpy as np

CUBE = 256
MIN_NONZERO = 0.9


def dset(path):
    f = h5py.File(path, "r")
    return f["main"] if "main" in f else f[list(f)[0]]


# ---------------------------------------------------------------- clahe
def equalise_section(sec, clip, tile, pct):
    """Percentile stretch of the non-zero pixels to 1..255, then CLAHE; zeros stay zero."""
    import cv2

    mask = sec > 0
    n = int(mask.sum())
    if n == 0:
        return sec.copy()
    cdf = np.cumsum(np.bincount(sec[mask], minlength=256)) / n
    lo = int(np.searchsorted(cdf, pct[0] / 100.0))
    hi = int(np.searchsorted(cdf, pct[1] / 100.0))
    scale = 254.0 / max(hi - lo, 1)
    out = np.clip((sec.astype(np.float32) - lo) * scale + 1.0, 1, 255).astype(np.uint8)
    out[~mask] = 0
    grid = (max(1, round(sec.shape[1] / tile)), max(1, round(sec.shape[0] / tile)))
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(out)
    out[mask] = np.maximum(out[mask], 1)
    out[~mask] = 0
    return out


def _clahe_slab(args):
    src, z0, z1, clip, tile, pct = args
    slab = dset(src)[z0:z1]
    return z0, np.stack([equalise_section(s, clip, tile, pct) for s in slab])


def crop_stats(sec):
    """Grey-level statistics of the central 1024^2 of a section's non-zero bounding box."""
    nz = np.argwhere(sec > 0)
    if len(nz) == 0:
        return None
    (y0, x0), (y1, x1) = nz.min(0), nz.max(0) + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h, w = min(512, (y1 - y0) // 2), min(512, (x1 - x0) // 2)
    c = sec[cy - h:cy + h, cx - w:cx + w]
    p = np.percentile(c, [1, 50, 99])
    return dict(p1=float(p[0]), p50=float(p[1]), p99=float(p[2]), std=float(c.std()))


def cmd_clahe(a):
    src = dset(a.src)
    if src.dtype != np.uint8:
        raise SystemExit(f"{a.src}: expected uint8, got {src.dtype}")
    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} exists; remove it to rewrite")
    nz, step = src.shape[0], (src.chunks[0] if src.chunks else 16)
    jobs = [(a.src, z, min(z + step, nz), a.clip, a.tile, tuple(a.pct)) for z in range(0, nz, step)]
    tmp = a.dst + ".part"
    zeros_changed = 0
    with h5py.File(tmp, "w") as f, Pool(a.cpus) as pool:
        out = f.create_dataset("main", src.shape, np.uint8, chunks=src.chunks or True,
                               compression="gzip", compression_opts=4)
        out.attrs["source"] = os.path.abspath(a.src)
        out.attrs["contrast"] = (f"per z-section: non-zero pixels stretched from percentiles {a.pct[0]}, {a.pct[1]} "
                                 f"to 1..255, cv2 CLAHE clipLimit {a.clip}, tiles of about {a.tile} px; zeros kept")
        for i, (z0, slab) in enumerate(pool.imap(_clahe_slab, jobs)):
            out[z0:z0 + len(slab)] = slab
            if i % 10 == 0:
                print(f"clahe {os.path.basename(a.src)} z {z0 + len(slab)}/{nz}", flush=True)
    os.replace(tmp, a.dst)
    # Verify: same shape, the zero mask is unchanged on sampled sections, contrast before/after.
    dst, report = dset(a.dst), dict(src=a.src, dst=a.dst, shape=list(src.shape), clip=a.clip, tile=a.tile,
                                    pct=list(a.pct), sections=[])
    for z in sorted({nz // 8, nz // 4, nz // 2, 3 * nz // 4, 7 * nz // 8}):
        s, d = src[z], dst[z]
        zeros_changed += int(((s == 0) != (d == 0)).sum())
        report["sections"].append(dict(z=int(z), before=crop_stats(s), after=crop_stats(d)))
    report["zero_mask_changed_pixels_on_sampled_sections"] = zeros_changed
    with open(a.dst[:-3] + ".json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    if zeros_changed:
        raise SystemExit("zero mask changed")


# ---------------------------------------------------------------- cubes
def quarter_starts(lo, hi):
    """Cube starts on the quarter points and the centre of the box [lo, hi) (image index)."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    size = hi - lo
    if (size < CUBE).any():
        raise SystemExit(f"shared region {size.tolist()} is smaller than a {CUBE} cube")
    zs = [0.25, 0.75] if size[0] >= 2 * CUBE else [0.5]
    starts, k = [], 0
    for fz in zs:
        for fy in (0.25, 0.75):
            for fx in (0.25, 0.75):
                c = lo + (size * np.array([fz, fy, fx])).astype(int)
                starts.append((f"q{k}", np.clip(c - CUBE // 2, lo, hi - CUBE)))
                k += 1
    starts.append(("ctr", np.clip(lo + size // 2 - CUBE // 2, lo, hi - CUBE)))
    return starts


def overlay_png(path, img, seg):
    """Mid z/y/x sections of a cube, label boundaries in red on the image."""
    import cv2

    panels = []
    for ax in range(3):
        i, s = np.take(img, CUBE // 2, ax), np.take(seg, CUBE // 2, ax)
        edge = np.zeros(s.shape, bool)
        edge[:-1] |= s[:-1] != s[1:]
        edge[:, :-1] |= s[:, :-1] != s[:, 1:]
        rgb = np.repeat(i[..., None], 3, 2)
        rgb[edge] = (0, 0, 255)
        panels.append(np.hstack([np.repeat(i[..., None], 3, 2), rgb]))
    cv2.imwrite(path, np.vstack(panels))


def cmd_cubes(a):
    im, seg = dset(a.image), dset(a.seg)
    off = np.array([int(v) for v in a.offset.split(",")])
    bbox = seg.attrs.get("crop_bbox_yx_inclusive")
    if bbox is not None and (int(bbox[0]), int(bbox[2])) != (int(off[1]), int(off[2])):
        raise SystemExit(f"--offset {off.tolist()} disagrees with the label file's crop_bbox_yx_inclusive {bbox}")
    if a.split:
        with open(a.split) as f:
            sp = json.load(f)
        if sp["offset"] != off.tolist():
            raise SystemExit(f"--offset {off.tolist()} differs from {a.split} offset {sp['offset']}")
        starts = [(t["name"].split("_", 1)[1], np.array(t["start_image"])) for t in sp["train"]]
        if any(t["size"] != [CUBE] * 3 for t in sp["train"]):
            raise SystemExit(f"{a.split}: cubes are not {CUBE}^3")
    elif a.starts:
        starts = [(t.split(":")[0], np.array([int(v) for v in t.split(":")[1].split(",")])) for t in a.starts]
    else:
        lo = np.maximum(off, 0)
        hi = np.minimum(np.array(seg.shape) + off, np.array(im.shape))
        starts = quarter_starts(lo, hi)
    os.makedirs(a.out, exist_ok=True)
    rows, bad = [], []
    for tag, st in starts:
        name = f"{a.name}_{tag}"
        sl_im = tuple(slice(int(s), int(s) + CUBE) for s in st)
        sl_seg = tuple(slice(int(s - o), int(s - o) + CUBE) for s, o in zip(st, off))
        img, lab = im[sl_im], seg[sl_seg]
        if img.shape != (CUBE,) * 3 or lab.shape != (CUBE,) * 3:
            raise SystemExit(f"{name}: cube at {st.tolist()} leaves a volume ({img.shape}, {lab.shape})")
        ids, inv = np.unique(lab, return_inverse=True)
        out = inv.reshape(lab.shape).astype(np.uint32)
        if ids[0] != 0:
            out += 1
        row = dict(name=name, start_image=[int(v) for v in st], size=[CUBE] * 3, n_labels=int((ids != 0).sum()),
                   seg_nonzero=float((out > 0).mean()), image_nonzero=float((img > 0).mean()),
                   image_std=float(img.std()))
        for suffix, arr in (("img", img), ("seg", out)):
            with h5py.File(os.path.join(a.out, f"{name}_{suffix}.h5"), "w") as f:
                f.create_dataset("main", data=arr, chunks=(32, 32, 32), compression="gzip", compression_opts=4)
        if tag == "ctr":
            overlay_png(os.path.join(a.out, f"{name}_overlay.png"), img, out)
        if row["seg_nonzero"] < a.min_seg or row["image_nonzero"] < MIN_NONZERO:
            bad.append(name)
        rows.append(row)
        print(row, flush=True)
    with open(os.path.join(a.out, f"cubes_{a.name}.json"), "w") as f:
        json.dump(dict(name=a.name, image=os.path.abspath(a.image), seg=os.path.abspath(a.seg),
                       seg_attrs={k: str(seg.attrs[k]) for k in ("gt_version", "source") if k in seg.attrs},
                       image_attrs={k: str(v) for k, v in im.attrs.items()},
                       convention="zyx; image index = seg index + offset", offset=off.tolist(),
                       positions_from=a.split or ("--starts" if a.starts else "quarter points of the shared region"),
                       min_seg=a.min_seg, cubes=rows), f, indent=1)
    if bad:
        raise SystemExit(f"image under {MIN_NONZERO} or labels under {a.min_seg} non-zero: {bad}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clahe")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)
    c.add_argument("--clip", type=float, default=4.0)
    c.add_argument("--tile", type=int, default=128, help="approximate CLAHE tile edge in pixels")
    c.add_argument("--pct", type=float, nargs=2, default=[0.5, 99.5])
    c.add_argument("--cpus", type=int, default=8)
    c.set_defaults(fn=cmd_clahe)
    c = sub.add_parser("cubes")
    c.add_argument("--name", required=True)
    c.add_argument("--image", required=True)
    c.add_argument("--seg", required=True)
    c.add_argument("--offset", required=True, help="z,y,x with image index = seg index + offset")
    c.add_argument("--split", help="JSON of an earlier cut; reuse its cube positions")
    c.add_argument("--starts", nargs="+", help="explicit cubes, tag:z,y,x in image index")
    c.add_argument("--min-seg", type=float, default=MIN_NONZERO, help="least non-zero label fraction per cube")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_cubes)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
