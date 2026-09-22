#!/usr/bin/env python3
"""Whole-volume ABISS merge-threshold sweep for the mip0 val affinity, scored on
the 18 nm grid against the SAME ground truth the mip1 baseline used.

This is `wholeval_sweep.py` for a volume 8x larger, and it differs in exactly two
ways, both forced.

1. NOTHING THE SIZE OF THE VOLUME IS HELD IN RAM. `wholeval_sweep.py` does
   `np.asarray(f["main"]).astype(np.float32)` (200 GB here), hands it to
   `_to_abiss_affinity` which makes a transposed float32 copy (another 200 GB),
   and receives back a dict holding one uint64 segmentation per threshold
   (133 GB each). That is the ~1.2 TB figure, and almost all of it is the python
   wrapper rather than ABISS: `ws` mmaps its affinity
   (src/ws/atomic_chunk.cpp, `MMArray<aff_t,4>`) and in batch mode frees each
   threshold's segmentation right after writing it. So here the affinity is
   streamed to `aff.raw` a z-slab at a time, and each segmentation is read back
   through a memmap and downsampled on the fly. Peak python RSS is a few GB plus
   the scoring arrays.

2. THE SCORE IS TAKEN ON THE 18 nm GRID. VOI is not comparable across voxel
   grids: scoring 16.65 Gvox against a nearest-upsampled GT and comparing that to
   0.9130 would be comparing two different quantities. The mip0 segmentation is
   therefore downsampled by [::2,::2,::2] onto the 18 nm grid and scored against
   `final_proofread/val/data.zarr/seg` -- the exact array the mip1 numbers used.
   That phase is not a guess: the mip0 GT is a bit-exact `np.repeat(...,2)` of the
   18 nm GT, so `[::2,::2,::2]` is its exact inverse (verified on 5/5 random
   blocks, 2026-09-20), and the two crops cover the identical physical region
   ([280,480,480]->[1110,9060,7170] is exactly 2x [140,240,240]->[555,4530,3585]).

   What this costs, and it is a real cost: point-sampling can drop a predicted
   fragment that is thin at mip0, which shows up as a split. `--phase` scores a
   different corner of each 2x2x2 block; if the two phases disagree by much more
   than the ~0.01 VOI noise floor, the downsample is doing work of its own and
   the number should not be read as a resolution effect.

Merge function is `max` in the compressed (scale_sigmoid) space, matching the
baseline exactly. `max` is monotone-invariant so the sweep explores the same
family of segmentations it would on probabilities; `mean` is not, and the chunked
ABISS path (`run_abiss_chunk.py`) is mean-only -- every `agg*` target in
lib/abiss/CMakeLists.txt compiles `src/agg/mean_aggl.cpp` -- which is why this
does not use it.

    python mip0_wholeval_sweep.py --affinity <h5> --label mip0 \
        --thresholds 0.41,0.44,0.47,0.50,0.53 --workdir <scratch on /projects>
"""
from __future__ import annotations
import argparse, gc, importlib.util, json, os, re, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np

REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
GT = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/seg"
WS = REPO / "lib/abiss/build/ws"
sys.path.insert(0, str(REPO))


def _load_runner():
    spec = importlib.util.spec_from_file_location("rav", REPO / "scripts/run_abiss_volume.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stream_write_aff_raw(aff_h5: Path, out: Path, *, zslab: int = 8) -> tuple[int, int, int]:
    """Write ABISS `aff.raw` (X,Y,Z,3) float32 Fortran order with a 1-voxel halo.

    Reproduces `_to_abiss_affinity(channels=[2,1,0], edge_storage="source")` plus
    `_write_affinity_with_halo(halo=1)` exactly, without ever holding the volume.

    The channel map, spelled out because getting it wrong is silent:
      pred ch0 = Z edge -> ABISS channel 2
      pred ch1 = Y edge -> ABISS channel 1
      pred ch2 = X edge -> ABISS channel 0
    and `edge_storage=source` shifts each ABISS channel k by +1 along axis k
    (this repo stores edge (i,i+1) at i; `ws` reads the value at i as edge
    (i-1,i)), zeroing the exposed face. The X and Y shifts are in-plane; the Z
    shift means output plane z reads source plane z-1, which is why the loop
    reads a z-window offset by one.
    """
    import h5py
    with h5py.File(aff_h5, "r") as f:
        d = f["main"]
        C, Z, Y, X = d.shape
        assert C >= 3, f"expected >=3 affinity channels, got {C}"
        xo, yo, zo = X + 2, Y + 2, Z + 2
        mm = np.memmap(out, dtype=np.float32, mode="w+", shape=(xo, yo, zo, 3), order="F")
        mm[...] = 0
        t0 = time.time()
        for z0 in range(0, Z, zslab):
            z1 = min(z0 + zslab, Z)
            # channels 0 (X edge) and 1 (Y edge): same z planes, in-plane shift
            blk = np.asarray(d[1:3, z0:z1]).astype(np.float32)   # (2, dz, Y, X)
            # ABISS ch0 <- pred ch2 (X edge), shift +1 along X
            cx = np.roll(blk[1], shift=1, axis=2); cx[:, :, 0] = 0.0
            # ABISS ch1 <- pred ch1 (Y edge), shift +1 along Y
            cy = np.roll(blk[0], shift=1, axis=1); cy[:, 0, :] = 0.0
            del blk
            mm[1:1 + X, 1:1 + Y, 1 + z0:1 + z1, 0] = np.transpose(cx, (2, 1, 0))
            mm[1:1 + X, 1:1 + Y, 1 + z0:1 + z1, 1] = np.transpose(cy, (2, 1, 0))
            del cx, cy
            # ABISS ch2 <- pred ch0 (Z edge), shift +1 along Z: plane z reads z-1
            s0, s1 = z0 - 1, z1 - 1
            if s1 > 0:
                lo = max(s0, 0)
                cz = np.asarray(d[0, lo:s1]).astype(np.float32)
                dst0 = 1 + z0 + (lo - s0)
                mm[1:1 + X, 1:1 + Y, dst0:1 + z1, 2] = np.transpose(cz, (2, 1, 0))
                del cz
            if z0 % (zslab * 8) == 0:
                print(f"    aff.raw {z1}/{Z} planes  {time.time()-t0:.0f}s", flush=True)
        mm.flush()
        del mm
        gc.collect()
    return (xo, yo, zo)


def read_seg_downsampled(seg_path: Path, xyz_shape: tuple[int, int, int],
                         *, halo: int = 1, factor: int = 2,
                         phase: int = 0) -> np.ndarray:
    """Memmap an ABISS seg (X,Y,Z uint64) and return ZYX downsampled by `factor`.

    Never materializes the full-resolution volume. Halo handling mirrors
    `run_abiss_volume._read_segmentation_xyz`: `ws` writes either the exact
    interior or the padded volume, distinguished by file size.
    """
    X, Y, Z = (int(v) for v in xyz_shape)
    itemsize = np.dtype(np.uint64).itemsize
    size = seg_path.stat().st_size
    if size == X * Y * Z * itemsize:
        mm = np.memmap(seg_path, dtype=np.uint64, mode="r", shape=(X, Y, Z), order="F")
        off = 0
    else:
        Xh, Yh, Zh = X + 2 * halo, Y + 2 * halo, Z + 2 * halo
        if size != Xh * Yh * Zh * itemsize:
            raise ValueError(
                f"Unexpected ABISS segmentation size: {size} bytes at {seg_path}; "
                f"expected {X*Y*Z*itemsize} (interior) or {Xh*Yh*Zh*itemsize} (halo {halo}).")
        mm = np.memmap(seg_path, dtype=np.uint64, mode="r", shape=(Xh, Yh, Zh), order="F")
        off = halo
    zs = range(phase, Z, factor)
    out = np.empty((len(zs), (Y - phase + factor - 1) // factor,
                    (X - phase + factor - 1) // factor), dtype=np.uint64)
    for i, z in enumerate(zs):
        plane = mm[off:off + X, off:off + Y, off + z]          # (X, Y)
        out[i] = plane[phase::factor, phase::factor].T          # -> (Y', X')
    del mm
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--affinity", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--thresholds", default="0.41,0.44,0.47,0.50,0.53")
    ap.add_argument("--ws-high", default="94%")
    ap.add_argument("--ws-low", default="20%")
    ap.add_argument("--workdir", type=Path, required=True,
                    help="scratch on /projects -- node-local /tmp cannot hold ~200 GB")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--phase", type=int, default=0,
                    help="which corner of each factor^3 block to sample; 0 is the "
                         "exact inverse of the GT's np.repeat upsample")
    ap.add_argument("--phases", default=None,
                    help="comma-separated phases to score from the SAME decode, "
                         "e.g. 0,1. This is the robustness check on the "
                         "downsample itself: point-sampling can drop a predicted "
                         "fragment that is thin at mip0, which shows up as a "
                         "split. If the phases disagree by much more than the "
                         "~0.01 VOI noise floor then the downsample is doing work "
                         "of its own and the number is not a clean resolution "
                         "effect. Overrides --phase.")
    ap.add_argument("--percentile-sample", type=int, default=64,
                    help="stride for the z-subsample used to resolve %% thresholds")
    ap.add_argument("--gt", default=GT,
                    help="GT zarr array; the default is the 18 nm val seg the mip1 baseline was scored against. Override only for smoke tests on crops.")
    ap.add_argument("--ws-binary", type=Path, default=WS,
                    help="ABISS watershed binary. The stock lib/abiss/build/ws uses a "
                         "uint32 internal index and asserts out above 2,147,483,648 "
                         "voxels per invocation; lib/abiss/build64/ws64 is the same "
                         "source built with WS_INTERNAL_SEG64. The on-disk output "
                         "format is identical either way.")
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument("--reuse-aff", action="store_true",
                    help="reuse an existing aff.raw/param.txt in the workdir")
    a = ap.parse_args()

    import h5py, zarr
    from connectomics.metrics.segmentation_numpy import adapted_rand, voi
    rav = _load_runner()
    mts = [float(v) for v in a.thresholds.split(",")]
    a.workdir.mkdir(parents=True, exist_ok=True)
    # `ws` is invoked with cwd=workdir, so every path handed to it must be
    # absolute or it silently cannot open its own inputs and aborts (SIGABRT,
    # "failed opening file ... iostream error"). The reference _run_abiss_ws
    # resolves its temp dir for exactly this reason.
    a.workdir = a.workdir.resolve()
    a.affinity = a.affinity.resolve()

    with h5py.File(a.affinity, "r") as f:
        C, Z, Y, X = f["main"].shape
        dtype = f["main"].dtype
    print(f"[{a.label}] affinity {a.affinity}", flush=True)
    print(f"[{a.label}] shape ({C}, {Z}, {Y}, {X}) {dtype}  "
          f"{Z*Y*X/1e9:.2f} Gvox", flush=True)

    # Percentile thresholds on a strided z-subsample: the full array is 200 GB as
    # float32 and np.percentile would sort a copy of it. Every 64th plane is
    # ~1.6% of the volume and its quantiles are stable to <1e-3 here.
    aff_raw_path = a.workdir / "aff.raw"
    param_path = a.workdir / "param.txt"
    if a.reuse_aff and aff_raw_path.exists() and param_path.exists():
        ws_xyz = tuple(int(v) for v in param_path.read_text().split("\n")[0].split())
        print(f"[{a.label}] reusing {aff_raw_path} ({ws_xyz})", flush=True)
        with h5py.File(a.affinity, "r") as f:
            sub = np.asarray(f["main"][:3, ::a.percentile_sample]).astype(np.float32)
    else:
        with h5py.File(a.affinity, "r") as f:
            sub = np.asarray(f["main"][:3, ::a.percentile_sample]).astype(np.float32)

    hi = rav._resolve_threshold(a.ws_high, sub, "ws_high")
    lo = rav._resolve_threshold(a.ws_low, sub, "ws_low")
    print(f"[{a.label}] subsample every {a.percentile_sample}th z-plane "
          f"({sub.shape[1]} planes): range=[{sub.min():.4f},{sub.max():.4f}] "
          f"p25/p50/p75={np.percentile(sub,[25,50,75]).round(4).tolist()}", flush=True)
    del sub
    gc.collect()
    print(f"[{a.label}] ws_high={hi:.6f} ws_low={lo:.6f} mts={mts} fn=max "
          f"space=compressed", flush=True)

    if not (a.reuse_aff and aff_raw_path.exists() and param_path.exists()):
        t0 = time.time()
        ws_xyz = stream_write_aff_raw(a.affinity, aff_raw_path)
        rav._write_abiss_param_file(param_path, ws_xyz, [1, 1, 1, 1, 1, 1], 0)
        print(f"[{a.label}] wrote aff.raw {ws_xyz} "
              f"({aff_raw_path.stat().st_size/1e9:.1f} GB) in {time.time()-t0:.0f}s",
              flush=True)

    ws_bin = a.ws_binary.resolve()
    if not ws_bin.exists():
        raise SystemExit(f"ABISS watershed binary not found: {ws_bin}")
    # Guard rail. The stock uint32 build asserts above 2^31 voxels -- but that
    # assert is compiled out under -DNDEBUG, and then the index silently
    # overflows and the decode returns a plausible-looking wrong segmentation.
    # Check it here so the failure cannot depend on how ABISS was built.
    nvox_halo = int(ws_xyz[0]) * int(ws_xyz[1]) * int(ws_xyz[2])
    print(f"[{a.label}] ws binary {ws_bin} | chunk {nvox_halo:,} voxels "
          f"({100.0 * nvox_halo / 0x80000000:.1f}% of the uint32 cap)", flush=True)
    if nvox_halo >= 0x80000000 and ws_bin.name != "ws64":
        raise SystemExit(
            f"{nvox_halo:,} voxels exceeds the uint32 watershed cap 2,147,483,648. "
            f"Use lib/abiss/build64/ws64 via --ws-binary; a larger --mem cannot help.")
    cmd = ["/usr/bin/time", "-v", str(ws_bin), str(param_path), str(aff_raw_path),
           str(hi), str(lo), str(10_000_000), str(200), a.label, "max"] + [str(m) for m in mts]
    print(f"[{a.label}] $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(a.workdir), capture_output=True, text=True)
    print(p.stdout[-6000:], flush=True)
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
    ws_peak_gb = int(m.group(1)) / 1e6 if m else float("nan")
    print(f"[{a.label}] ws rc={p.returncode} peak_rss={ws_peak_gb:.1f} GB "
          f"wall={time.time()-t0:.0f}s", flush=True)
    if p.returncode != 0:
        print(p.stderr[-6000:], flush=True)
        raise SystemExit(f"ABISS ws failed rc={p.returncode}")

    interior_xyz = (X, Y, Z)
    phases = ([int(v) for v in a.phases.split(",")] if a.phases else [a.phase])
    gt = np.asarray(zarr.open(a.gt, mode="r")[:])
    print(f"[{a.label}] GT {gt.shape} {gt.dtype}", flush=True)

    rows = []
    for i, mt in enumerate(mts):
        seg_file = a.workdir / f"seg_{a.label}_{i}.data"
        if not seg_file.exists():
            single = a.workdir / f"seg_{a.label}.data"
            if len(mts) == 1 and single.exists():
                seg_file = single
            else:
                raise FileNotFoundError(f"ABISS did not produce {seg_file}")
        for phase in phases:
          t0 = time.time()
          seg = read_seg_downsampled(seg_file, interior_xyz, halo=1,
                                     factor=a.factor, phase=phase)
          if seg.shape != gt.shape:
              raise ValueError(
                  f"downsampled prediction {seg.shape} != GT {gt.shape}; the mip0 and "
                  f"18 nm crops are supposed to cover the identical region at 2x.")
          vs, vm = voi(seg, gt)
          ar = adapted_rand(seg, gt)
          nseg = int(seg.max())
          nlab = int(len(np.unique(seg)))
          rows.append({"mt": mt, "phase": phase, "voi_split": float(vs),
                       "voi_merge": float(vm), "voi": float(vs + vm),
                       "adapted_rand_error": float(ar), "nseg": nseg,
                       "n_distinct_labels": nlab})
          tag = "" if len(phases) == 1 else f" phase={phase}"
          print(f"[{a.label}] mt={mt:.2f}{tag}  VOI={vs+vm:.4f} (split {vs:.4f} / "
                f"merge {vm:.4f})  ARerr={ar:.4f}  nseg={nseg}  labels={nlab}  "
                f"[{time.time()-t0:.0f}s]", flush=True)
          del seg
          gc.collect()
        if not a.keep_workdir:
            seg_file.unlink(missing_ok=True)

    ref_phase = phases[0]
    best = min((r for r in rows if r["phase"] == ref_phase), key=lambda r: r["voi"])
    if len(phases) > 1:
        print(f"[{a.label}] downsample-phase spread per mt:", flush=True)
        for mt in mts:
            v = [r["voi"] for r in rows if r["mt"] == mt]
            print(f"    mt={mt:.2f}  min={min(v):.4f} max={max(v):.4f} "
                  f"spread={max(v)-min(v):.4f}", flush=True)
    print(f"[{a.label}] best VOI: mt={best['mt']:.2f} -> {best['voi']:.4f}", flush=True)
    edge = best["mt"] in (mts[0], mts[-1])
    if edge:
        print(f"[{a.label}] WARNING: optimum at an END of the range -- widen it and "
              f"re-run, keeping the incumbent inside the new grid.", flush=True)

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(
            {"label": a.label, "affinity": str(a.affinity), "ws_high": a.ws_high,
             "ws_low": a.ws_low, "merge_function": "max", "space": "compressed",
             "whole_volume": True, "scored_on": "18nm grid", "gt": a.gt,
             "downsample_factor": a.factor, "downsample_phases": phases,
             "ws_peak_rss_gb": ws_peak_gb, "optimum_at_range_edge": edge,
             "ws_binary": str(a.ws_binary),
             "rows": rows, "best": best}, indent=2))
        print(f"wrote {a.json}", flush=True)

    if not a.keep_workdir:
        aff_raw_path.unlink(missing_ok=True)


main()
