#!/usr/bin/env python3
"""Pick the ABISS merge threshold for a moe volume, GT-free.

WHAT TRANSFERS IS THE PERCENTILE, NOT THE VALUE. The merge threshold is compared
against affinities, and the affinity distribution is not constant across this
batch: its median drifts down with expansion factor (0.53 on the in-domain IST
val, 0.47-0.49 at 18x/22x, 0.42 at 32x), because image contrast falls as more of
the field resolves into boundary. A fixed absolute threshold therefore means a
different operating point on every volume. Two demonstrations, both measured:

  * IST's swept 0.47 applied to the first moe volume produced a largest segment
    of 975 um^3 -- 17.35% of the whole 5624 um^3 field -- with a second at 6.58%
    and the third down at 0.63%. Two runaway merge chains, not a size
    distribution.
  * moe's 0.60, in turn, does not transfer to the rest of the batch. On the 28x
    and 32x volumes, agglomeration STOPS ENTIRELY at 0.68-0.69: every threshold
    at or above that returns the raw watershed unchanged (identical label count
    and coverage), because no region-graph edge is that strong. 0.60 is below
    that on some volumes and near it on others.

So the anchor is carried as a percentile of each volume's *own* affinity.
`--percentile` defaults to **62.89**, which is what the published operating point
(0.60 on `ExPID96_2ndgel_S1_40XW001_18x`) corresponds to on that volume. This is
the same reasoning that already makes `ws_high`/`ws_low` percentiles rather than
absolutes -- the merge threshold was simply left behind.

WHY NOT THE ORIGINAL "KNEE" RULE. The first run picked 0.60 by hand as the lowest
threshold whose largest segment stayed under 2% of the volume. That rule does not
survive the batch, for two independent reasons:

  * Share is not comparable across fields. These volumes run 1035-5624 um^3, so a
    fixed 2% cap is three times stricter on a 32x field than on the 18x one it
    was calibrated on.
  * Size alone cannot tell a merge chain from a soma. A 12 um box can be
    dominated by a single cell body. Volumes that have one show a "runaway"
    segment at every threshold below saturation, and the rule then selects the
    saturation point itself -- i.e. the un-agglomerated watershed, coverage 0.29
    against 0.71 for the published volume.

The sweep table and `inspect_top_segments.py` (which separates chains from somata
by SHAPE -- a chain threads the field through thin bridges, so its bounding box
is near the full volume at a few percent fill) are kept as the evidence a human
reads before accepting the pick. `plateau` in the table marks thresholds where
agglomeration is doing nothing.

`--ws-merge-thresholds` runs the watershed and region graph ONCE and agglomerates
per threshold, so the whole grid costs barely more than a single decode.

    python tutorials/neuron_liconn_moe/sweep_merge_threshold.py --volume <name>
    python tutorials/neuron_liconn_moe/sweep_merge_threshold.py --volume <name> --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volumes as V  # noqa: E402

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# Context grid. The percentile-matched threshold is inserted into it, so these
# points exist to show the neighbourhood -- where the merge chains switch on
# below, and where agglomeration saturates above -- not to be selected from.
DEFAULT_THRESHOLDS = "0.40,0.47,0.55,0.60,0.65,0.70,0.75"

# The published operating point (0.60 on ExPID96_2ndgel_S1_40XW001_18x) expressed
# as a percentile of that volume's own affinity. Everything else inherits it.
PUBLISHED_PERCENTILE = 62.89


def run_sweep(aff: Path, out_dir: Path, thresholds: str, workdir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable, str(here / "scripts/run_abiss_volume.py"),
        "--input", str(aff),
        "--output", str(out_dir / "seg.h5"),
        # The vendored ABISS build (build/ws) is a compiled artifact that lives
        # only in the main checkout, so this path is absolute on purpose. In the
        # cloud image there is no vendored copy: ABISS is built into the image
        # at /opt/abiss and `ABISS_HOME` points there (gcloud/Dockerfile).
        "--abiss-home", os.environ.get("ABISS_HOME") or str(V.REPO / "lib/abiss"),
        # NOTE: `workdir`/`timeout_sec` in 2_abiss.yaml are kwargs of the
        # `decode_abiss` decoder, not flags of this script.
        "--abiss-workdir", str(workdir / "ws_scratch"),
        "--input-dataset", "main", "--output-dataset", "main",
        # This repo stores channel c as the edge along array axis c (z, y, x);
        # ABISS `ws` reads (X, Y, Z, C) with channel 0 = X edge, so reverse.
        "--channels", "2,1,0",
        # `banis` stores edge (i, i+1) at voxel i; `ws` reads it as (i-1, i).
        "--edge-storage", "source",
        # Percentiles: self-adapting, unaffected by the scale_sigmoid compression.
        "--ws-high-threshold", "94%",
        "--ws-low-threshold", "20%",
        "--ws-size-threshold", "10000000",
        "--ws-dust-threshold", "200",
        "--ws-merge-function", "max",
        "--ws-merge-thresholds", thresholds,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(here))


def check_nonempty(out_dir: Path, n: int) -> None:
    """Fail loudly on the silent all-zero decode.

    ABISS can log correct supervoxel counts and still have its `seg_*.data`
    payload come back all zeros from /projects under load -- observed on 3 of 7
    concurrent sweeps here, and once before on a single decode (see README).
    Nothing downstream notices: the h5 is written, it is merely empty, and it
    then gets meshed and published. Identical file sizes across thresholds that
    should differ are the tell.

    Do not co-schedule many of these. The re-run that fixed it was serialised.
    """
    empty = []
    for i in range(n):
        path = out_dir / f"seg_mt{i}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            if int(np.asarray(f["main"][f["main"].shape[0] // 2]).max()) == 0:
                d = f["main"]
                if int(np.asarray(d[::max(1, d.shape[0] // 8)]).max()) == 0:
                    empty.append(path.name)
    if empty:
        raise SystemExit(
            f"{len(empty)} of {n} sweep outputs in {out_dir} have no foreground: "
            f"{', '.join(empty)}\nThis is a transient /projects write, not a decode "
            "bug -- delete the directory and re-run with less concurrency."
        )


def measure(out_dir: Path, thresholds: list[str], vox_nm3: float,
            min_share: float, min_span: float) -> list[dict]:
    rows = []
    for i, mt in enumerate(thresholds):
        path = out_dir / f"seg_mt{i}.h5"
        if not path.exists():
            print(f"{mt:>6} MISSING {path}")
            continue
        with h5py.File(path, "r") as f:
            seg = np.asarray(f["main"])
        ids, cnt = np.unique(seg, return_counts=True)
        fg_ids, fg_cnt = ids[ids != 0], cnt[ids != 0]
        fg = np.sort(fg_cnt)[::-1]
        frac = fg / seg.size
        rows.append({
            "mt": float(mt), "index": i, "path": str(path), "labels": int(len(fg)),
            "largest_share": float(frac[0]),
            "second_share": float(frac[1]) if len(frac) > 1 else 0.0,
            "n_above_2pct": int((frac > 0.02).sum()),
            "covered": float(fg.sum() / seg.size),
            "p50_um3": float(np.percentile(fg, 50) * vox_nm3 / 1e9),
            "total_um3": float(seg.size * vox_nm3 / 1e9),
            "shape": list(seg.shape),
            "chains": find_chains(seg, fg_ids, fg_cnt, min_share, min_span),
        })
        del seg
    return rows


def find_chains(seg, fg_ids, fg_cnt, min_share: float, min_span: float) -> list[dict]:
    """Segments that are big AND span the whole field: merge chains.

    Calibrated on the published volume, whose 0.47 decode is known bad and whose
    0.60 is the published operating point. At 0.47 the top two segments (975 and
    370 um^3) have a bounding box covering EVERY axis of the field in full; at
    0.55 and above nothing spans more than 0.78 of it. Size alone does not
    separate the two -- a soma can legitimately fill a 12 um box -- and share
    alone is not comparable across fields that differ threefold in volume.

    Both conditions are needed. A thin process can touch all six faces and score
    span 1.000 at 21 um^3 (seen on the 22x volume at 0.55); it is not a chain.
    """
    out = []
    for lid, c in zip(fg_ids, fg_cnt):
        share = float(c) / seg.size
        if share < min_share:
            continue
        m = seg == lid
        span = 1.0
        for ax in range(3):
            hit = m.any(axis=tuple(a for a in range(3) if a != ax))
            idx = np.flatnonzero(hit)
            span *= float(idx[-1] - idx[0] + 1) / seg.shape[ax]
        if span >= min_span:
            out.append({"label": int(lid), "share": share, "bbox_span_frac": span})
    return out


def affinity_percentile(aff: Path, pct: float, n_slices: int = 8) -> float:
    """Absolute affinity value at `pct` of this volume's own distribution.

    Sampled on evenly spaced Y planes across all three channels; the full array
    is 1-3 GB and the percentile is stable to <0.002 over this many slices.
    """
    with h5py.File(aff, "r") as f:
        a = f["main"]
        ys = np.linspace(0, a.shape[1] - 1, n_slices).astype(int)
        v = np.concatenate([np.asarray(a[:, y]).astype(np.float32).ravel() for y in ys])
    return float(np.percentile(v, pct))


def mark_plateau(rows: list[dict]) -> None:
    """Flag thresholds where agglomeration is doing nothing.

    Above the strongest region-graph edge, every threshold returns the raw
    watershed: identical label count, identical coverage. Selecting there is not
    an operating point, it is a decode that skipped its own merge step.
    """
    for i, r in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        prv = rows[i - 1] if i else None
        r["plateau"] = bool(
            (nxt and nxt["labels"] == r["labels"]) or (prv and prv["labels"] == r["labels"])
        )


def pick(rows: list[dict], target: float) -> dict:
    """Percentile-matched threshold, vetoed by the chain test and the plateau.

    The percentile sets the intended operating point; the veto is a guard against
    a demonstrable failure, not a second opinion. Stepping UP is the safe
    direction -- a higher threshold merges less -- so on a veto the pick walks up
    the grid to the first chain-free threshold that is still doing agglomeration.
    """
    ordered = sorted(rows, key=lambda r: r["mt"])
    start = min(range(len(ordered)), key=lambda i: abs(ordered[i]["mt"] - target))
    for i in range(start, len(ordered)):
        r = ordered[i]
        if not r["chains"] and not r.get("plateau"):
            return dict(r, selected_by="percentile" if i == start else "chain-veto",
                        target_mt=target, steps_up=i - start)
    # Nothing clean below the plateau: keep the percentile pick and say so.
    return dict(ordered[start], selected_by="percentile-unvetoed", target_mt=target,
                steps_up=0, no_clean_threshold=True)


def report(rows: list[dict], chosen: dict) -> None:
    if rows:
        print(f"       volume = {rows[0]['total_um3']:.0f} um^3, {tuple(rows[0]['shape'])}")
    print(f"{'mt':>6} {'labels':>9} {'largest %vol':>13} {'largest um^3':>13} "
          f"{'2nd %vol':>9} {'covered':>8} {'chains':>7}  note")
    for r in rows:
        note = "plateau" if r.get("plateau") else ""
        if r["mt"] == chosen["mt"]:
            note = (note + "  <- chosen").strip()
        print(f"{r['mt']:>6.3f} {r['labels']:>9} {r['largest_share']*100:>12.2f}% "
              f"{r['largest_share']*r['total_um3']:>13.1f} {r['second_share']*100:>8.2f}% "
              f"{r['covered']:>7.3f} {len(r['chains']):>7}  {note}")
    print(f"\nselected by {chosen['selected_by']}: percentile target mt "
          f"{chosen['target_mt']:.4f}" +
          (f", stepped up {chosen['steps_up']} grid point(s) off a merge chain"
           if chosen.get("steps_up") else ""))
    if chosen.get("no_clean_threshold"):
        print("WARNING: no threshold below the saturation plateau is chain-free on "
              "this volume. The published layer will contain at least one merge chain.")
    if chosen.get("plateau"):
        print("WARNING: the chosen threshold is on the saturation plateau -- ABISS "
              "agglomeration is a no-op there and this is the raw watershed.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--volume", help="moe volume name (see volumes.py)")
    ap.add_argument("--index", type=int, help="index into volumes.PENDING")
    ap.add_argument("--affinity", type=Path, help="override the step-1 affinity path")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                    help="context grid; the percentile-matched value is added to it")
    ap.add_argument("--percentile", type=float, default=PUBLISHED_PERCENTILE,
                    help="percentile of this volume's own affinity to decode at")
    ap.add_argument("--chain-min-share", type=float, default=0.02,
                    help="a chain must hold at least this share of the field")
    ap.add_argument("--chain-min-span", type=float, default=0.90,
                    help="...and its bbox must cover at least this much of it")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    name = a.volume or V.PENDING[a.index]
    aff = a.affinity or V.affinity_h5(name)
    out_dir = a.out_dir or V.sweep_dir(name)
    vox_nm3 = float(np.prod(V.plan(name)["spacing_zyx"]))
    print(f"volume {name}\naffinity {aff}\nvoxel {vox_nm3/1e3:.3f} x10^3 nm^3", flush=True)

    matched = round(affinity_percentile(aff, a.percentile), 4)
    # Insert it into the context grid, but do NOT let it sit next to a
    # near-identical point: `seg_mt{i}.h5` is indexed by position, so an extra
    # entry at 0.6001 beside 0.60 shifts the label of every file after it.
    grid = sorted(float(x) for x in a.thresholds.split(","))
    if all(abs(t - matched) > 0.005 for t in grid):
        grid = sorted(grid + [matched])
    else:
        matched = min(grid, key=lambda t: abs(t - matched))
    print(f"percentile {a.percentile} of this volume's affinity -> mt {matched}\n"
          f"grid {grid}", flush=True)

    if not a.report_only:
        run_sweep(aff, out_dir, ",".join(f"{t:g}" for t in grid), out_dir)

    check_nonempty(out_dir, len(grid))
    rows = measure(out_dir, [f"{t:g}" for t in grid], vox_nm3,
                   a.chain_min_share, a.chain_min_span)
    if not rows:
        raise SystemExit("no sweep outputs found")
    mark_plateau(rows)
    chosen = pick(rows, matched)
    report(rows, chosen)

    final = V.work_dir(name) / f"{V.layer_name(name, chosen['mt'])}.h5"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        final.unlink()
    os.link(chosen["path"], final)
    summary = {"volume": name, "affinity": str(aff), "chosen": chosen,
               "percentile": a.percentile, "segmentation": str(final), "rows": rows,
               "spacing_zyx_nm": V.plan(name)["spacing_zyx"]}
    (V.work_dir(name) / "mt_sweep.json").write_text(json.dumps(summary, indent=2))
    print(f"\nchosen mt {chosen['mt']:.2f} -> {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
