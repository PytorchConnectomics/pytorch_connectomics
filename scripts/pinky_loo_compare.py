"""Leave-one-out comparison of the Pinky decoders.

Every threshold here was chosen by looking at the test volumes, so the tuned
optimum is optimistic. For each volume the parameter is re-chosen on the other
five and the held-out volume is scored at that choice; the LOO column is the
number to quote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path, param_key: str, label: str) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    out = []
    for row in payload["rows"]:
        # the CC sweep keys volumes by result-dir name (``pinky_vol101_image``),
        # the ABISS driver by GT stem (``pinky_vol101``); normalise to the stem.
        volume = row["volume"]
        if volume.endswith("_image"):
            volume = volume[: -len("_image")]
        out.append({
            "decoder": label,
            "volume": volume,
            "param": float(row[param_key]),
            "voxels": row["voxels"],
            "voi": row["base"]["voi"],
            "are": row["base"]["are"],
            "voi_split": row["base"]["voi_split"],
            "voi_merge": row["base"]["voi_merge"],
            "oracle_voi": row["oracle_merge"]["voi"],
        })
    return out


def report(rows: list[dict], label: str, metric: str = "voi") -> None:
    if not rows:
        print(f"[{label}] no rows")
        return
    volumes = sorted({r["volume"] for r in rows})
    params = sorted({r["param"] for r in rows})
    table = {(r["volume"], r["param"]): r for r in rows}
    # only keep params measured on every volume, so the choice is comparable
    params = [p for p in params if all((v, p) in table for v in volumes)]
    if not params:
        print(f"[{label}] no parameter measured on all {len(volumes)} volumes; "
              f"per-volume coverage: "
              + ", ".join(f"{v}:{sum(1 for p in sorted({r['param'] for r in rows}) if (v, p) in table)}"
                          for v in volumes))
        return

    weights = {v: table[(v, params[0])]["voxels"] for v in volumes}
    total = sum(weights.values())

    def pooled(param: float, subset: list[str]) -> float:
        w = sum(weights[v] for v in subset)
        return sum(table[(v, param)][metric] * weights[v] for v in subset) / w

    best_all = min(params, key=lambda p: pooled(p, volumes))
    print(f"\n[{label}]  volumes={len(volumes)}  params={params}")
    print(f"{'volume':<16}{'LOO param':>11}{'LOO ' + metric:>11}{'tuned':>9}{'oracle':>9}")
    loo_vals, tuned_vals = [], []
    for v in volumes:
        others = [o for o in volumes if o != v]
        p_loo = min(params, key=lambda p: pooled(p, others))
        loo = table[(v, p_loo)][metric]
        tuned = min(table[(v, p)][metric] for p in params)
        loo_vals.append(loo * weights[v])
        tuned_vals.append(tuned * weights[v])
        print(f"{v:<16}{p_loo:>11.2f}{loo:>11.4f}{tuned:>9.4f}"
              f"{table[(v, p_loo)]['oracle_voi']:>9.4f}")
    print(f"{'voxel-weighted':<16}{'':>11}{sum(loo_vals)/total:>11.4f}"
          f"{sum(tuned_vals)/total:>9.4f}")
    print(f"  single global best param = {best_all:.2f} "
          f"({metric} {pooled(best_all, volumes):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--metric", default="voi", choices=("voi", "are"))
    args = parser.parse_args()
    d = args.results_dir

    cc = load_rows(d / "decode_threshold_sweep.json", "threshold", "affinity CC")
    abiss = (load_rows(d / "abiss_max_small_report.json", "merge_threshold", "ABISS max")
             + load_rows(d / "abiss_max_vol501_report.json", "merge_threshold", "ABISS max"))
    abiss_mean = load_rows(d / "abiss_mean_small_report.json", "merge_threshold", "ABISS mean")

    for label, rows in (("affinity CC", cc), ("ABISS max", abiss), ("ABISS mean", abiss_mean)):
        report(rows, label, metric=args.metric)

    # ABISS max restricted to the five small volumes, so CC gets a like-for-like row
    small = {r["volume"] for r in abiss_mean} or {
        "pinky_vol101", "pinky_vol102", "pinky_vol103", "pinky_vol104", "pinky_vol503"}
    report([r for r in cc if r["volume"] in small], "affinity CC (5 small only)", metric=args.metric)
    report([r for r in abiss if r["volume"] in small], "ABISS max (5 small only)", metric=args.metric)


if __name__ == "__main__":
    main()
