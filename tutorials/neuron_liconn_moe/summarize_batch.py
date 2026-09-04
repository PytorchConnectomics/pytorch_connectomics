#!/usr/bin/env python3
"""One table for the whole moe batch: prep grid, affinity, chosen threshold, layer.

Reads each volume's `mt_sweep.json` (written by `sweep_merge_threshold.py`) and
the staged precomputed tree, so it reflects what is actually on disk rather than
what was intended. Run it after every stage to see where the batch stands.

    python tutorials/neuron_liconn_moe/summarize_batch.py
    python tutorials/neuron_liconn_moe/summarize_batch.py --urls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volumes as V  # noqa: E402


def row(name: str, with_sizes: bool = False) -> dict:
    p = V.plan(name)
    r = {"name": name, "shape": p["shape"],
         "spacing": [round(v, 4) for v in p["spacing_zyx"]],
         "prepared": V.prepared_image(name).exists(),
         "affinity": V.affinity_h5(name).exists()}
    sj = V.work_dir(name) / "mt_sweep.json"
    if name in V.LEGACY and not sj.exists():
        r["mt"] = V.LEGACY[name]["merge_threshold"]
        r["layer"] = V.layer_name(name, r["mt"])
    elif sj.exists():
        s = json.loads(sj.read_text())
        c = s["chosen"]
        r.update(mt=c["mt"], labels=c["labels"], largest=c["largest_share"],
                 covered=c["covered"], selected_by=c.get("selected_by", "?"),
                 chains=len(c.get("chains", [])),
                 no_clean=c.get("no_clean_threshold", False),
                 layer=V.layer_name(name, c["mt"]))
    if r.get("layer"):
        staging = V.OUT_ROOT / "precomputed" / r["layer"]
        r["staged"] = (staging / "info").exists()
        # Walking a precomputed tree is ~10^5 small stat()s on /projects and
        # takes minutes; only do it when asked.
        if with_sizes and r["staged"]:
            r["staged_bytes"] = sum(
                (Path(d) / f).stat().st_size for d, _, fs in os.walk(staging) for f in fs)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls", action="store_true", help="print neuroglancer layer URLs only")
    ap.add_argument("--sizes", action="store_true",
                    help="also total the staged precomputed trees (slow on /projects)")
    a = ap.parse_args()

    rows = [row(n, with_sizes=a.sizes and not a.urls) for n in V.VOLUMES]
    if a.urls:
        for r in rows:
            if r.get("layer"):
                print(V.layer_url(r["layer"]))
        return 0

    print(f"{'volume':34s} {'shape':>18s} {'spacing ZYX nm':>26s} {'prep':>5s} {'aff':>4s} "
          f"{'mt':>5s} {'labels':>9s} {'largest%':>9s} {'cov':>6s} {'layer':>9s}")
    for r in rows:
        mt = f"{r['mt']:.2f}" if "mt" in r else "-"
        lab = f"{r['labels']}" if "labels" in r else "-"
        lar = f"{r['largest']*100:.2f}" if "largest" in r else "-"
        cov = f"{r['covered']:.3f}" if "covered" in r else "-"
        gb = (f"{r['staged_bytes']/1e9:.2f}" if r.get("staged_bytes")
              else ("built" if r.get("staged") else "-"))
        warn = f"  {r['selected_by']}" if "selected_by" in r else ""
        if r.get("no_clean"):
            warn += "  !! contains a merge chain"
        print(f"{r['name']:34s} {str(r['shape']):>18s} {str(r['spacing']):>26s} "
              f"{'y' if r['prepared'] else '-':>5s} {'y' if r['affinity'] else '-':>4s} "
              f"{mt:>5s} {lab:>9s} {lar:>9s} {cov:>6s} {gb:>9s}{warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
