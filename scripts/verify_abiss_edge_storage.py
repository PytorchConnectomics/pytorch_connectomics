"""Verify the `edge_storage` option end to end, including through YAML kwargs.

Three checks on one cached Pinky volume:

1. ``edge_storage=source`` reproduces the number measured with the ad-hoc roll
   that preceded this option (vol103, mt 0.58: VOI 0.4830).
2. ``edge_storage=destination`` (the default, i.e. unchanged behaviour) is
   materially worse, so the option is doing something and the default is not
   silently applying the shift.
3. The kwargs block in tutorials/neuron_microns_pinky_abiss.yaml, fed verbatim
   to ``decode_abiss``, produces the same segmentation as (1) -- proving the
   YAML -> cli_args -> ``--edge-storage`` -> runner path is wired.

Run with cwd = the tree holding the edited scripts/run_abiss_volume.py.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import yaml

from connectomics.decoding.decoders.abiss import decode_abiss
from pinky_test_ceiling import read_h5, score

EXPECTED_SOURCE_VOI = 0.4830
TOL = 1e-3


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("run_abiss_volume", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", type=Path,
                        default=Path("/projects/weilab/dataset/mito/microns/pinky/split/test"))
    parser.add_argument("--volume", default="pinky_vol103")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--abiss-home", required=True, type=Path)
    parser.add_argument("--runner", default=Path("scripts/run_abiss_volume.py"), type=Path)
    args = parser.parse_args()

    runner = load_runner(args.runner)
    ws_binary = args.abiss_home / "build" / "ws"
    aff = read_h5(args.results_dir / f"{args.volume}_image" / "raw_x1_head-aff_ch0-1-2.h5")
    aff = aff.astype(np.float32)
    gt = read_h5(args.gt_dir / f"{args.volume}_label.h5")

    cfg = yaml.safe_load(args.config.read_text())
    kwargs = cfg["default"]["decoding"]["steps"][0]["kwargs"]
    cli = dict(kwargs["cli_args"])
    mt = float(cli["ws_merge_threshold"])
    print(f"config kwargs: channels={kwargs['channels']} cli_args={cli}")

    segs = {}
    for storage in ("source", "destination"):
        segs[storage] = runner._run_abiss_ws(
            predictions_czyx=aff,
            ws_binary=ws_binary,
            ws_high_threshold=float(cli["ws_high_threshold"]),
            ws_low_threshold=float(cli["ws_low_threshold"]),
            ws_size_threshold=int(cli["ws_size_threshold"]),
            ws_dust_threshold=int(cli["ws_dust_threshold"]),
            boundary_flags=[1, 1, 1, 1, 1, 1],
            offset=0,
            channels=list(kwargs["channels"]),
            ws_merge_threshold=mt,
            ws_merge_function=str(cli["ws_merge_function"]),
            edge_storage=storage,
        )

    # (3) the same parameters, but routed the way a YAML decode step routes them
    yaml_seg = decode_abiss(
        aff,
        command=kwargs["command"],
        input_dataset=kwargs["input_dataset"],
        output_dataset=kwargs["output_dataset"],
        channels=list(kwargs["channels"]),
        timeout_sec=int(kwargs["timeout_sec"]),
        cli_args={**cli, "abiss_home": str(args.abiss_home)},
    )

    results = {k: score(np.ascontiguousarray(v), gt) for k, v in segs.items()}
    results["yaml/decode_abiss"] = score(np.ascontiguousarray(yaml_seg), gt)

    print(f"\n{'path':<24}{'frag':>8}{'ARE':>10}{'VOI':>10}")
    for name, s in results.items():
        print(f"{name:<24}{s['n_pred']:>8}{s['are']:>10.4f}{s['voi']:>10.4f}")

    ok = True
    src, dst, via_yaml = results["source"], results["destination"], results["yaml/decode_abiss"]

    if abs(src["voi"] - EXPECTED_SOURCE_VOI) > TOL:
        print(f"\nFAIL: edge_storage=source VOI {src['voi']:.4f} != "
              f"pre-refactor {EXPECTED_SOURCE_VOI:.4f}")
        ok = False
    else:
        print(f"\nPASS: edge_storage=source reproduces the pre-refactor roll "
              f"({src['voi']:.4f})")

    if dst["voi"] <= src["voi"]:
        print(f"FAIL: default (destination) VOI {dst['voi']:.4f} is not worse than "
              f"source {src['voi']:.4f}; the default may be shifting too")
        ok = False
    else:
        print(f"PASS: default (destination) is unchanged and worse "
              f"({dst['voi']:.4f} vs {src['voi']:.4f})")

    same = np.array_equal(np.ascontiguousarray(yaml_seg), np.ascontiguousarray(segs["source"]))
    if not same:
        print(f"FAIL: YAML path segmentation differs from the direct call "
              f"(VOI {via_yaml['voi']:.4f} vs {src['voi']:.4f})")
        ok = False
    else:
        print("PASS: YAML cli_args -> --edge-storage reaches the runner "
              "(segmentation is bit-identical)")

    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
