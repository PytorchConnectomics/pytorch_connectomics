"""Command-line interface for inspectable segmentation checkpoint passes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .engine import (
    apply_pass,
    describe_pass,
    load_description,
    load_plan,
    load_result,
    plan_pass,
    run_pass,
    verify_pass,
)
from .serialization import load_spec
from .verification import verification_passed


def _one_pass(path: Path) -> Mapping[str, Any]:
    raw = load_spec(path)
    if "passes" not in raw:
        return raw
    passes = raw["passes"]
    if not isinstance(passes, list) or len(passes) != 1 or not isinstance(passes[0], dict):
        raise ValueError("the single-pass CLI requires exactly one pass specification")
    merged = dict(passes[0])
    if "checkpoint_id" in raw:
        merged.setdefault("checkpoint_id", raw["checkpoint_id"])
    return merged


def _require_anchor_totals(expected: str, supplied: Path) -> None:
    if Path(expected).resolve() != supplied.resolve():
        raise ValueError(
            f"anchor totals mismatch: record requires {expected}, supplied {supplied.resolve()}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="checkpoint")
    sub = parser.add_subparsers(dest="command", required=True)

    describe = sub.add_parser("describe", help="emit immutable observations and certificates")
    describe.add_argument("--spec", type=Path, required=True)
    describe.add_argument("--output-dir", type=Path, required=True)
    describe.add_argument("--min-share", type=float, required=True)

    plan = sub.add_parser("plan", help="write an inspectable action plan")
    plan.add_argument("--spec", type=Path, required=True)
    plan.add_argument("--description", type=Path, required=True)
    plan.add_argument("--anchor-totals", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--min-share", type=float, required=True)

    apply = sub.add_parser("apply", help="execute a previously serialized plan")
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--anchor-totals", type=Path, required=True)
    apply.add_argument("--output-dir", type=Path, required=True)

    verify = sub.add_parser("verify", help="verify a prior execution and replay determinism")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--anchor-totals", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)

    run = sub.add_parser("run", help="describe, plan, apply, and verify")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--min-share", type=float, required=True)
    run.add_argument(
        "--dry-run", action="store_true", help="stop after writing the inspectable plan"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        describe_pass(_one_pass(args.spec), args.output_dir, min_share=args.min_share)
        return 0
    if args.command == "plan":
        description = load_description(args.description)
        _require_anchor_totals(description.anchor_totals_artifact.uri, args.anchor_totals)
        plan_pass(
            _one_pass(args.spec),
            description,
            args.output,
            min_share=args.min_share,
        )
        return 0
    if args.command == "apply":
        plan = load_plan(args.plan)
        _require_anchor_totals(plan.anchor_totals_artifact.uri, args.anchor_totals)
        apply_pass(plan, args.output_dir)
        return 0
    if args.command == "verify":
        plan = load_plan(args.plan)
        _require_anchor_totals(plan.anchor_totals_artifact.uri, args.anchor_totals)
        result = verify_pass(plan, load_result(args.result), args.output_dir)
        return 0 if verification_passed(result) else 1
    spec = _one_pass(args.spec)
    if args.dry_run:
        description = describe_pass(spec, args.output_dir, min_share=args.min_share)
        plan_pass(
            spec,
            description,
            args.output_dir / "plan.json",
            min_share=args.min_share,
        )
        return 0
    _plan, result = run_pass(spec, args.output_dir, min_share=args.min_share)
    return 0 if verification_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
