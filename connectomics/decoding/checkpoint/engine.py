"""Orchestration for DESCRIBE -> CERTIFY -> ACT -> VERIFY."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .actions import execute_plan
from .registry import create_operator
from .schema import CheckpointPlan, CheckpointResult, DescriptionBundle
from .serialization import read_json, write_json
from .verification import verify_result


def describe_pass(
    spec: Mapping[str, Any], output_dir: str | Path, *, min_share: float
) -> DescriptionBundle:
    output = Path(output_dir)
    operator = create_operator(spec, min_share=min_share)
    description = operator.describe(output)
    write_json(output / "description.json", description)
    return description


def plan_pass(
    spec: Mapping[str, Any],
    description: DescriptionBundle,
    output_path: str | Path,
    *,
    min_share: float,
) -> CheckpointPlan:
    operator = create_operator(spec, min_share=min_share)
    plan = operator.plan(description)
    write_json(output_path, plan)
    return plan


def apply_pass(plan: CheckpointPlan, output_dir: str | Path) -> CheckpointResult:
    result = execute_plan(plan, output_dir)
    write_json(Path(output_dir) / "result.json", result)
    return result


def verify_pass(
    plan: CheckpointPlan,
    result: CheckpointResult,
    output_dir: str | Path,
) -> CheckpointResult:
    verified = verify_result(plan, result, Path(output_dir) / "verification")
    write_json(Path(output_dir) / "verified_result.json", verified)
    return verified


def run_pass(
    spec: Mapping[str, Any], output_dir: str | Path, *, min_share: float
) -> tuple[CheckpointPlan, CheckpointResult]:
    output = Path(output_dir)
    description = describe_pass(spec, output, min_share=min_share)
    plan = plan_pass(spec, description, output / "plan.json", min_share=min_share)
    result = apply_pass(plan, output)
    return plan, verify_pass(plan, result, output)


def run_pipeline(
    pass_specs: Sequence[Mapping[str, Any]], output_dir: str | Path, *, min_share: float
) -> tuple[CheckpointResult, ...]:
    """Run independent passes sequentially, preserving a directory per pass.

    Only ``nucleus_anchor`` is registered today.  A later pass can consume a
    preceding pass's sparse segmentation delta by declaring that artifact in
    its own input specification; the runner does not fuse pass policies.
    """

    results = []
    root = Path(output_dir)
    for index, spec in enumerate(pass_specs):
        pass_id = str(spec.get("pass_id", spec.get("operator", index)))
        _plan, result = run_pass(spec, root / f"{index:02d}_{pass_id}", min_share=min_share)
        results.append(result)
    return tuple(results)


def load_description(path: str | Path) -> DescriptionBundle:
    return read_json(path, DescriptionBundle)


def load_plan(path: str | Path) -> CheckpointPlan:
    return read_json(path, CheckpointPlan)


def load_result(path: str | Path) -> CheckpointResult:
    return read_json(path, CheckpointResult)


__all__ = [
    "apply_pass",
    "describe_pass",
    "load_description",
    "load_plan",
    "load_result",
    "plan_pass",
    "run_pass",
    "run_pipeline",
    "verify_pass",
]
