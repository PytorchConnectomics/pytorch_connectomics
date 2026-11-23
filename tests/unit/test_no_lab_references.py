"""Prevent deployment files from depending on private research resources."""

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SURFACES = {
    "README.md",
    "QUICKSTART.md",
    "INSTALLATION.md",
    "TROUBLESHOOTING.md",
    "tutorials/README.md",
    "justfile",
    "AGENTS.md",
    "CLAUDE.md",
    "pyproject.toml",
}
PATTERN = re.compile(
    r"/projects/weilab|/data/donglai|/home/donglai|weidf|liupeng|"
    r"dev/zebrafinch|lib/abiss|lib/em_erl|lib/waterz|"
    r"zebrafinch|matchguard|gtfree|arm0_96|j0126|moritz|pinky|seuron|lessons L\d+|HANDOFF_",
    re.I,
)
# The guard itself is the only attribution exception. The paths below are the j0126
# research workflow, restored so its one-command reproduction runs from this repository;
# they are exempt as a whole and nothing outside them is.
SELF = "tests/unit/test_no_lab_references.py"
J0126_WORKFLOW = (
    "connectomics/data/keep_mask.py",
    "connectomics/decoding/error_correction/",
    "connectomics/playbooks/",
    "connectomics/runtime/abiss_chunk.py",
    "connectomics/runtime/volume_pipeline.py",
    "connectomics/utils/yaml_config.py",
    "scripts/build_j0126_keep_mask.py",
    "scripts/evaluate_j0126.py",
    "scripts/run_abiss_chunk.py",
    "scripts/run_error_correction.py",
    "scripts/run_j0126.py",
    "scripts/run_playbook.py",
    "tests/fixtures/playbook_baseline/j0126.json",
    "tests/unit/test_abiss_chunk_executor.py",
    "tests/unit/test_cube_playbook.py",
    "tests/unit/test_error_correction_contact_spacing.py",
    "tests/unit/test_error_correction_workflow.py",
    "tests/unit/test_keep_mask.py",
    "tests/unit/test_volume_pipeline.py",
    "tutorials/_base/abiss.yaml",
    "tutorials/neuron_j0126/",
)


def test_no_lab_references():
    worktree = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True
    )
    if worktree.returncode or worktree.stdout.strip() != "true":
        pytest.skip(
            "Lab reference inventory requires a Git work tree (unavailable in source archives)"
        )
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    failures = []
    for name in paths:
        if (
            name == SELF
            or name.startswith(J0126_WORKFLOW)
            or not (
                name in SURFACES
                or name.startswith(
                    ("connectomics/", "scripts/", "tutorials/", "tests/", "prompts/")
                )
            )
        ):
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                failures.append(f"{name}:{number}: {line.strip()}")
        if name.startswith("connectomics/") and name.endswith(".py"):
            tree = ast.parse(text)
            parents = {
                child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
            }
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Path"
                    and any(isinstance(arg, ast.Name) and arg.id == "__file__" for arg in node.args)
                ):
                    continue
                expression = node
                while isinstance(parents.get(expression), ast.expr):
                    expression = parents[expression]
                source = ast.unparse(expression)
                if re.search(r"parents\[|parent\.parent", source) and re.search(
                    r"[\"\x27](tutorials|scripts|dev|lib)(?:/|[\"\x27])", source
                ):
                    failures.append(f"{name}:{node.lineno}: path escapes package: {source}")
    assert not failures, "\n".join(failures)
