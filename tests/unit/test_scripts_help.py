"""Every shipped source script must expose help without running its workload."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Both failures were reproduced at the run's base revision. Require the known
# failure text so that unrelated import errors cannot hide behind this allowlist.
BASE_HELP_FAILURES = {
    "scripts/images_to_h5.py": "Usage: python scripts/images_to_h5.py",
    "scripts/profile_dataloader.py": "FileNotFoundError: Config file not found:",
}


def _script_paths() -> list[str]:
    in_work_tree = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if in_work_tree.returncode != 0 or in_work_tree.stdout.strip() != "true":
        pytest.skip(
            "Script inventory requires a git work tree to include tracked and new files.",
            allow_module_level=True,
        )
    names = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "scripts",
            ],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    return sorted({name for name in names if name.endswith(".py") and (ROOT / name).is_file()})


@pytest.mark.parametrize("script", _script_paths())
def test_script_help(script: str) -> None:
    import connectomics

    assert Path(connectomics.__file__).resolve().is_relative_to(ROOT), connectomics.__file__
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 and script in BASE_HELP_FAILURES:
        assert result.returncode == 1 and BASE_HELP_FAILURES[script] in output, output
    else:
        assert result.returncode == 0, f"{script} --help exited {result.returncode}:\n{output}"
