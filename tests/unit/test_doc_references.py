"""Check documented repository paths on the supported reference surfaces."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SURFACES = (
    "README.md",
    "QUICKSTART.md",
    "INSTALLATION.md",
    "TROUBLESHOOTING.md",
    "tutorials/README.md",
    "justfile",
    "AGENTS.md",
    "CLAUDE.md",
    "scripts/README.md",
)
# Explicit template syntax; literal missing paths are never exempted.
PLACEHOLDER_PATTERNS = (r"<[^>]+>", r"\{\{[^}]+\}\}", r"\*", r"^/path/to/")
REFERENCES = re.compile(
    r"(?<![\w/])(?:tutorials/[^\s`\"\x27()\[\]]+?\.(?:yaml|md|sh)"
    r"|scripts/[^\s`\"\x27()\[\]]+?\.(?:py|sh)"
    r"|prompts/[^\s`\"\x27()\[\]]+?\.md"
    r"|[A-Za-z][A-Za-z_\d-]*\.md)(?![\w.])"
)


def test_documented_paths_exist():
    worktree = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True
    )
    if worktree.returncode or worktree.stdout.strip() != "true":
        pytest.skip(
            "Documentation reference inventory requires a Git work tree (unavailable in source archives)"
        )
    inventory = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    files = {ROOT / name for name in SURFACES}
    files.update(
        ROOT / name
        for name in inventory
        if (ROOT / name).is_file()
        and (
            (Path(name).parent == Path("prompts") and name.endswith(".md"))
            or (name.startswith("tutorials/") and Path(name).name == "README.md")
        )
    )
    missing = []
    for path in sorted(files):
        assert path.exists(), str(path)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            # Omit external URLs; only local repository references are in scope.
            line = re.sub(r"https?://[^\s<>`\"\x27)]+", "", line)
            for match in REFERENCES.finditer(line):
                reference = match.group()
                if any(re.search(pattern, reference) for pattern in PLACEHOLDER_PATTERNS):
                    continue
                if not (ROOT / reference).exists():
                    missing.append(f"{path.relative_to(ROOT)}:{number}: {reference}")
    assert not missing, "\n".join(missing)
