# PyTorch Connectomics

Read [AGENTS.md](AGENTS.md) for architecture, package ownership, strict config,
and the execution contract. This repository ships the public library, workflow
configs, and general-purpose tools. Research development happens in a private
repository; research workflows are not part of this distribution.

## Commands

Use the `pytc` conda environment. Do not install dependencies into `base`.

```bash
conda run -n pytc python -m build
conda run -n pytc python scripts/main.py --demo
conda run -n pytc python -m pytest tests -q
conda run -n pytc python scripts/validate_tutorial_configs.py --glob 'tutorials/*.yaml' --glob 'tutorials/**/*.yaml'
conda run -n pytc ruff check <changed_py_files>
conda run -n pytc ruff format --check <changed_py_files>
conda run -n pytc mypy --config-file .github/mypy_changed.ini <changed_py_files>
```

Keep changes focused. Preserve existing user edits. Do not add compatibility
facades or undeclared config reads. Run checks for the changed surface and
report unavailable checks explicitly.
