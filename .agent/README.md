# `.agent/` — Internal Knowledge Base

Project-local docs for the codebase: architecture contracts, recent landed
PR summaries, how-to guides, external library references. Read by humans
and AI coding agents working in this repo.

This directory is **not** Claude Code's project-local config (no
`settings.json`, `agents/`, `commands/`, or `skills/` live here). Project
instructions for Claude Code live at the repo-root `CLAUDE.md`.

## Layout

| dir | what's in it | when to read it |
|---|---|---|
| `architecture/` | Active v3 architecture contract + audit | Touching package boundaries, schema, or cross-stage APIs |
| `changes/` | Rolling landed-PR summaries + reviews | Recent feature / refactor history (last few weeks) |
| `guides/` | Practical how-to docs (augmentation, finetuning, NaN debugging, …) | Onboarding or troubleshooting a specific area |
| `integrations/` | External-library integration notes (CellMap, MedNeXt, nnUNet, Optuna) | Wiring up an external dep |
| `reference/` | External-library / repo references (BANIS, DeepEM, waterz, …) | Comparing pytc against a reference implementation |
| `banis/` | BANIS-specific deep-dives (affinity convention, inference parity) | Working on the NISB BANIS reproduction tutorials |
| `benchmark/` | Benchmark gap-analysis docs (SNEMI3D vs DeepEM) | Tracking benchmark performance |

## What's NOT here

- Source code (under `connectomics/`).
- User-facing tutorials (under `tutorials/`).
- Test code (under `tests/`).
- Top-level developer guide (`CLAUDE.md`, `AGENTS.md`, `QUICKSTART.md`,
  `README.md` at repo root).
- Claude Code per-user config (lives at `~/.claude/`).

## House rules

- **`changes/`** is rolling — landed-PR summaries get added; older entries
  graduate to `architecture/` (if they pin contract) or get pruned (if
  superseded).
- **`architecture/`** is the contract. Keep tight. The v3 docs are cited
  from `CLAUDE.md` and `AGENTS.md`; renaming them requires updating those.
- **`reference/`** files describe external code, not pytc internals. They
  shouldn't drift when pytc changes.
- **No code paths read from this directory**. Everything here is
  documentation. If a pytc Python file links here in a docstring, it's a
  pointer for human readers only.
