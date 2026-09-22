# review_v0 raw examination notes (planner-owned, performed in-session)

`review_vN` is planner-owned and the planner is this session, so this stage ran in-session by
reading files directly rather than through a companion CLI. There is therefore no external
reviewer transcript; this file is the raw evidence gathered, recorded so the attested summary in
`artifacts/review_v0.md` can be checked against it.

## Commands run and their output

```text
$ python -m pytest tests/unit/test_abiss_nucleus_competition.py -q
collected 18 items
..................                                                       [100%]
18 passed in 1.20s
```

Codex claimed "18 passed"; independently reproduced. Test file grew 442 -> 796 lines.

```text
$ wc -l lib/abiss/scripts/nucleus_competition.py lib/abiss/scripts/nucleus_overlay.py
1441 lib/abiss/scripts/nucleus_competition.py      (was 747)
 351 lib/abiss/scripts/nucleus_overlay.py          (was 279)
```

Mechanism presence check (grep counts in nucleus_competition.py):

```text
internal_territory_id 7   emitted_id 5      plan_digest 20    NUC_MAX_UNITS 3
stage_report.partial 1    scan_geometry 5   map_to_watershed 5
separation_claim 2        zero_repairs 4    os.replace 2
```

Overlay: `emitted_id` 2, `completion` 3, `plan_digest` 6, `schema` 0 -- so the "schema 2.0 rejects
1.2" behaviour is enforced through the completion-marker and plan-digest checks, not a literal
schema string comparison.

```text
$ git rev-parse HEAD
c705458ae5b907bb9c32a75c63c85c6aad7edec7   (equals run_start_ref; no commits during the run)
```

## Attribution by mtime (git could not be used; see run.md Diff-Surface Correction)

`lib/abiss` is a nested git repository, `dev/` is gitignored (`.gitignore:159`), and
`tests/unit/test_abiss_nucleus_competition.py` and `connectomics/runtime/abiss_chunk.py` are
untracked in the parent repo. Only two of the declared files produce a git diff at all.

```text
code_v0 (2026-08-17 14:58-15:14):
  lib/abiss/scripts/nucleus_competition.py      15:08
  lib/abiss/scripts/nucleus_overlay.py          15:08
  connectomics/runtime/abiss_chunk.py           15:08   <-- NOT in plan_v3's file list
  dev/zebrafinch/nucleus_acceptance_report.py   15:08
  tests/unit/test_abiss_nucleus_competition.py  15:12
  dev/zebrafinch/sbatch_nuccomp_flood.sh        14:58   (new, declared)
  dev/zebrafinch/sbatch_nuccomp_scan.sh         14:5x   (new, UNDECLARED)
  dev/zebrafinch/sbatch_nuccomp_merge.sh        14:5x   (new, UNDECLARED)
  dev/zebrafinch/sbatch_nucleus_competition.sh  14:59
  dev/zebrafinch/submit_wholevol_sharded.sh     14:59
pre-existing in lib/abiss, 2026-08-16, from the earlier codex thread and NOT this run:
  CMakeLists.txt, scripts/composite_chunk_me.sh, scripts/cut_chunk_agg.py,
  scripts/cut_chunk_remap.py, src/agg/mean_aggl.cpp, src/seg/match_chunks.cpp,
  tests/test_nuc_agg.py, tests/test_nuc_match.py, build/
```

Codex touched only the two declared files inside `lib/abiss`. Had `git status` been trusted alone,
nine files from 2026-08-16 would have been misattributed to this run.

## The undeclared change, read in full

`connectomics/runtime/abiss_chunk.py:643,660`:

```python
chunkmap_input_cloudpath = _normalize_cloudpath(param.get("CHUNKMAP_INPUT", chunkmap_cloudpath))
...
    "CHUNKMAP_INPUT": chunkmap_input_cloudpath,
```

This is exactly the repair `plan_v3` §D.3 described -- default to `CHUNKMAP_OUTPUT` only when
unset so `param_overrides` wins. But §D.3 scoped it as a **written recommendation**, and §Scope
listed watershed reuse as out of scope. Correct code, outside the declared surface.

## Schema compatibility, checked against the real artifacts

```text
$ schema_version of existing production manifests
  wholevol_arm0_native96_nuc_matchguard : 1.2
  wholevol_arm096_nuc_competitive_v2    : 1.0
```

Codex states the schema-2.0 overlay intentionally rejects these, and that existing runs need a
fresh nuccomp publication before downstream agglomeration with this runtime. Both reference runs
that `plan_v3` gates 2 and 3 name as the regression oracle are pre-2.0.
