# review_v0 — raw planner notes (in-session; planner == claude == this session)

Method: read the tracked diff vs 312bf54 file by file, then independently ran the checks that
decide the verdict rather than relying on code_v0.md. The coder's own transcript is preserved at
`state/code_v0.stdout.log` / `state/code_v0.last_message.md`.

## Independent reproduction (not taken from the coder's report)

**V4 — run by me, reported NOT RUN by the coder (correctly; it needs SLURM).**
New arm `dev/zebrafinch/nuc_z1_y7_x6/nuctable` with `NUC_TABLE` set, driven by a new
`run_arm_table.sh` that builds the table BETWEEN `ws remap` and `agg` because it is keyed by
chunkmap-remapped ids. SLURM 2819570, COMPLETED 00:37:15.

    STAGE ws 01:28:43 | ws remap 01:36:54 | table 01:41:00 | agg 01:50:40 | agg remap 02:00:42 | DONE 02:05:54

`nuc_arm_summary.py w2ctl nuctable`:

    w2ctl     3,613 segs   NONE 3412 PROPER 200 CONFLICT 1
              72198606672811349  291,235,662 vox  CONFLICT tagged=28,576,326
    nuctable  3,619 segs   NONE 3418 PROPER 200 CONFLICT 1
              72198606672811349  291,649,755 vox  CONFLICT tagged=28,576,381

`nucleus_shell_contamination.py --tol 0.0` on nuctable:

    275  0.8957   72198469032353291:0.896  72198606672811349:0.101
    319  0.9104   72198400447162578:0.910  72198606672811349:0.084
    373  0.8988   72268975416894476:0.899  72198606672811349:0.097
    286/293/325/337/377 unchanged at 1.000 / 0.9998 / 1.000 / 1.000 / 1.000

Identical to the control to four decimals. Plan v2 V4 criterion 1 FAILS.

Run counters (summed over the crop), table vs control:

    conflict_sv               66   vs      1
    subfloor_sv            5,577   vs 13,975
    minority_sv                1   vs     11
    load_conflict_collisions   4   vs      4      <-- UNCHANGED

The unchanged collision count is the diagnostic in the review's Summary: supervoxel-level
collisions are impossible under a global table, so the residual four are segment-level.

**Refactor regression check.** Extracted the three moved helpers from `git show
312bf54:scripts/cut_chunk_agg.py` and diffed them against `scripts/nucleus_utils.py`:

    nucleus_axis_vector     IDENTICAL
    validate_nucleus_cutout CHANGED — quote style only ('F' -> "F")
    cut_nucleus_data        CHANGED — comments rewritten, logic identical;
                            the run-1 clamp (limit/slack/numpy.clip) is intact at lines 52-63

**Reduction-rule equivalence.** Read `reduce_histograms` in `build_nucleus_table.py` against
`NucExtractor.hpp:66-77`: `tagged < min_tagged` -> NONE with count=total=0; `max_count*den >=
num*tagged` -> PROPER with smallest-id tie-break; else CONFLICT. Same env knobs
(`ABISS_NUC_MIN_TAGGED`, `ABISS_NUC_DOMINANCE`). Equivalent.

**C++ read.** `NucExtractor` table path: `m_use_table` requires both a mask and a table path, so a
stale `NUC_TABLE` alone is inert (matches V2). Table validation covers state range,
PROPER-implies-id, non-PROPER-implies-no-id, strict sort, truncation (`gcount() != 0` after the
read loop is correct), and fails closed on a missing sid. PROPER emits the local count of the
global id against the local tagged total — which is what makes the `Types.h` invariant comment
false per-chunk.

**Working tree.** `git -C work2/abiss rev-parse HEAD` = 312bf54…, matching run_start_ref; no
commits created; modified: `scripts/{cut_chunk_agg,set_env}.py`, `src/seg/{NucExtractor.hpp,
atomic_chunk_ME.cpp}`; new untracked: `scripts/{build_nucleus_table,nucleus_utils}.py`, `work/`.

## Verdict reasoning

The code does what plan v2 specified and V3 proves the defect is fixed. V4 shows the fix does not
touch the reported symptom; that is a plan outcome, not a code defect, and the task owner asked for
correctness first. NEEDS_CHANGES is for the two majors — per-worker table memory and unenforced
staleness — both of which bite at whole-volume scale rather than on the crop.
