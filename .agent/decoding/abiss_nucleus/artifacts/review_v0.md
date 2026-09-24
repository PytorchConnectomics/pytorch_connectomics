# Review v0

## Summary

The implementation is faithful to plan v2, and its own tests prove the defect is real and fixed:
V3's adversarial fixture goes from `PROPER(1)` / `PROPER(2)` -> parent `CONFLICT` to `PROPER(1)`
in both children, equal to the monolithic result. V2 bit-invariance holds on both inert paths.
The reduction rule in `build_nucleus_table.py` matches `NucExtractor.hpp:66-77` exactly, including
the smallest-id tie-break and the `count=total=0` NONE algebra, and reads the same
`ABISS_NUC_MIN_TAGGED` / `ABISS_NUC_DOMINANCE` knobs. Run 1's far-edge clamp survived the move to
`nucleus_utils.py` with its logic intact -- that was the highest regression risk in the refactor
and I checked it line by line.

V4, which the code stage correctly reported NOT RUN, has now been run by the coordinator. **It
changes nothing about the reported contamination.** Per-nucleus dominance is identical to the
control to four decimals, and the same segment holds the same shares:

```
                control (w2ctl)                      table (nuctable)
275   0.8957   72198606672811349:0.101      0.8957   72198606672811349:0.101
319   0.9104   72198606672811349:0.084      0.9104   72198606672811349:0.084
373   0.8988   72198606672811349:0.097      0.8988   72198606672811349:0.097
CONFLICT segment  291,235,662 vox / 28,576,326 tagged   291,649,755 / 28,576,381
root segments     3,613                                  3,619
```

Plan v2's V4 criterion 1 (shared mask mass 110,244 -> <=27,500) therefore FAILS, and criteria 2-6
are met only trivially because nothing moved. That is a plan-level outcome, not a code defect: the
task owner directed correctness first, the code delivers the correctness fix, and the fix does not
happen to touch this symptom. The findings below are about the code.

The diagnostic that explains why is in the run's own counters: `load_conflict_collisions` is still
**4**, unchanged from the control. With a globally resolved table every *supervoxel* has one
identity, so a supervoxel-level collision is now impossible -- meaning the residual collisions are
at the *segment* level. A segment spanning two chunks accumulates a different subset of its member
supervoxels in each, so chunk A sees it as `PROPER(275)` and chunk B as `PROPER(319)`, and the
merge in each chunk is authorised against a partial segment record. A table keyed by supervoxel
cannot fix that by construction, and neither could any pre-pass, because segments do not exist
until agglomeration runs.

Extraction did change, which is the correctness fix showing up: `conflict_sv` 1 -> 66,
`subfloor_sv` 13,975 -> 5,577, `minority_sv` 11 -> 1.

## Diff Baseline

run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb

## Findings

* **[major] Every atomic worker loads the whole table into a hash map.**
  `NucExtractor::loadTable` fills `MapContainer<Tseg, nuc_record_t> m_table` with every record in
  the file, in each of the workers running concurrently on a node. The crop table is 139,294
  records / 4.0 MB, but the whole volume is unmeasured and 32 workers multiply it.
  `mean_aggl.cpp`'s `load_nuc` solves exactly this problem without a hash map: the file is already
  required to be strictly sorted, so a `std::vector<nuc_wire_t>` plus `std::lower_bound` gives the
  same fail-closed lookup at a fraction of the footprint. Measure the whole-volume table size
  before the whole-volume run either way.
* **[major] Stale-table staging is unenforced and fails obscurely.** The table is keyed by
  chunkmap-remapped supervoxel ids, so it must be rebuilt between `ws remap` and `agg` for every
  run. Nothing records which watershed it was built from; a table from a different run aborts every
  worker with `no record for supervoxel <id>`, which reads as data corruption rather than
  staleness. Add provenance -- the `WS_PATH`, chunk count and BBOX it was built from -- and check
  it on load, so the failure names its cause.
* **[minor] `Types.h`'s `nuc_record_t` invariant is now false for unreduced records.** The comment
  states `PROPER => count * ratio.den >= ratio.num * total`. On the table path a per-chunk PROPER
  record carries the local count of the *global* id against the local tagged total, so a chunk
  whose local voxels are mostly another nucleus emits a PROPER record that fails dominance -- and
  `count` may legitimately be 0. The reduced record still satisfies it. Nothing reads the invariant
  today (`nuc_join` and `nuc_can_merge` do not check), so this is documentation, but a contract
  comment that is quietly false is how the next change breaks.
* **[minor] The run-1 clamp comment was rewritten and its worked example lost.** The logic in
  `nucleus_utils.py:52-63` is intact, but the explanation dropped from "mip0 z=5698 with ratio 4
  maps to (5698+2)//4 = 1425, one past a 1425-long axis ... is what the whole-volume run needs" to
  a single line, and the inline `# tolerate at most one source voxel of rounding` is gone. That
  comment exists because the missing clamp killed 160 of 240 whole-volume tasks. Restore it; a
  moved function should move unchanged.
* **[minor] `minority_sv` means something different on each path** (`tagged > local_count` with the
  table, `tagged > max_count` without), so the counter is not comparable across a config change.
  These counters are the only in-run visibility this feature has; give the table path its own name.
* **[minor] Cosmetic churn in the moved helpers** (`'F'` -> `"F"`). Harmless, but it makes a pure
  move read as an edit.

## Tests to Add

* A whole-volume-scale table-size measurement, as the input to the memory decision above.
* A test that a table built against a *different* watershed is rejected with a staleness message
  rather than aborting on the first missing supervoxel.
* V4 as run here should be folded into the harness as a scripted comparison, since it is now the
  standing regression check for the feature's effect on real data.

## Questions

1. Given V4, is the segment-level partition dependence (`load_conflict_collisions` = 4) worth
   attacking, or does the frozen-boundary mechanism relaxed by `#ifdef EXTRA` need looking at
   first? The two are the same leak seen from different ends.
2. Should `NUC_TABLE` become a required stage in the batch scripts rather than an optional param,
   now that using it correctly depends on ordering the pipeline around it?

## Verdict

VERDICT: NEEDS_CHANGES
