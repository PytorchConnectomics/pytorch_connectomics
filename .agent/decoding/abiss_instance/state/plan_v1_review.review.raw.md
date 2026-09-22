## Summary

Plan v1 is substantially better, but it is not ready for implementation as the complete CCC task.

The instrumentation-first sequencing is correct. Treating instrumentation as the entire increment is not: under the coordinator’s adopted reading that the CCC task authorizes implementation, this plan produces no efficiency improvement despite “more efficient” being a central requirement. It needs either explicit user approval to narrow the task to a measurement milestone or a measurement checkpoint followed by a concrete, selected optimization within this run.

Of the previous findings, #2, #3, #5, and #12 are genuinely resolved. #8 is substantially addressed. #1, #4, #6, #7, #9, #10, and #11 remain partial or unresolved.

## Findings

1. [major] Prior #1 is only partially resolved. Preallocating temporary IDs for every territory removes the “scan cannot know the largest territory” contradiction. However, assigning the parent ID through an overlay-time mapping introduces a new artifact representation and requires a canonical translation contract in `nucleus_overlay.py`. Section B calls this fully designed, while the file is excluded and no test requires final sharded output to equal the serial output voxel-for-voxel.

2. [major] Prior #2 is resolved. The fixed-capacity array is statically submittable, represents zero units, and fails closed above capacity.

3. [major] Prior #3 is resolved in principle. The gate now checks territory artifacts as well as the manifest, preserves serialization, and compares each run only with its own parameters. The rerun must use a separate output namespace so it cannot overwrite the reference artifacts it needs as its oracle.

4. [major] Prior #4 remains unresolved. “Removes or refuses to overwrite” is not one policy: refusing leaves the stale manifest visible after failure. The plan must require run-scoped temporary output or remove/quarantine the old publication before execution, then publish only after validated completion. Path and size are also insufficient input fingerprints; the watershed manifest, nucleus input, affinity/cost input, configuration, and relevant code version need content or authoritative artifact identities. `stage_report.json` needs atomic publication and defined failure semantics too.

5. [major] Prior #5 is genuinely resolved. The B2 test now has an independent oracle: expected competitive labels must survive the real remap path and distinct owners must remain distinct. Prefer asserting the complete expected output label array, but the described oracle is materially adequate.

6. [major] Prior #6 is resolved only as milestone sequencing. The numeric threshold makes the decision point explicit, but the scan-dominant branch still says “design scan sharding” rather than describing an executable implementation. The exact 40%/60% boundary can also select both branches. More importantly, deferring every optimization means this CCC run cannot satisfy the efficiency portion of the task.

7. [major] Prior #7 is not fully resolved. Section D promises R1–R7, reuse, and a ship verdict, but omits §7 deliverable 2 as a written deliverable: an ordered efficiency plan with expected speedup and determinism risk for each proposed change. The flood branch has an Amdahl estimate; the scan branch has neither a design nor an estimate. The watershed-reuse deliverable must also explicitly answer how `CHUNKMAP_INPUT` precedence should be repaired, not merely promise a “recommendation.”

8. [major] Prior #8 is substantially addressed but remains underspecified. Outside-repair-box equality and rejected-edge counts are now gates, which corrects the blanket deferral. The plan still omits the “already-clean nuclei unchanged” half of acceptance item 4 and does not identify how reference rejected-edge counts will be established. These checks need named inputs, comparison outputs, and failure criteria.

9. [major] Prior #9 is only cosmetically addressed. Saying the document will contain “a position” on R3, R5, and R6 does not plan the work needed to answer them. The plan should require:

   - R3: inspect manifest evidence and emit `local_only` unless global containment is demonstrable.
   - R5: trace ordering inputs and test invariance across ordering/shard counts.
   - R6: test or document whether local scoring excludes cross-chunk merges; otherwise retain the null as an unresolved ship blocker.
   - R4: qualify every cross-arm conclusion because the configurations are non-equivalent.

10. [major] Prior #10 remains incomplete. Reachability is stated for seeds, coordinates, overlaps, and collisions, but not for an incompatible watershed manifest. The plan also promises tests for conditions it calls unreachable without identifying the validation seam those tests will drive. It must audit whether each abort check actually exists and add production validation where necessary, rather than adding tests alone.

11. [major] Prior #11 remains partial. The instrumented baseline and critical-path metric are correct, but there is no efficiency acceptance criterion for this increment because it contains no optimization. The scan branch lacks expected speedup and determinism risk, and the flood estimate is deferred until after implementation planning. Thus the plan measures efficiency without improving it.

12. [minor] Prior #12 is resolved for the serial increment. `peak_rss_kib_process` and `topology: "serial"` give the measurement an unambiguous meaning.

13. [major] R2’s launcher behavior is not executable as written. `submit_wholevol_sharded.sh` runs before the result exists and cannot surface a future zero-repair outcome without a dependent checker. The stage should emit the warning and durable flag; an acceptance/reporting step should decide whether zero repairs is acceptable for the intended experiment.

14. [minor] “Committed as” conflicts with verification requiring no commits during the run. The plan should say the review is “written” or “produced” in the run folder.

## Questions

**Q-1:** Keep `docs/nucleus_competition_review.md` in the run folder as the reviewed source of truth. Do not make `dev/zebrafinch/lesson_nucleus_competition.md` the sole copy, particularly because `dev/` has no recoverable git history. Accepted operational conclusions can later be summarized or linked from the lesson file.

**Q-2:** Yes, rerunning `nuccomp` alone against an existing completed watershed is the appropriate baseline. The watershed is nucleus-blind, and rebuilding it would add cost and unrelated variance. The rerun must use frozen, fingerprinted inputs and parameters, write to a fresh output namespace, preserve the reference artifacts read-only, and record stage critical-path elapsed time. Both reference parameterizations should receive their own baseline.

READY: no