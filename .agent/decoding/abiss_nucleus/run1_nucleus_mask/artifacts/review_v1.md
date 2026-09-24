# Review v1

## Summary

Reviewer: claude (planner, in-session). Raw notes: `state/review_v1.review.raw.md`.

**All six review_v0 findings are fixed, and I verified each one against the code rather than
against `code_v1.md`.** The headline check: I ran `work/test/run_v2_invariance.sh` cold, as an
outsider would, and it passed end to end —

```text
V2 build current: PASS
V2 baseline comparison: identical=35 differing=0 missing=0
V2 current-only empty sidecars: done_nuc.data nuc_cuts.data ongoing_nuc.data
V2 default-path nucleus log lines: 0
V2 stale nucleus environment without nuc.raw: PASS
run_v2_invariance: PASS
```

That single run confirms F2 (the gate is now reproducible by someone who did not write it), F1
(stale nucleus environment with no `nuc.raw` no longer aborts), and F4 (zero default-path nucleus
log lines, asserted rather than assumed). F3 is fixed at all three sites by gating on the
*transition* into CONFLICT (`!was_conflict && result == CONFLICT`). F6 is documented.

**F5 deserves specific credit:** investigating my open question surfaced a real bug rather than
just answering it. `reduce_nuc` kept boundary supervoxel ids while `reduce_counts`
(`reduce_chunk.cpp:137-139`) drops them, so `load_nuc` could legitimately meet a sid absent from
`seg_indices` and abort. The fix mirrors `reduce_counts` exactly and is correct.

Two majors below. **G1 is the scope item the user directed be folded in** from the `lib/em_seg`
read. **G2 is new, and it is the other half of F5's fix** — the half that was missed. Neither is a
regression; `V2` still passes and nothing from `code_v0` broke.

## Diff Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

Reviewed: `git diff 3c4f5621` — 13 tracked files, 511 insertions (vs 484 in `code_v0`), plus the
untracked `src/seg/NucExtractor.hpp` and `work/test/`. `HEAD` equals `run_start_ref`; no commits
created; the review itself made no edits.

## Findings

**G1 [major] — the mask-resolution contract makes the feature unusable on a realistically
produced mask.**
`scripts/cut_chunk_agg.py:24-27` hard-raises when the nucleus cutout's shape differs from
`seg.raw`'s, and there is no `NUC_RATIO` or `NUC_OFFSET` anywhere in the tree. But nucleus and
cell-body masks are normally produced at a lower mip, and `lib/em_seg` shows exactly that
production pattern for the same class of data:

```yaml
# em_seg/demo/r0.yaml:20-23   (source: .../jwr15/nucleus/cell_yl_cb_cc_fix.h5)
SOMA_EROSION : 0
SOMA_RATIO   : [4,16,16]
SOMA_OFFSET  : [14,0,0]
```

```python
# em_seg/em_seg/dataloader.py:46-51
zz = (z + self.ratio[0]/2) // self.ratio[0]   # EM z -> nearest low-res z
zz = int(zz + self.st[0])                     # plus offset
```

The mask is stored downsampled with a z-offset and the loader reconciles resolution on read
(in-plane zoom and erosion in the subclass — the demos import `scipy.ndimage.zoom` and
`binary_erosion`). Our contract rejects such a mask outright, and nothing documents that the caller
must pre-upsample and re-align first. Either add `NUC_RATIO`/`NUC_OFFSET` mirroring the `SOMA_*`
params, or state the pre-processing requirement explicitly in the README with the exact expected
alignment.

**G2 [major] — boundary nucleus identity is dropped in the OVERLAP=2 path and never restored.**
F5's fix is correct but is only half of the pattern it mirrors. Sizes dropped by `reduce_counts`
come back:

```text
match_chunks.cpp:329        writes extra_sv_counts.data
composite_chunk_me.sh:45    cat extra_sv_counts.data >> ongoing_supervoxel_counts.data
```

There is no nucleus analogue — `grep` for `extra_nuc` / `extra_*nuclei` finds nothing. So a
supervoxel on a chunk boundary retains its **size** across an overlap reduction but loses its
**nucleus record**. A soma straddling an overlap boundary therefore silently loses its cannot-link
constraint at that level, and in a whole-volume run boundaries are everywhere — which is precisely
where the constraint matters most.

`code_v1.md`'s evidence for this area is `V9 T1 boundary filter: nucleus/count sid sets remain
aligned`. That demonstrates the *absence of an abort*, not the *survival of identity*; they are
different claims and only the first is tested.

Stated honestly: I reasoned this from the file flow and the missing analogue and did **not** execute
a case proving loss. The test that settles it is named under Tests to Add. The fix that preserves
the existing symmetry is an `extra_nuc.data` written by `match_chunks` and appended in
`composite_chunk_me.sh` beside line 45.

**G3 [minor] — must-link is absent by design; record it as a limitation.**
Per the user's decision this stays out of the run, so it is not a defect against plan_v6, which
never promised it. But `lib/em_seg` shows what we are giving up: `seg_pipeline.py:366-380` snaps
every watershed fragment overlapping a soma onto a single reserved id **before the region graph
exists**, so the soma becomes one node with one aggregated affinity per neighbour. That is what
prevents soma fragmentation and makes the size veto irrelevant for cell bodies. Our feature only
*observes* identity — it never forces a merge — so `mean_aggl.cpp:709-718` can still block a real
soma from absorbing proximal dendrites. The README should say this plainly and name the follow-up.

## Tests to Add

1. **Boundary-spanning soma across an overlap reduction** (settles G2, highest value): two chunks,
   a tagged pair straddling the overlap boundary, run `reduce_chunk` → `match_chunks` → composite
   `agg`, and assert the pair is *still vetoed*. Today's test asserts only that sid sets stay
   aligned.
2. **Mask-resolution acceptance** (G1): if `NUC_RATIO`/`NUC_OFFSET` are added, a test that a
   `[4,16,16]`-downsampled mask with a z-offset is accepted and lands voxel-aligned with `seg.raw`;
   if instead documented, a test asserting the error message names the required pre-processing.

## Questions

1. G2 is the one thing I could not settle by reading alone. Is there any route by which a boundary
   supervoxel's nucleus record re-enters the ongoing payload at the next hierarchy level that I
   missed? If there is, G2 collapses to a documentation note.
2. For G1, which way do you want it — a real `NUC_RATIO`/`NUC_OFFSET` upsample path, or a documented
   caller requirement? The former is more code in a file we have already touched; the latter is free
   but pushes an easy-to-get-wrong alignment step onto the user.

## Verdict

VERDICT: NEEDS_CHANGES
