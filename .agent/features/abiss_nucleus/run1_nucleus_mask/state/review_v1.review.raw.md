# review_v1 — raw planner notes (in-session)

Focus per protocol: were review_v0's findings fixed, and did the fixes introduce anything new.
Plus one finding the user directed be folded in from the lib/em_seg read.

## F1-F6 verification (independent, not from code_v1.md)

F1 FIXED. atomic_chunk_ME.cpp:84-108 — nuc_ratio/nuc_min_tagged declared with defaults, then
   assigned at 102-103 INSIDE `if (std::filesystem::exists("nuc.raw"))` at line 86.
F2 FIXED. work/test/run_v2_invariance.sh exists and I ran it cold myself:
     V2 build current: PASS
     V2 baseline comparison: identical=35 differing=0 missing=0
     V2 current-only empty sidecars: done_nuc.data nuc_cuts.data ongoing_nuc.data
     V2 default-path nucleus log lines: 0
     V2 stale nucleus environment without nuc.raw: PASS
     run_v2_invariance: PASS
   (absl configure warning is the pre-existing benign one.) This is now reproducible by
   someone who did not write it, which was the entire point of F2.
F3 FIXED. All three sites now gate on the TRANSITION: `!was_conflict && result == CONFLICT`
   (mean_aggl.cpp:302, reduce_chunk.cpp:217, match_chunks.cpp:427).
F4 FIXED and asserted — the harness checks "default-path nucleus log lines: 0".
F5 FIXED, and the investigation found a real bug I only suspected. reduce_nuc now skips
   boundary_sv, mirroring reduce_counts (reduce_chunk.cpp:137-139) exactly. Previously
   reduce_nuc kept boundary sids while reduce_counts dropped them, so load_nuc could see a sid
   absent from seg_indices and abort. Correct fix.
F6 FIXED as documentation — README carries the mixed-binary upgrade prohibition.

## New findings

G1 [major] mask-resolution gap (user-directed fold-in from lib/em_seg).
   cut_chunk_agg.py:24-27 hard-raises when the nucleus cutout shape != seg.raw's, and there is
   no NUC_RATIO/NUC_OFFSET anywhere (grep: none). em_seg shows the production pattern:
     demo/r0.yaml:20-23  SOMA + SOMA_EROSION:0 + SOMA_RATIO:[4,16,16] + SOMA_OFFSET:[14,0,0]
     em_seg/dataloader.py:46-51  zz = (z + ratio[0]/2)//ratio[0]; zz = int(zz + st[0])
   i.e. the soma/nucleus mask is STORED DOWNSAMPLED with a z-offset and the loader reconciles
   resolution on read (in-plane zoom + erosion in the subclass; the demos import
   scipy.ndimage.zoom and binary_erosion). Nucleus/cell-body masks are normally produced at a
   lower mip. Our NUC_PATH rejects exactly that. Feature is unusable on a realistically
   produced mask without an external pre-upsample step that nothing documents.

G2 [major] boundary nucleus identity dropped in OVERLAP=2 and never restored.
   F5's fix is right, but it is only half the pattern. Counts dropped by reduce_counts come
   BACK: match_chunks.cpp:329 writes extra_sv_counts.data and composite_chunk_me.sh:45 does
   `cat extra_sv_counts.data >> ongoing_supervoxel_counts.data`. There is no nucleus analogue
   (grep extra_nuc / extra_*nuclei: none). So a supervoxel on a chunk boundary keeps its size
   across an overlap reduction but LOSES its nucleus record. A soma straddling an overlap
   boundary silently loses its cannot-link constraint at that level — and in a whole-volume run
   boundaries are everywhere.
   code_v1's evidence is "V9 T1 boundary filter: nucleus/count sid sets remain aligned", which
   demonstrates absence of abort, NOT survival of identity. Those are different claims.
   I reasoned this from the file flow and the missing analogue; I did not execute a case that
   proves loss. The test that would settle it: two chunks, a tagged pair straddling the overlap
   boundary, assert still vetoed after reduce_chunk + match_chunks + composite agg.
   Fix that preserves existing symmetry: write extra_nuc.data in match_chunks, append it in
   composite_chunk_me.sh next to line 45.

G3 [minor] must-link absent by design — user decided to keep it out of this run.
   em_seg snaps every watershed fragment overlapping a soma to ONE reserved id before the
   region graph exists (seg_pipeline.py:366-380), which is what prevents soma fragmentation and
   makes the size veto moot for cell bodies. Ours only observes identity; it never forces a
   merge, so mean_aggl.cpp:709-718 can still block a soma from absorbing proximal dendrites.
   Not a defect against plan_v6 — plan_v6 never promised it. Belongs in the README as a stated
   limitation with the follow-up named.

Nothing regressed. V2 still 35/0/0. Diff is 13 files, 511 insertions vs 484 in code_v0.
