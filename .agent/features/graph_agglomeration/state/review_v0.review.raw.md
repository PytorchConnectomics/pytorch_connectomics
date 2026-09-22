# review_v0 raw — code review of code_v0 vs plan_v5

Reviewer: claude (planner, in-session), aided by an independent general-purpose
code-review subagent that read all four files in full plus plan_v5 and verified
several contracts empirically. The coordinator (also claude) independently
verified the highest-risk contracts by direct code read:
- oracle_stitch `--remap` additive: default path unchanged (main:200), `R` applied
  exactly once (main:220), oracle not double-remapped (score:181, comment:218),
  total-coverage hard-fail (TotalRemap.apply:143), background preserved (147-148).
- Firewall try_union (241-280): per-root marker SET, reject if `|combined|>1`,
  quarantine short-circuit, transitive A->unmarked->B unit-tested (295-297),
  post-run invariants (282-289).
- Scoring: raw has no ambiguity term (506-513); ambiguity uses best-other-partner
  `max(raw(a,x) for x!=b)` (542-556) with 1e-6 guard + clip (558); one-union-per-pair
  guard (587-589); decision enum incl. same_root_noop (600-623).

============================================================
INDEPENDENT SUBAGENT REVIEW (verbatim)
============================================================

Overall: The implementation is faithful to the spec. No blockers and no majors
that are actual code defects. The traps the spec calls out (cross-chunk label
collision, non-total remap, R applied twice/to the oracle, ambiguity leaking into
raw, second-highest vs best-other-partner, stale-root firewall checks, transitive
marker leak, footprint frame mismatch, off-by-one at faces, uint overflow,
default-path drift) are all handled correctly. Findings are minor/latent. Several
contracts verified empirically.

Contract-by-contract (each PASSES):
1. Total namespace/remap: enumerate_nodes/_labels_in_chunk (121-166) bincounts
   EVERY label per chunk (critical: the evaluator samples nodes that can land on
   any tiny segment); node=(chunk_key,label) so no cross-chunk collision;
   build_global_ids (652-672) compresses roots to 1..N uint32, distinct-root=>
   distinct-id asserted; background never enumerated (present[0]=False) and eval
   keeps 0->0; totality enforced by KeyError on uncovered sampled label
   (oracle_stitch:139-145). R applied exactly once, never to oracle output
   (216-223, score:180-188). uid = local + uid*OFFSET with uid=cz*169+cy*13+cx
   matches sample_variant (oracle_cc3d_chunked:183,207); index grid 6x13x13=1014,
   600-box (y,x<=9<13) safe; OFFSET=1<<21 vs max label 208880 checked.
2. Scoring non-circular: raw (506-513) weighted mean of iou/tangent/caliber/perp
   only; tangent floor 0.5 as separate gate (604); IoU floor exactly 0 (no >0 gate
   in 600-607); no caliber floor; tie-break (-raw,keys) exact (rank_key:377);
   ambiguity best-other-partner (542-558), 1e-6 guard, clip, defer >0.8;
   mutual-best via object-identity `is` on per-face min(rank_key) correct.
3. Firewall (263-280): rejects quarantined; combined=union; reject if >1; check on
   FRESH find() roots; quarantine flag per-node never mutated (safe: a >=2-marker
   node never unions so is always its own root); transitive blocked + self-tested
   (292-316); invariants asserted (282-289, 646).
4. One union attempt per pair: canonical_pair (88-89); attempted_pairs guard +
   final assert (587-589,647-648); same-root->same_root_noop (616-617); every
   candidate certified once (624).
5. Marker mapping: caff=(loc*CBSCALE)//POOL, POOL=[8,16,16], CBSCALE=[4,8,8],
   in-box caff<COARSE_SHAPE — byte-identical to soma_recon_wholevol.py:63-64
   (sample_markers:66-67); inversion points=caff*POOL+POOL//2 correct cell-center
   in anisotropic [20,10,10]nm grid (decode=g*CBSCALE confirmed: 1425*4=5700=HIz,
   630*16=10080=HIy/x); 0.95 gate enforced (224-235); source pinned, --min-ratio
   can't weaken it (269-276).

Empirically verified: faces/z0_y0_x0.h5 sz1/sz0/sx1 BYTE-IDENTICAL to
results/z0_y0_x0_decode_v1.h5 boundary slabs (IoU reads same labeling as areas/
enumeration); length_threshold=1000 (only in --remap path) yields IDENTICAL graph
as =0 for 600-cropped test-50 (50 skel, 470450 nodes) so default path unchanged;
default oracle_stitch rows=[cc3d,decode_v1] match historical JSON + L43 (decode_v1
oracle ~0.905); existing crossings/*.npz already carry caliber (schema change
additive); cross-chunk boundary planes adjacent non-overlapping cores (halo
cropped) so no off-by-one, in-plane matching ignores axis coord so 1-voxel z gap
irrelevant.

Minor/latent findings:
- M1: KD-tree radius anisotropic in z — radius_voxels=200/10=20 applied
  isotropically but z=20nm/voxel, so y/x-faces get ~400nm effective z tolerance
  (over-generates candidates in z on 4/6 face types; downstream gates filter; a
  z-adjacent smoke won't show it). Cue-permissiveness, not corruption.
- M2: representative-cell z-weighting (sample_markers:204) weights z by 80nm but a
  coarse cell is 160nm in z; only affects which nucleus cell is probed; inert
  (nuclei compact); new logic, not a spec item-5 violation.
- M3: _snap_to_foreground can allocate (2r+1)^3 up to r=1024 (sample_markers:
  154-166) -> OOM/hang for a genuinely isolated missed marker; gated behind 0.95,
  runs only for <5% missed (normally near fg, terminates early). Operational risk.
- M4: --remap pins length_threshold=1000 vs default 0 (oracle_stitch:202,206);
  currently inert (identical graph); latent cross-run divergence if test-50 ever
  gains a <1000nm skeleton. Recommend dropping to 0 or asserting min length.
- M5 (trivial): plan_v5 says "465 nonzero"; code correctly 464 nonzero + bg = 465
  total, asserted. Plan wording imprecise; no code action.

Modeling risk (by design, not a code defect): a non-overlapping IoU=0 pair with
collinear tangent + matched caliber/perp scores raw=(0+2+1+1)/5=0.8 > 0.6=tau ->
merges. Two distinct neurites passing within 200nm of a face with parallel
orientation can be linked; safeguards are tangent floor + mutual-best +
ambiguity-defer + firewall + the oracle-flat acceptance gate. A 2-chunk smoke
won't stress it; the 600-chunk run (thousands of internal faces) will. This is the
case the whole-volume Delta oracle <= eps gate exists to catch, wired correctly.
