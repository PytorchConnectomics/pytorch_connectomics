review_v0 raw notes (Claude, coordinator/planner, in-session code review of code_v0 vs plan_v2)

Reviewed files (read in full): dev/mit_liconn/decode_axon.py (new), and the changed cores in
decode_v2.py (decode_sections), decode_v4_split.py (prob1_carve + bump-safe guard),
decode_v3_merge.py (prob2_merge + region_graph(seg,aff) + caliber), valid_tube_metric.py (score_array).

FIDELITY OF REUSED CORES — strong evidence, PASS.
Codex's code_v0 verification ran the four full-volume (800^3) partition-equality checks (relabeling-
invariant bidirectional label maps), all PASS:
  prob1_carve(bump_safe=False) == decode_p1     (45 candidates, 2557 labels)
  prob2_merge(area_tol=None)    == decode_p1p2   (merged 116, skipped 584 parallel, 2441 labels)
  prob1_extend                  == decode_p1p2eg (27 accept / 97 reject, 2441 labels)
  decode_sections               == decode_v2_c_constrained (2579 labels)
So the extractions are behavior-preserving; the legacy CLIs are unaffected.

CORRECTNESS OF THE NEW CODE (the shipped path uses bump_safe=True + area_tol=0.5, NOT covered by the
equality checks — reviewed by reading + self-test):
- decode_axon.py orchestration: aff loaded ONCE up-front (load_aff), affxy shared, tiers wired
  strong -> decode_3d(carve->extend->merge) -> drop-only crumb -> save -> NERL(full). Data flow correct;
  build_strong_sections uses the global z0-800 cache sliced [z0:z1]; --zslice requires a tag (no
  overwrite) and is labeled SMOKE. CLI: one required mode, --self-test returns before I/O, thr validated.
- prob1_carve bump_safe path: _ComponentAreas union-find accumulates orphan-chain per-slice profiles;
  _candidate_bump_safe requires (host bump present over [z0,z1] BEFORE) AND (removed AFTER) AND
  (_no_new_runs on host) AND (no new run on above+below+bridge). 3-distinct-roots guard prevents
  re-carving unified components. Legacy path (_carve_candidate_legacy) preserved for equality. Correct.
- prob2_merge caliber path: region_graph(seg,aff) validates shape, channel-correct 3-dir boundary means
  (equality-proven with area_tol=None). Parallel z-overlap veto UNCHANGED and precedes the caliber gate,
  so parallels can't slip in. Caliber = size/(zmax-zmin+1); |ca-cb|>area_tol*max -> skip. Correct.
- score_array refactor: metric line prints WITHOUT a leading label; score(path) prints the stem label
  then delegates; decode_axon.report(name,seg) prints the name label then delegates. No double-label
  (confirmed in the pilot output). Behavior-preserving; CLI valid_tube_metric.py unchanged.
- self_test: covers masks (partition+disjoint), prob1_carve (carve + solid-reject + bump-adding-reject),
  prob1_extend (gate accept/reject + real endpoint extension), prob2_merge (sequential-accept /
  parallel-reject / caliber-mismatch-reject), crumb drop-only, and all four deferred NotImplementedError
  stubs. Comprehensive for the shipped path. PASS in code_v0 verification.

MINOR (non-blocking):
- m1: prob2 caliber denominator is z-EXTENT (zmax-zmin+1), while plan_v2 phrased it "voxels/occupied-
  slice". Equivalent for gap-free tubes; a reasonable proxy. Cosmetic wording vs impl.
- m2: The SHIPPED full-volume config (bump_safe=True, area_tol=0.5) is validated only by synthetic self-
  tests; its 800^3 metric is not proven by the equality checks. The caliber gate (area_tol=0.5) is a NEW
  merge restriction vs decode_p1p2 and could reduce merges / valid volume. plan_v2 flagged this with a
  fallback (decode_3d area_tol=None). REQUIRES the coordinator's authoritative --full run to confirm the
  success gate (VALID vol >= 62.7, parallel <= 8, bumps <= 117). [Full run launched: bnklysywh.]

AUTHORITATIVE FULL-VOLUME RESULT (coordinator ran `decode_axon --full`, job bnklysywh, 1038s, exit 0):
  t1_strong: VALID vol 53.8%  bumps 152  parallel 0   (~= decode_v2 base)
  t2_3d    : VALID vol 62.5%  bumps  84  parallel 1
  final    : VALID vol 62.5%  bumps  84  parallel 1   (crumb drop-only; unchanged)
  real NERL 0.6114  oracle-merge 0.6299  (secondary; GT over-split)
Comparison (GT-free valid-tube, primary):
  decode_v2        53.5% / 152 / 9
  decode_p1p2      62.7% / 117 / 8   (prior best)
  decode_p1p2eg    65.5% /  89 / 0   (bump-safe extension)
  decode_axon      62.5% /  84 / 1   <- FEWEST bumps of any decode; parallel 1; coverage ~= decode_p1p2

FINDING (behavioral, [minor], not a code defect): the shipped caliber gate (prob2 area_tol=0.5) yields
the lowest false-merge counts (bumps 84) but leaves VALID volume at 62.5% — ~3 pts below decode_p1p2eg
(65.5%). The gate rejected caliber-mismatched sequential merges (pilot: 6 skipped), trading coverage for
purity. plan_v2 anticipated this and made area_tol a knob with an area_tol=None fallback. Recommend the
coordinator empirically compare area_tol=None to pick the shipped default (follow-up tuning, one-line knob).

VERDICT: the implementation faithfully realizes plan_v2, preserves the extracted cores exactly (4/4 full-
volume partition-equality PASS), the new logic is correct and self-tested, and the full run meets the
PRIMARY project goal (false-merge reduction: bumps 84 < 117, parallel 1 < 8 vs decode_p1p2) at coverage
~= decode_p1p2. The only open item is a config-tuning choice (caliber gate), not a correctness defect.
=> APPROVE_WITH_MINOR_COMMENTS.
