# Review v0

## Summary

Code review of code_v0 against the approved plan_v5, by claude (planner,
in-session) with an independent general-purpose reviewer that read all four files
in full and verified several contracts **empirically**. Attested from
`state/review_v0.review.raw.md`.

**The implementation is faithful to plan_v5 with no blockers and no majors that
are actual code defects.** Every spec contract PASSES: total collision-free
namespace enumerating *every* label (not just big branches), remap applied exactly
once and never to the oracle output, background preserved, non-circular raw score,
best-other-partner ambiguity defer, marker-set firewall with quarantine and the
transitive `A→unmarked→B` rejection, one union attempt per undirected pair, and the
marker mapping byte-identical to `soma_recon_wholevol.py`. Codex's synchronous
controls (600/600 symlinks, marker gate 0.9954, firewall unit tests, zero-union
namespace 10,237 distinct ids, no-edge control exact) and the 2-chunk smoke
(**base 0.332→0.377 = +0.045 at oracle flat to 8 decimals**, tau 0.4–0.7; tau 0.8
no-op) corroborate correctness and merge-safety on real data.

**Empirical verifications by the reviewer:** `faces/*.h5` boundary slabs are
byte-identical to the `results/*_decode_v1.h5` boundaries (so the IoU cue reads the
same labeling as areas/enumeration); the uid encoding `local + uid*OFFSET`,
`uid=cz*169+cy*13+cx` matches `sample_variant` and the 6×13×13 index grid;
`length_threshold=1000` in the `--remap` path yields an identical graph to `=0` on
the current test-50 (so the default oracle path is unchanged); existing
`crossings/*.npz` already carry `caliber` (the extractor change is genuinely
additive). No git-tracked file changed (`dev/zebrafinch/` is gitignored; HEAD and
the 24 baseline dirty files unchanged).

## Diff Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b

Review surface: the on-disk `dev/zebrafinch/` files (gitignored, so not in
`git diff`) — `graph_link_whole.py` (new, core), `sample_markers.py` (new),
`fix_decode_v1_symlinks.py` (new), `graph_link_whole.README.md` (new), and the
additive changes to `big_branch_extract.py` and `oracle_stitch_decode_v1.py`.

## Findings

All minor / latent; none block approval.

- [minor] **M1 — KD-tree search radius anisotropic in z** (`graph_link_whole.py`
  ~462). `radius_voxels = 200nm/10 = 20` is applied isotropically, but z is
  20 nm/voxel, so on y/x-faces the effective z tolerance is ~400 nm. Over-generates
  candidates in z on 4/6 face types; downstream gates + the oracle-flat acceptance
  filter them. Cue-permissiveness, not corruption. Fix later: scale the z tolerance
  by RES per axis.
- [minor] **M3 — `_snap_to_foreground` unbounded worst case** (`sample_markers.py`
  ~154-166). The expanding-cube search can allocate `(2r+1)³` up to r=1024 for a
  genuinely isolated missed marker → possible OOM/hang. Gated behind the 0.95 gate,
  runs only for <5% missed markers (normally near fg; both actual misses resolved
  nearby). Operational risk only; add a hard radius cap.
- [minor] **M4 — `--remap` pins `length_threshold=1000`, default pins `0`**
  (`oracle_stitch_decode_v1.py` ~202/206). Currently inert (verified identical
  graph); latent cross-run divergence if test-50 ever gains a <1000 nm skeleton.
  Recommend dropping to `0` or asserting a minimum skeleton length.
- [minor] **M2 — representative-cell z-weighting** (`sample_markers.py` ~204):
  weights z by 80 nm vs the 160 nm coarse z-cell; only selects which nucleus cell
  is probed; inert (nuclei compact). New logic, not a spec violation.
- [minor] **M5 (trivial) — doc wording:** plan_v5 says "465 non-zero distinct IDs";
  code correctly uses 464 nonzero + background = 465 total, asserted. No code
  action.

**Modeling risk (by design, not a code defect):** a non-overlapping (IoU=0) pair
with collinear tangent + matched caliber/perp scores `raw=(0+2·1+1+1)/5=0.8 >
0.6=tau_score` and will merge; two distinct neurites passing within 200 nm of a
face with parallel orientation could be linked. The only safeguards are the
tangent floor + mutual-best + ambiguity-defer + firewall + the whole-volume
oracle-flat gate — exactly as plan_v5 states. The 2-chunk smoke cannot stress this;
the 600-chunk run (thousands of internal faces) will, and the `Δoracle ≤ eps` gate
+ the bounded `tau_score` sweep are wired to catch and price it.

## Tests to Add

- Whole-volume acceptance (the decisive test; pending — see Questions): the
  `tau_score ∈ {0.4..0.8}` sweep base+oracle vs the decode_v1 baseline; select the
  smallest `tau_score` with `Δoracle ≤ eps`.
- A y/x-adjacent 2-chunk smoke would exercise M1 (z-anisotropic radius).
- A hard cap unit test for `_snap_to_foreground` radius (M3).

## Questions

- **Whole-volume acceptance is still pending.** Codex's launched run died with its
  sandbox (25/600 crossings); the coordinator relaunched it parallel and
  harness-tracked (`bymr2tlee`). The code is approved; the empirical Step-1
  result (whole-volume base↑, oracle-flat, tau sweep) will be appended when that
  job completes. If the whole-volume oracle drops at every `tau_score`, that is the
  modeling risk above realizing — a parameter outcome, not a code defect.

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS
