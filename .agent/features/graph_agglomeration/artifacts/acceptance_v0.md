# Step 1 whole-volume acceptance (code_v0)

Empirical acceptance for code_v0, per plan_v5 (whole-volume `test_50_skeletons`
NERL, `oracle=branch_merge`, `eps=1e-4`). Run: coordinator relaunch `bymr2tlee`,
a002, `--workers 16`, 2026-07-10 00:46 → 05:19; 600/600 crossings; 1580 internal
faces; 18,427 candidates; namespace over 2,341,137 foreground nodes.

## Result table (600-box test-50)

| variant | accepted | base | oracle | Δbase | Δoracle |
|---|---:|---:|---:|---:|---:|
| decode_v1 baseline | — | 0.0049 | **0.9055** | — | — |
| linked zero (no-edge control) | 0 | 0.0049 | 0.9055 | +0.0000 | +0.0000 |
| tau 0.4 | 14209 | 0.0087 | 0.8894 | +0.0038 | **−0.0161** |
| tau 0.5 | 14101 | 0.0087 | 0.8908 | +0.0038 | **−0.0147** |
| tau 0.6 | 13816 | 0.0086 | 0.8908 | +0.0037 | **−0.0147** |
| tau 0.7 | 12688 | 0.0082 | 0.8908 | +0.0033 | **−0.0147** |
| tau 0.8 | 35 | 0.0049 | 0.9055 | +0.0000 | +0.0000 |

miss = 0.0315 for all (linker does not touch coverage). cc3d reference:
base 0.0081 / oracle 0.8362 / miss 0.0586.

## Verdict: FAILS the merge-safety gate (honest negative)

- **Oracle drops ~0.015 at every tau that accepts links** (`Δoracle < −eps`) ⇒ the
  linker introduces cross-neuron false merges at whole-volume scale. The only
  oracle-flat point (tau 0.8) is a no-op (35 accepts, base unchanged). The bounded
  `tau_score` sweep found **no** operating point with a base gain at flat oracle.
- **Realized base gain is negligible** (~0.005 → ~0.009, both ≈ 0). The linker does
  **not** realize the ~47% tiling lever — it links adjacent-face pairs, but whole
  neurons cross many faces (L44 chain-multiplication), so realized NERL stays ~0.
- Net: negligible gain **and** an oracle drop = a net-negative operation under the
  length²-weighted metric (a false merge costs ~2×). Do not ship this remap.

## Why this happened (matches the lessons, and the review's flagged risk)

1. **Firewall worked as designed** — 33 `firewall_conflict` edges rejected, 20
   pre-existing multi-marker segments quarantined. Zero-union control reproduced
   the baseline exactly; namespace structural asserts PASS. So the *nucleus*
   constraint held.
2. **The false merges are markerless neurite–neurite links** the firewall cannot
   see (no marker on either side) but the sparse test-50 oracle *does* detect
   (oracle drop). This is precisely the by-design modeling risk `review_v0`
   flagged: a low/zero-overlap pair with collinear tangent + matched caliber
   scores `raw≈0.8 > tau` and merges; tangent floor + mutual-best + ambiguity-defer
   were insufficient to stop it at scale.
3. **Confirms L44**: per-face / adjacent-face linking cannot realize the tiling
   lever and introduces merges. The fix is **structural**, not a threshold: whole
   **trajectory / skeleton-endpoint path matching** across many faces, or
   **halo-overlap volume matching** / bigger effective tiles (fewer faces per
   neuron), so a neuron is reconstructed as one path rather than a chain of
   independently-risky face links.

## Status of the code (unchanged)

The code is correct and approved (`review_v0` APPROVE_WITH_MINOR_COMMENTS); the
acceptance harness, firewall, namespace, and controls all work. This is an
**algorithm / operating-point** negative result, not a code defect — plan_v5's
honesty clause ("if even tau_score=0.8 drops the oracle, the cue is unsafe and
Step 1 fails honestly — report, don't ship") fired exactly as intended.

## Recommended next step (Step 2 design, out of scope for this run)

Replace adjacent-face endpoint linking with **trajectory-consistent multi-face
path matching** (or halo-overlap matching): require a candidate to extend a
skeleton path with consistent tangent across ≥2 faces, and/or gate on r10
long-range agreement, so isolated parallel-neurite face contacts are rejected.
Re-run the same whole-volume acceptance; success = base rises toward the 0.9055
ceiling with `Δoracle ≤ eps`. Also apply review_v0 M1 (per-axis search radius) —
the z-anisotropic 200 nm→400 nm over-generation likely contributed to the
markerless merges on y/x faces.

Artifacts: `dev/zebrafinch/results/graph_link_remap_tau0{4..8}.npz`,
`graph_link_certs_tau0{4..8}.csv` (every candidate + decision for forensic audit of
which links dropped the oracle), `oracle_stitch_decode_v1.json`,
`graph_link_whole_run.log`.
