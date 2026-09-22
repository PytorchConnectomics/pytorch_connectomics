# MESA-EM for NISB base_banis+

## Decision and scope

MESA-EM is a signed medial edge graph whose connected components carry instance identity. The
existing merge-safe banis+ affinities remain the body anchor; new image-conditioned relational
heads repair thin false splits; a sparse object-context validator decides ambiguous links; body
growth and fill may realize an identity but may never create or union identities.

This workspace implements two CPU/no-retrain gates before any training:

- **G0** is the human-ratified proven-ancestor oracle: 1-voxel GT skeleton nodes, GT-derived
  same-edge channels, identity-blind union-find, and watershed on GT-SDT. It proves the
  representation/decode plumbing can carry connected identities. It does **not** prove that a
  network can learn a medial band or signed edge logits.
- **G1** is a banis+ feasibility surrogate: proposal recall, geometry-only fragment-degree
  matching, a GT-confirmed identity-gain upper bound, and fragment-graph realization. It is not
  Phase-1 certification because there are no trained MESA-EM heads or validators yet.

Training, GPU execution, predicted-SDT growth, production fill, a second chunk, and full-volume
decode are outside these gates.

## Bound assets and conventions

All spatial arrays use `(X,Y,Z)` with physical sampling `[9,9,20]` nm. No transpose is applied.

| Concept | Bound implementation or asset |
|---|---|
| banis+ affinity | `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/raw_x1_ch0-1-2.h5`, `main[:,1000:2000,1000:2000,450:900]` |
| affinity decode | `decode_affinity_cc(aff, 0.66, backend="numba", edge_offset=0)`; channel `c` is the `+1` face on array axis `c` |
| GT | `dev/nisb/data/center_chunk/seg.h5`, `(1000,1000,450)` |
| local skeleton | `dev/nisb/data/center_chunk/seg.erlgraph.npz`; 626 skeletons, physical-nm nodes and edge lengths |
| coordinate transform | `load_nerl_graph(..., resolution=[9,9,20])`, then `get_nodes_position([9,9,20])` |
| local NERL | `compute_nerl_score_details(..., resolution=[9,9,20], merge_threshold=1, num_workers=16)` |
| whole-volume sanity | cached node LUTs scored with `per_chunk_nerl.load_cache`, `import_em_erl`, and `compute_erl_score`; never decode the 12.15-billion-voxel volume |
| G0 ancestor | `dev/nisb/scripts/centerline_graph_decode_gt.py` |
| GT-SDT | `make_gt_sdt_region.py` mechanics, with explicit `(1,1,0)` ero1 for this `(X,Y,Z)` array |
| tips | `ec_endpoint_bridge._extract_tips_fast` (two farthest-point tips and physical outward tangents) |

The dense GT reuses some values across disconnected objects. For identity audits and SDT targets,
the scripts first split it into 6-connected local GT components. The G0 graph paint remains exactly
`node_skeleton_index + 1`; a separate SDT-only paint stores the connected GT component at each
skeleton voxel so the SDT helper cannot silently fall back to ordinary EDT.

## Model outputs and GT targets

Warm-start the current MedNeXt-L and preserve the existing nine affinity channels. The dense output
has exactly 43 channels, indices 0–42, plus a sparse queried-edge head.

| idx | channel | exact target and role |
|---|---|---|
| 0–8 | existing affinity | Current banis+ `aff_r1 + aff_r10`, erosion-2 targets. These stay the merge-safe anchor. |
| 9 | graded medialness `m*` | For voxel `v` in instance `G_i`, nearest skeleton point `pi_i(v)`, radius `r_i(pi)`, and `s_xy=9 nm`: `sigma=clip(0.35*r_i(pi), s_xy, 2.5*s_xy)` and `m*(v)=exp(-||x_v-x_pi||^2/(2*sigma^2))`. **Set `m*(v)=0` exactly when `||x_v-x_pi|| > 0.6*r_i(pi_i(v))`.** This supplies nodes, never identity by scalar CC. |
| 10 | signed log boundary distance `d_b` | Interior-positive convention: `+log(1+d_in(v)/s_xy)` for `v in G`, and `-log(1+d_out(v)/s_xy)` for `v not in G`; `d_in/d_out` are physical EDTs to `partial G`, `s_xy=9 nm`. Radius/body support only; never the deciding identity signal. |
| 11–13 | signless tangent | Principal skeleton direction from PCA in physical window `R_t=max(4*s_xy,2*r_i)`. Loss `1-|t_hat dot t*|`. **Mask tangent loss wherever `q_branch=1` or `q_multi=1`.** |
| 14–39 | short SAME/MUTEX | Thirteen half-space 26-neighbor offsets, two logits per offset. SAME iff both endpoints lie in one instance medial band, their nearest skeleton points obey `d_S(pi_u,pi_v) <= 1.35*||x_u-x_v|| + 2*s_xy`, and the straight corridor tube (radius `0.35*min(r_u,r_v)`) crosses no foreign band. MUTEX iff instances differ or the corridor pierces a foreign band. Otherwise DEFER/ignore. |
| 40 | `q_multi` | In the veto neighborhood, at least two signless orientation modes differ by more than 35 degrees. |
| 41 | `q_foreign` | In the veto neighborhood, medial bands from at least two GT instances occur. |
| 42 | `q_branch` | In the veto neighborhood, at least three arms of one GT skeleton leave the neighborhood. |

All three vetoes use the physical ball

`R_v = max(3*s_xy, 1.5*r_i)`

about each voxel. The crossing/contact pattern is high `q_multi`, high `q_foreign`, low
`q_branch`; a true branch instead opens angle-separated degree slots.

### Sparse skip-edge target

The sparse head emits SAME/MUTEX logits for queried offsets in the full `|o|_inf=2` half-shell.
Its SAME/MUTEX/DEFER target uses the same local-path and foreign-corridor rule as short edges. A
skip query is eligible only where no accepted short-edge path already connects its endpoints.
Mark DEFER when the true skeleton path leaves the crop or is ambiguous. Accepted production skip
edges are painted as a narrow Hermite/geodesic centerline path; this painting is a Phase-2 learned
decode operation, not part of the G0 oracle.

## Training contract

Initial multi-task weighting is:

`1.00 L_aff + 0.25 L_anchor + 0.50 L_medial + 0.10 L_SDT + 0.15 L_tangent +`
`1.00 L_short-edge + 0.75 L_skip-edge + 0.50 L_veto`.

- Distill frozen high-confidence affinity logits with `L_anchor` so auxiliary heads do not damage
  the 0.627 anchor.
- Use focal BCE plus Huber for medialness, smooth L1 for `d_b`, independent focal BCE for SAME and
  MUTEX, and focal BCE for vetoes. Cross-GT/contact negatives receive extra weight.
- Do not use MALIS, clDice, cbDice, soft-skeleton, or another scalar topology loss as the deciding
  repair objective.
- Weight positive link sampling by clipped `sqrt(2*l_A*l_B)` and dangerous negative contacts by
  clipped `sqrt(l_A^2+l_B^2)`, preserving NERL's physical-length stakes without unstable extremes.
- Sample 50% marginal-NERL false-split sites, 25% hard contacts/Z gaps, and 25% uniform
  skeleton/background. Tune thresholds on a disjoint validation volume, never seed101.

Warm-start in three steps: train new heads with backbone and affinity head frozen; unfreeze the last
two MedNeXt stages at 0.1x the head learning rate; continue affinity rehearsal and hard-negative
mining from the actual decoder.

## Decode: link, grow, fill

### Conservative trunk graph

Decode medial nodes by hysteresis (`m>=0.65` high, `m>=0.25` low). Retain a low node only through
accepted edge support to a high node. Process short SAME edges in descending confidence. Reject a
union if it internalizes a hard MUTEX, fuses protected identities, crosses a two-plane barrier, or
has the crossing/contact veto pattern. Initial smoke thresholds are `p_same>=0.999` and
`p_mutex<=0.001`; production thresholds come from empirical risk control.

Consider a skip only if no short path exists, its physical distance is

`d <= min(750 nm, max(8*r_max, 3*s_z, 12*s_xy))`,

`|Delta Z|<=2`, tangent/radius evidence agrees, and no mutex/competitor trunk is crossed. Protect a
trunk at `L>=2 um`, or at `L>=0.75 um` with calibrated short-edge and `aff_r10` support. The
calibration contract is zero observed cross-GT protected trunks.

### Sparse repair and object context

Generate endpoint–endpoint, endpoint–trunk, protected–protected, and atomic
trunk–crumb–trunk candidates. Candidate scheduling may use `2*l_A*l_B`, but utility cannot override
a veto.

PairNet consumes an oriented raw-EM/affinity/medial/radius/tangent/veto/mask crop plus geometry and
predicts SAME, DIFFERENT, or DEFER. BiLOQ is an EM-trained seed-conditioned local object query run
forward and backward with competitor trunks as negative prompts. Both may reject; neither may
override a mutex. Calibrate each event stratum for selective prediction; unsupported strata remain
DEFER.

Score every candidate against one immutable graph snapshot. Commit mutual-best, margin-qualified,
degree-constrained local transactions atomically. Each ordinary fragment has degree capacity one in
an uncertain repair round; true branches open separated angular slots; a crumb two-sided repair is
one hyperedge. Newly attached fragments are not proposal anchors in the same round. This is local
set packing, not global multicut/agglomeration.

### Merge-safe grow and abstaining fill

Build strict face-affinity supervoxels. Assign an SV only if it touches one centerline identity;
split or abstain when it touches several. Orphans compete by affinity/body/geodesic cost and remain
UNKNOWN unless the runner-up margin is at least `log(9)`. Body growth cannot union centerline IDs.

Production fill is bounded by predicted body support, radius envelope, hard mutexes and competitors,
a 250 nm outside-body step, `log(9)` competition margin, and a 30% local-volume cap. Filled voxels
never become link nodes, proposal anchors, or sources of an unbounded second wave.

## G0: oracle plumbing gate

Paint 1-voxel skeleton nodes and generate GT SAME edges. Strip identity values from decoder input;
union-find receives only node occupancy plus accepted endpoint pairs. For offsets longer than one
voxel, reject a same-edge target when its digital-line interior crosses a foreign connected GT
component. Spatial CC is never allowed to create graph unions.

The cumulative `(X,Y,Z)` banks are:

- B0: 13 half-space 26-neighbors.
- B1: B0 plus `(2,0,0)`, `(0,2,0)`, `(0,0,2)` = 16.
- B2: B1 plus every `(dx,dy,2)`, `dx,dy in {-1,0,1}` = 24.
- B3: B0 plus the 49-offset `|o|_inf=2` half-shell = 62.

For C0, grow with `watershed(-gt_sdt, markers=components, mask=(gt_sdt>-0.3) & GT_fg)`.
Report canonical local NERL, grown labels spanning multiple GT components, mean distinct grown labels
over all 626 skeletons, and retained physical edge length. Graph-only perturbations reset
`default_rng(101)` per condition:

- C1 deletes up to 300 degree-2 interior voxels and requires both graph neighbors to reconnect.
- C2 deletes each eligible skeleton's median occupied interior Z section and requires every cut
  strand's nearest surviving Z-side anchors to reconnect.
- C3 deletes sampled contiguous 2–5-node arcs and reports closure by length.

Pass the smallest bank with zero cross-GT unions/false merges, fragments/GT `<=1.02`, retained
length `>=99.5%`, C1 `>=95%`, C2 `>=90%`, and C0 GT-SDT NERL `>=0.990`. Kill on any cross-GT union,
identity-dependent decode, best-bank NERL `<0.985`, or fragments/GT `>1.05`.

G0 proves decode mechanics only. Learned medial-band nodes, signed logits, skip painting, and their
false-merge calibration remain Phase 2.

## G1: banis+ no-retrain feasibility

Bind `BASE_NERL` to the measured center decode, not rounded 0.836. Assert each positive baseline
fragment overlaps only one connected dense GT component and owns nodes from at most one local
skeleton.

### Break-cluster oracle and utility

A cut skeleton edge has two positive, different baseline fragment labels. Group all cuts by local
skeleton and unordered fragment pair; retain every physical midpoint and discard an event only when
all breaks are within two voxels of a crop face. Full oracle unions every same-skeleton fragment.

For event `(f_a,f_b)`, assign full length to an intra-fragment skeleton edge and half a cut edge to
each endpoint fragment. With `D=sum_s L_s^2`, use `DeltaNERL=2*L_a*L_b/D`. Compare a deterministic
20-event sample with the canonical scorer; if any absolute error exceeds `1e-3`, fall back to
canonical per-event deltas for the bounded event set.

### Proposal geometry (physical nm)

Tips come from `_extract_tips_fast`. For each tip, compute the fragment-mask
`distance_transform_edt(mask, sampling=[9,9,20])`; `r_est` is the median EDT over its 12 physically
nearest fragment voxels. For tips `u,v`:

- `d=(p_v-p_u)/||p_v-p_u||` and `A=0.5*(t_u dot d + t_v dot -d)`.
- Require distance `<=min(750,max(8*r_max,60,108))` nm.
- Proposal gates: `A>=0.65`, radius ratio `<=3.0`, `rho_min>=1.25*r_max`.
- Acceptance gates: `A>=0.80`, radius ratio `<=2.5`, `rho_min>=1.5*r_max`.

For curvature, let chord `L=||p_v-p_u||`, Hermite handles `m_u=L*t_u`, `m_v=L*t_v`, sample the
standard cubic Hermite curve at 32 points, take finite-difference `H'` and `H''`, and compute
`rho=||H'||^3/||H' x H''||` (`infinity` for near-zero cross product). Use the minimum sampled rho.
Fragments below 12 voxels and undefined tangents are skipped with counts.

Generate all proposals. Report mean/p95/max raw candidates per endpoint. Keep deterministic top-8 at
either endpoint ranked by agreement descending, distance ascending, and partner fragment ID
ascending. A proposal covers an event only when the fragment pair agrees and at least one proposed
tip is within 45 nm of any break in the cluster.

### Matching, upper bound, and realizer

Geometry-only matching uses one immutable round. Collapse duplicate candidates per fragment pair,
rank by agreement descending, distance ascending, then unordered fragment IDs, and accept only a
mutual fragment-level top choice with both fragments free. Thus every baseline fragment has degree
at most one and no two-tip fragment can form a chain. Report precision, false merges, and NERL; this
diagnostic is expected to wall and is not a gate.

The identity-gain upper bound unions only capped-covered GT-same events. Report

`(NERL_capped_oracle-BASE_NERL)/(NERL_full_oracle-BASE_NERL)`.

Flag gain `<0.02` as low split headroom and recommend `tile_0_0_2`; do not gate recovered fraction on
that crop.

For realization, 6-connect the `BASE_SEG==0` mask into pseudo-fragments (the explicit
face-graph-consistent connectivity assumption), add positive base fragments, and aggregate shared
faces. Edge cost is `-log(mean aff_r1 + 1e-6)` using the channel matching the shared-face axis; the
reverse step uses that same undirected face. Supervoxels stay atomic. Multi-source graph search keeps
the best and runner-up distinct identities; adopt an orphan only at margin `>=log(9)`, otherwise set
UNKNOWN=0.

Run both:

- control: each skeleton-bearing base fragment is its own identity;
- repaired: seed identities first union capped-covered same-GT events.

The control must stay within 0.005 of measured `BASE_NERL`, with zero cross-GT identities and zero
multi-ID assignments. The repaired run requires dense-GT adoption precision `>=0.95`, nonzero
coverage (target `>=0.5`), reported abstention, zero cross-GT identities, and grown NERL at least
pre-grow repaired NERL minus 0.005.

G1 passes when count recall `>=80%`, weighted recall `>=90%`, raw max workload `<=8`, both realizer
contracts pass, and recovered fraction `>=0.60` when full-oracle gain is at least 0.02. Kill when
weighted recall `<80%`, raw max `>12`, recall needs whole-volume/global BFS, or realization fails.
Whole-volume 0.752/0.742 gates do not apply to this coverage-limited crop.

## Experiment ladder and stop rules

| phase | experiment | success | kill |
|---|---|---|---|
| 0 | This G0 CPU oracle | merge-safe GT-SDT NERL `>=0.990`, fragment and perturbation gates above | Any cross-GT union, best `<0.985`, identity-dependent decode |
| 1 | Candidate recall, first as this G1 surrogate and then with predicted medial heads | `>=80%` count, `>=90%` DeltaNERL-weighted, bounded workload | Weighted `<80%`, max workload `>12`, global tracing required |
| 2a | Predicted medial nodes only, then +short edges | no decrease below `-0.002`; short-edge cumulative delta `>=+0.025` off 0.627 | Any protected merge |
| 2b | +skip, +veto/local matching | skip cumulative `>=+0.050` and incremental `>=+0.015`; veto cumulative `>=+0.070` | Skip adds `<+0.015` or any merge |
| 2c | +PairNet and BiLOQ | final heal `>=0.727`, target `0.742`; BiLOQ incremental `>=+0.015`; zero merges | Final delta `<+0.080`, unbounded risk strata |
| 3 | Grow then bounded fill | grow within 0.005; fill `>=+0.040`, target `+0.050`; no ID change/merge | Fill `<+0.020`, any seed-ID change or merge |

Every row reports NERL, fragments/GT, physical retained/recovered length, cross-GT graph components,
new node-LUT merges, accepted links by stratum, and abstention. No GPU training starts unless G0 and
the G1 cheap gates pass. A gray or low-headroom center result is a no-go until the named cheap
refinement, especially `tile_0_0_2`, is measured.

## Forecast and falsified boundaries

The honest central forecast is `0.627 + 0.115 heal + 0.050 fill = 0.792`; stretch is 0.812. All
realized deltas are quoted off 0.627. The 0.772 heal oracle and 0.921 heal+fill oracle are ceilings,
not engineering results. Reaching 0.80 requires both identity repair and bounded fill.

Do not chase MALIS; scalar SDT/topology losses as the identity signal; 2x-XY; banis2 occupancy;
waterz/ABISS/low-threshold salvage; naive EDT fill; global multicut/agglomeration/BFS flood;
whole-volume rooted tracers; or methods requiring perfect production seeds. MESA-EM clears the
frozen-map and thin-context constraints only through learned SAME/MUTEX relations plus sparse
PairNet/BiLOQ object context.

## Failure audit

Parallel equal-caliber bundles, X/T contacts, missing Z sections, two-sided crumbs, self-contacts,
abrupt caliber changes, baseline merges, SV membrane crossings, borders, and calibration shift are
the principal risks. Mutex/veto channels, competitor trunks, two-plane barriers, strict range and
degree limits, atomic hyperedges, bidirectional confirmation, identity-preserving growth, and
UNKNOWN abstention reduce them. Where image/context evidence cannot distinguish two identities, the
required output is DEFER; the method intentionally leaves some heal headroom unrecovered.

The first deployment check after these gates is a second local skeleton/affinity crop from
`tile_0_0_2`, especially if the center full-oracle gain is below 0.02. Thresholds and risk bounds
must then be calibrated on a disjoint volume before any seed101 report is treated as realized gain.
