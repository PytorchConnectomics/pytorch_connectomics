# CCC Task

Build the second-pass error-correction model as an **end-point growth** task, replacing
the crop/seed formulation in feature `ec_model`.

a) Read `.agent/features/ec_end_pt/design.md` first — it is the agreed design and carries
   the measurements behind every choice. Do not re-derive; do re-verify anything you are
   about to depend on.
b) The task is deployment-shaped: enumerate tips of the *predicted* segmentation's
   skeletons, centre a crop on each tip, and grow the fragment that owns the centre voxel.
   The centre is the prompt — no movable seed, no promptable segmentation.
c) Seed only from trunks (cable ≥ 1 µm); crumbs stay in the target but are never seeded.
   Exclude tips on an ROI face.
d) Precompute all candidate centres per tip (accept while write-region overlap with kept
   centres < 0.80); each centre is its own training example. No on-the-fly jitter.
e) Keep the 128³ crop / 64³ write region. Do not enlarge it — see design.md §6.
f) Gate on segmentation, not masks: max GT span must stay 1, frags/GT must fall, and the
   model must beat `b0 = (seg == owner)`. `dev/ec_model/decode_merge.py` scores 1 and 2.
g) Read the prior lessons in
   `/projects/weilab/weidf/lib/dw-research/projects/2026_nisb_base` and the previous
   attempt in `dev/ec_model` — the gap-centred design there could not deploy, and
   design.md §2 explains exactly why so it is not rebuilt by accident.
h) Two decisions are still open and are the user's, not yours: the cap on centres per tip
   (§5) and how to re-mine val A without spending the B reservation (§11).

Invocation: /ccc /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/ec_end_pt/ "<task above>" (defaults: p2-c2, normal, plan-code from default)
Repository root: /projects/weilab/weidf/lib/pytorch_connectomics
Worktree: /projects/weilab/weidf/lib/pytorch_connectomics/.claude/worktrees/ccc-ec_model
