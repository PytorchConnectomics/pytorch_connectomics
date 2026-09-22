# Task

Read the latest axon error-correction (EC) decode method design and lessons, then produce a
**reframed, concise method-design document** at `.agent/features/ec_axon/method.md`.

Be critical and constructive: find a better framing of the core method and improve it.
**Remove the bloat / overfitting** — the case-by-case gates, per-example seg-ids, superseded
sub-methods, and error taxonomy that read as fit-to-this-volume rather than the durable idea.
Keep it **concise but not too simple**: the reader should understand the method, the signals it
relies on, why it works, and its known limits — without wading through the history.

## Source material (to synthesize, not copy)
- `dev/mit_liconn/PIPELINE.md` — the CURRENT pipeline (v0 → v1 split → v2 merge → v3 weak), the
  freshest framing (split-then-merge; IoU-primary + cross-section completion; thr=10 fair metric).
- `/projects/weilab/pytc-agent/projects/2026_mit_liconn/tracklet.md` — the prior "v3_ab" design
  (comprehensive but bloated: M1–M8 taxonomy, per-case gates, seg-ids — trim this).
- `/projects/weilab/pytc-agent/projects/2026_mit_liconn/decode_3round_method.md` — older 3-round framing.
- `/projects/weilab/pytc-agent/projects/2026_mit_liconn/findings.md` L59–L74 — the durable lessons
  (the metric-fairness finding, the sep=0 wall, what worked vs the shape-ambiguity dead ends).

## Deliverable
`.agent/features/ec_axon/method.md` — a single self-contained method design. Markdown only; no code
changes. Success = a senior reader grasps the method and its limits in one read, with no overfit detritus.
