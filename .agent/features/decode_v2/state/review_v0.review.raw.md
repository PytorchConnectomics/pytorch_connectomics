Claude (planner) code review of code_v0 — raw notes.

Read dev/mit_liconn/decode_v2.py (680 lines) and force_split_decode.py force_split extraction.
Re-ran `--self-test`: PASS. Codex-reported pilot `--zslice 0:96`: exit 0, 47 fusions, 9 eligible,
87 sections split, a/b/c F1 0.9133 (unchanged on z0:96 — expected, few merges there).

Invariants verified against the spec:
- Constrained union-find (ConstrainedUnionFind + constrained_relabel): run->side dicts, unions
  rejected on conflict; forced mand_links applied first; ASSERTS both separation (sides' roots
  differ) AND connectivity (every piece/terminal in its seed anchor's component) at lines 266-275.
  Correct — this is the central invariant.
- Eligibility (_upstream_reachable/eligible_fusions): traverses only decreasing-z edges, blocks t,
  requires disjoint upstream components each >= min_len distinct z-slices (axial depth). Upstream-
  only, resolves the plan_v2/v3 convergence + branching findings. Self-test covers it.
- Detection: exactly-2, --fuse-iou (0.2, separate param), area-matched, N>2 skipped+counted.
- Background 0: section_index excludes label 0 (labels>0); self-test asserts bg repeats legal +
  nonzero cross-slice repeat rejected.
- Modes: mutually exclusive --full/--zslice/--self-test required; --self-test early-exits; slice
  bounds 0<=z0<z1<=depth; shape asserts. Slice artifacts z-suffixed (no full-vol overwrite).
- force_split: all stop causes explicit (conflict_skip/seed_skip/volume_end/no_next/owned_successor/
  two_incoming_fusion/degenerate_split/ambiguous_successor/fusion_successor_pair/separated); atomic
  claim() with owner assertions; terminal one-to-one bijection (_terminal_assignment, higher total,
  per-side min overlap); mand_links thread each side incl. terminals.
- No-regression: selected=argmax real NERL{a,c}; asserts selected>=max(recomputed a, 0.593);
  regression banner; GT-based selection flagged EVAL-ONLY.

MATERIAL ISSUE:
[major] Line 501-503 hard-asserts substrate oracle-merge NERL >= 0.78 (SUBSTRATE_ORACLE_FLOOR).
That 0.78 was measured for force-split-ALL (534 runs / 9421 sections split — splits ALL fusions
incl. same-axon over-splits). decode_v2 splits ELIGIBLE fusions ONLY (~9 runs/87 sections on z0:96;
~75 runs full-volume — far fewer). The eligible substrate therefore splits much less than the
force-split-all baseline, so its oracle-merge NERL sits near the unsplit-sections value 0.760, very
likely < 0.78 -> the full-volume verification (the primary gate) ABORTS at line 501 before
producing NERL. Root cause: plan_v3 fixed the floor to the wrong (more aggressive) baseline's value.
Fix: bind the hard floor to >= 0.760 (the unsplit-sections baseline — the eligible substrate must
not regress vs no split) and report the exact substrate value; keep 0.78 only as a REPORTED target,
not a hard abort. One-line change.

Everything else is correct and matches the spec; no other major issues found.
