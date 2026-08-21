# Segmentation Checkpoint DSL

The Segmentation Checkpoint DSL is a typed intervention record for local refinement of a
provisional segmentation. Its stable protocol is:

```text
DESCRIBE -> CERTIFY -> ACT -> VERIFY
```

The protocol deliberately separates immutable measurements, trusted assertions, requested
actions, deterministic execution, and postcondition checks. A measurement can be reused by a
different policy without granting that policy write access. An action plan can be inspected,
hashed, replayed, rejected when its inputs change, and verified independently of the code that
requested it.

## Descriptions, not biological classes

Core descriptors state what was measured. For example, `anchor.distinct_count`,
`anchor.overlap_fraction`, and `component.volume_voxels` are observations. They do not assign a
cell class. The core schema therefore has no semantic flags such as glia, neuron, vessel, or cell
type.

A nucleus instance is a trusted identity anchor, not a cell-class prediction. Its contract is
narrow: one trusted nucleus belongs to one cell, and two distinct trusted nucleus IDs imply two
distinct identities. One nucleus does not prove that its whole component is pure, and absence of
a nucleus says nothing.

Only the `nucleus_anchor` pass is implemented. It detects components with significant mass from
multiple nucleus instances, certifies that identity conflict, and uses a six-connected seeded
flood to refine eligible caller-supplied contact scopes. The policy does not try to distinguish
cell types or infer morphology. In particular, a post-hoc split is a local partition refinement;
it is not mathematically equivalent to online constrained agglomeration.

## Pass composition

Operators implement a small common interface and run sequentially with separate provenance and
action logs. A later pass can consume the prior pass's corrected artifact without sharing its
detector or policy. Future passes may emit descriptive keys such as:

- `morphology.porosity`
- `morphology.sheetness`
- `graph.contact_degree`
- `graph.distinct_tube_contacts`
- `growth.threshold_velocity`
- `junction.incident_arm_count`
- `junction.orientation_modes`
- `acquisition.tile_seam_distance`

Adding those descriptors does not change the executor. A future agent may read immutable
descriptions and emit the same validated `ActionSpec` records, but it cannot mutate segmentation
data directly. Only the deterministic executor applies a serialized plan after checking its
configuration, input artifacts, anchor totals, and action preconditions.

## Actions and later decoding

The implemented action order is `split_by_anchor`, `consolidate_same_anchor`, `forbid_merge`, then
`rebuild_local_rag`. Exclusion must precede same-anchor consolidation. The cannot-link is bound to
the final consolidated anchor territory sets, so it never references piece IDs removed by
consolidation.

These concepts are distinct:

- `forbid_merge` is a durable hard constraint between distinct final anchor territories.
- `hold_edge` is a typed future placeholder for a temporary decision; it has no executor yet.
- local partition refinement splits one provisional component only inside a declared repair
  scope.
- later global reclustering is a fresh decoder pass over the corrected graph and constraints.

The canonical `cannot_links.json` manifest is currently write-only. This repository has no global
decoder that consumes a constraint list: `NUC_PATH` is a mask input and `nuc_cuts.data` is an
output. A future consumer must accept the manifest's final territory IDs, expand every pair in
`pairs` as an immutable cannot-link, and reject or rebind constraints when segmentation identity
changes. Exportability is implemented; decoder consumption is not.

## Containment and claims

A bounded ROI can leave two new territories joined through the untouched exterior. The operator
therefore examines the actual repair ROI faces. If the parent component can continue through any
interior face, the plan records `scope.containment: partial` and
`separation_claim: local_only`. It still performs the local refinement but excludes that component
from the verified repaired set. Verification additionally rejects any newly assigned territory
touching a face where the original parent continues. Only a proven-contained component is subject
to the hard no-multi-anchor output invariant.

The result always enumerates both `repaired_components` and `certified_unrepaired`, including the
reason for each unresolved certificate. Silence is never treated as successful repair.

## Complete pass specification

Specifications are data-only YAML or JSON. Conditions use the fixed `field`, `operator`, `value`
syntax; arbitrary expressions and Python evaluation are not supported. `min_share` has no default
and is supplied explicitly on the command line.

```yaml
checkpoint_id: worst3_nucleus_anchor
passes:
  - pass_id: nucleus_anchor
    operator: nucleus_anchor
    segmentation:
      uri: file:///path/to/provisional/precomputed/segmentation
    nuclei:
      uri: /path/to/nucleus_instances.h5
      dataset: main
    affinity:
      uri: file:///path/to/prediction.h5.chunks
      dataset: main
    scope_zyx: [1260, 9324, 2772, 2520, 10584, 3780]
    contact_scopes: /path/to/contact_scopes.json
    channel_indices: [0, 1, 2]
    affinity_channel_axis: 0
    affinity_convention: banis
    sigmoid_restore: 0.2
    pooling_factor: 4
    nucleus_scale_zyx: [4, 8, 8]
    max_read_bytes: 67108864
```

`describe` writes a separately hashed `anchor_totals.json`. That exact artifact is required by
`plan`, `apply`, and `verify`.

## Example emitted plan

The actual JSON includes complete provenance and typed entity records; this abbreviated example
shows the policy boundary and ordering.

```json
{
  "schema_version": "1.0",
  "operator_name": "nucleus_anchor",
  "certificates": [
    {
      "certificate_type": "distinct_anchor_identity_conflict",
      "strength": "hard",
      "distinct_anchor_ids": ["275", "319", "373"]
    }
  ],
  "actions": [
    {
      "operation": "split_by_anchor",
      "parameters": {
        "connectivity": 6,
        "tie_break": "lowest_anchor_id",
        "separation_claim": "local_only"
      },
      "preconditions": [
        {"field": "anchor.distinct_count", "operator": "ge", "value": 2}
      ]
    },
    {"operation": "consolidate_same_anchor"},
    {"operation": "forbid_merge", "parameters": {"binding": "anchor_territory_sets"}},
    {"operation": "rebuild_local_rag", "parameters": {"mode": "recompute"}}
  ]
}
```

## CLI

`scripts/checkpoint.py` exposes `describe`, `plan`, `apply`, `verify`, and `run`. `run --dry-run`
stops after serializing the inspectable plan. Applying a frozen plan produces a sparse segmentation
delta rather than copying the whole volume, an append-only execution log, a local RAG record, and
the canonical write-only cannot-link manifest.
