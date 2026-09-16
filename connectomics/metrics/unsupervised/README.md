# Metrics without ground truth

Use these modules directly; they accept arrays or measurements and perform no
dataset loading, report writing or cloud publishing.

| Module | Input | Result |
|---|---|---|
| `morphology` | Integer ZYX label volume and physical voxel spacing | Per-label profiles, radius, length, angles, crop continuity and candidate classes; aggregate radius/angle statistics |
| `arbor` | Physical ZYX skeleton vertices, edges and local radii | Protected longest paths, caliber, conservative twig pruning and branching measurements |
| `classification` | Profile candidate, voxel count, optional arbor and reviewed class | Separate profile, automatic and display classes with their evidence basis |

`tube` remains the existing diagnostic for tube-like, predominantly Z-directed
objects. Morphology profiles follow each object's dominant physical PCA direction
and support other orientations. Profile length and skeleton length measure
different things; both are approximations and neither supplies GT NERL.

## Label-volume measurements

```python
import numpy as np

from connectomics.metrics.unsupervised.morphology import MorphologyConfig, analyze_morphology

labels = np.zeros((80, 32, 32), dtype=np.uint64)
labels[:, 14:18, 14:18] = 7
analysis = analyze_morphology(
    labels,
    MorphologyConfig(voxel_size_um=(0.1, 0.1, 0.1), min_voxels=1),
)
record = analysis.records[0]
print(record.label, record.length_um, record.radius_um, record.morphology_class)
print(analysis.summary["radius_bins"])
```

Set `min_voxels=1` to include crumbs in the measurements. The default analysis
cutoff is 1,000 voxels; it excludes smaller objects before classification. Labels
are never relabeled or mutated. Sparse positive uint64 IDs are preserved, and
background `0` has no record.

`MORPHOLOGY_CLASSES` defines the canonical profile classes. Low PCA elongation
with multiple section components is `branched_process_candidate`: both dendrites
and glia can have that geometry. Equivalent section radius can be inflated by
branching; local skeleton caliber offers additional evidence.

## Skeleton measurements and class decisions

```python
from dataclasses import asdict
import json

from connectomics.metrics.unsupervised.arbor import ArborConfig, BackboneConfig, analyze_arbor
from connectomics.metrics.unsupervised.classification import (
    SegmentClassificationConfig,
    classify_segment,
)

# A trunk with a fine side branch; coordinates and radii are already in µm.
vertices = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0], [4, 0.4, 0]])
edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
radii = np.array([0.2, 0.2, 0.2, 0.04])
arbor = analyze_arbor(
    vertices, edges, radii, ArborConfig(max_twig_radius_um=0.15)
)
decision = classify_segment(
    "branched_process_candidate",
    voxel_count=2000,
    arbor=arbor,
    config=SegmentClassificationConfig(
        crumbs_max_voxels_exclusive=1000,
        backbone=BackboneConfig(
            min_diameter_length_um=5,
            min_diameter_radius_um=0.15,
            min_diameter_to_width_ratio=10,
        ),
    ),
)
print(decision.automatic_class)  # dendrite_like_candidate
payload = {"id": "7", "classification": asdict(decision), "arbor": asdict(arbor)}
print(json.dumps(payload, allow_nan=False))
retained_edges = edges[list(arbor.retained_edge_indices)]
```

`arbor` accepts a graph from any skeletonizer; it does not run one. Coordinates
are ZYX micrometers in the same frame as the volume; volume-derived voxel-center
coordinates are `(index + 0.5) * voxel_size_um`. Edges are zero-based integer
vertex pairs, and radii must be strictly positive. Keep vertex and edge order
when using the returned retained/diameter indices for drawing or manipulation.

Twig pruning requires all three limits: original distal reach, median radius,
and radius relative to its attachment. It preserves every forest component's
longest path and tracks original distal reach across repeated pruning passes.
The default twig radius limit is 0.10 µm; 0.15 µm reproduces the cerebellar report.
Cycles disable pruning and make overall diameter measurements unavailable.

The backbone gate requires one acyclic component with enough length, caliber
and length-to-width ratio. Its diameter and caliber are protected during pruning,
so pruning thresholds change branch counts rather than the class gate. TEASAR
produces forests by construction; tree topology does not establish neuron identity.

`classify_segment` applies these rules in order:

1. A passing backbone gate supplies a dendrite candidate; otherwise keep the
   profile class, including an existing profile-based dendrite candidate.
2. Below the configurable voxel cutoff, display `crumbs` regardless of geometry
   or annotation. Use a cutoff of one to disable this size category.
3. For larger objects, use `reviewed_class` when supplied, otherwise the automatic
   candidate. A reviewed label such as `glia` changes only the display decision.

Store review provenance separately from this result. The class decision does not
certify segmentation correctness or produce a calibrated probability. A neutral
branched class can include neurons, glia and segmentation errors; a clean-looking
axon can still contain a false merge. The crumbs threshold is a voxel-count rule
and must be adjusted when the sampling grid changes.

`MorphologyRecord` can also be converted with `asdict`, but some degenerate
profile measurements are undefined/infinite. Reporting adapters must encode
unavailable values as JSON `null`. Arbor results use `None` for unknown values;
class decisions are directly JSON-safe. Encode label IDs as decimal strings in
webapp payloads to preserve values beyond JavaScript's safe integer range.

The dataset-specific workflow in `dev/astra_nogt_eval/glia_revision/` imports
these modules. Its legacy-class migration, frozen catalog/schema, review records,
file paths and GCS publication stay outside the metric package.
