# NEURD Reference

**GitHub:** https://github.com/reimerlab/NEURD
**Paper:** [Nature 2025](https://www.nature.com/articles/s41586-025-08660-5)
**Stars:** 22 | **Language:** Python (Jupyter Notebook heavy)
**Docs:** https://reimerlab.github.io/NEURD/

A mesh decomposition framework for **automated proofreading** and **morphological analysis** of neuronal EM reconstructions. Decomposes neuron meshes into branches/limbs, detects errors via graph-based filters, and produces corrected skeletons with compartment labels.

## Core Idea

Takes a segmented neuron mesh → decomposes into a hierarchical graph (soma → limbs → branches) → applies graph filters to detect merge/split errors → outputs a proofread skeleton with axon/dendrite/soma labels.

Unlike voxel-based approaches (our waterz pipeline), NEURD operates on **meshes** — it processes the output of a segmentation pipeline, not raw affinities. It's a **downstream consumer** of segmentations like ours.

## Architecture

```
Neuron Mesh (from segmentation)
  → Soma extraction (mesh clustering)
  → Limb decomposition (connected components after soma removal)
  → Branch decomposition (skeleton-guided mesh splitting via CGAL)
  → Concept network (hierarchical graph: Neuron > Limb > Branch)
  → Graph filters (error detection on branch graph)
  → Proofreading (split/merge suggestions)
  → Compartment labeling (axon/dendrite/soma/AIS)
  → Morphological features (width, spine density, boutons, etc.)
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `neuron.py` | Core `Neuron` class — hierarchical data structure (soma/limb/branch) |
| `neuron_pipeline_utils.py` | Full pipeline: mesh → decomposition → proofreading → classification |
| `preprocess_neuron.py` | Mesh decomposition into limbs and branches |
| `error_detection.py` | Low-level error detection (double-back, width jumps, skeleton angles) |
| `graph_error_detector.py` | Graph filter framework for structured error detection |
| `graph_filters.py` | Specific filter implementations (upstream pair matching, degree checks) |
| `proofreading_utils.py` | Split/merge suggestion generation and application |
| `axon_utils.py` | Axon identification and tracing |
| `apical_utils.py` | Apical dendrite detection |
| `spine_utils.py` | Dendritic spine detection on mesh branches |
| `synapse_utils.py` | Synapse association with branches |
| `cell_type_utils.py` | Excitatory/inhibitory classification |
| `gnn_cell_typing_utils.py` | GNN-based cell type classification |
| `connectome_utils.py` | Connectivity matrix construction and analysis |
| `proximity_utils.py` | Inter-neuron proximity detection |
| `width_utils.py` | Branch width measurement from mesh |
| `soma_extraction_utils.py` | Soma mesh extraction |
| `soma_splitting_utils.py` | Multi-soma neuron splitting |
| `vdi_default.py` | Volume Data Interface — abstract data access layer |
| `vdi_microns.py` | MICrONS dataset interface |
| `vdi_h01.py` | H01 (human cortex) dataset interface |

## Error Detection Approach

NEURD detects errors via **graph filters** on the branch decomposition graph:

### Merge Error Detection
- **High-degree branching**: Nodes with degree > 3 in skeleton graph → check if branches are compatible (skeleton angle, width continuity, synapse density)
- **Width jump detection**: Sudden width changes along a path suggest a merge of different neurites
- **Double-back detection**: Branch that reverses direction suggests it belongs to a different neuron
- **Axon-on-dendrite**: Thin axonal branch attached to thick dendrite → likely false merge

### Split Error Detection
- Not the primary focus of NEURD — it assumes the segmentation is over-merged rather than over-split
- Relies on upstream proofreading tools (e.g., PyChunkedGraph) for split correction

### Filter Pipeline
```python
# Pseudocode from graph_error_detector.py
for each high-degree node in skeleton graph:
    1. Check distance from soma (skip if too close)
    2. Filter short endpoints (< min_skeletal_length)
    3. For each upstream-downstream pair:
       a. Compare skeleton angles (alignment)
       b. Compare widths (continuity)
       c. Compare synapse densities
       d. Score match quality
    4. If best match score < threshold → flag as merge error
    5. Generate split suggestion (which branches to detach)
```

## Data Structure: Neuron Object

```
Neuron
├── soma (mesh, center, radius)
├── limbs[] (one per connected component after soma removal)
│   ├── branches[] (skeleton-guided mesh segments)
│   │   ├── skeleton (3D coordinates)
│   │   ├── mesh (trimesh object)
│   │   ├── width_array (per-skeleton-node width)
│   │   ├── synapses[] (associated synapses)
│   │   ├── spines[] (detected spines)
│   │   └── labels (axon/dendrite/etc.)
│   └── concept_network (networkx graph of branch connectivity)
└── neuron_graph (full skeleton graph)
```

## Dependencies

**Core:** numpy, scipy, networkx, trimesh, meshparty, pykdtree, pymeshfix, scikit-learn, matplotlib
**Data access:** datajoint, datasci-stdlib-tools, neuron_morphology_tools
**ML (optional):** torch, torch_geometric (for GNN cell typing)
**Mesh processing:** CGAL (via Docker — C++ mesh segmentation/skeletonization)

## Relevance to PyTC Decoding

NEURD operates **downstream** of our segmentation pipeline — it takes a segmented neuron mesh and proofreads it. Key connections:

### What NEURD does that we don't
1. **Mesh-based error detection**: Uses 3D mesh geometry (width, angles, surface area) rather than voxel-based affinity
2. **Structured decomposition**: Soma → limb → branch hierarchy enables local reasoning about errors
3. **Morphology-aware proofreading**: Width continuity, synapse density, and skeleton angle are strong signals for merge detection
4. **Multi-soma splitting**: Can detect and split neurons that were falsely merged at the soma level

### Ideas to port to PyTC (voxel-level)
1. **Width-based merge detection**: NEURD's width-jump filter could be adapted for voxel-based segments — compute skeleton width (via EDT) and flag segments with discontinuous width profiles
2. **Skeleton angle matching**: For our Stage 3 (skeleton split/re-merge), NEURD's upstream-downstream angle comparison is a proven criterion
3. **Graph filter framework**: The `graph_error_detector.py` pattern (parameterized filters applied to a neuron graph) could structure our branch_merge stages
4. **Multi-soma detection**: Before waterz agglomeration, detect soma regions and prevent cross-soma merges (similar to zwatershed's `somaBFS`)

### What we do that NEURD doesn't
1. **Voxel-level affinity-based segmentation**: NEURD assumes meshes are already available
2. **Training + inference pipeline**: NEURD is pure post-processing
3. **Affinity-based merge evidence**: We use raw model predictions; NEURD uses geometry only

## Tutorials

| Tutorial | Description |
|----------|-------------|
| Auto Proofreading Pipeline | Full pipeline: mesh → decomposition → proofread |
| Neuron Features | Hierarchical data access, feature extraction |
| Proximities | Inter-neuron contact analysis |
| GNN Cell Typing | Graph neural network cell classification |
| VDI Override | Custom dataset integration |
| Spine Detection | Dendritic spine detection on mesh branches |
