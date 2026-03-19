# ReconstructionEvaluations.jl

**GitHub:** https://github.com/seung-lab/ReconstructionEvaluations.jl
**Language:** Julia | **Stars:** 1

Julia package to evaluate neuronal reconstructions from electron micrographs. Computes reconstruction quality metrics including NRI (Normalized Reconstruction Index).

## Key Features
- Load corrected and uncorrected consensus edge lists
- Build count tables from ground truth vs reconstruction edges
- Compute NRI (Normalized Reconstruction Index)
- Assess broken spines, axon splits, and other reconstruction errors

## API
```julia
ground_truth = load_edges(corr_fn)
reconstruction = load_edges(uncorr_fn)
count_table, corr_to_inds, uncorr_to_inds = build_count_table(ground_truth, reconstruction)
compute_nri(count_table)
```

## Relevance to Connectomics
Provides quantitative evaluation of neuron reconstruction quality, essential for benchmarking segmentation and proofreading accuracy.
