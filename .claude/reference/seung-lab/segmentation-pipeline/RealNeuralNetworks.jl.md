# RealNeuralNetworks.jl

**GitHub:** https://github.com/seung-lab/RealNeuralNetworks.jl
**Language:** Julia | **Stars:** 41

Unified Julia framework for skeletonization, morphological analysis, and connectivity analysis of 3D neuron models extracted from EM image segmentation.

## Key Features
- Sparse skeletonization from colored segmentation labels (TEASAR algorithm)
- Extensive morphological analysis (path length, Sholl analysis, convex hull, tortuosity, fractal dimension, branching angles)
- Synapse-aware neural network models (unlike most morphology tools)
- NBLAST algorithm for neuron morphology similarity measurement
- CloudVolume integration for reading segmentation from cloud storage
- SWC format export
- Docker support with cloud authentication

## API
```julia
using RealNeuralNetworks.NodeNets, RealNeuralNetworks.Neurons, RealNeuralNetworks.SWCs

nodeNet = NodeNet(seg::Array{UInt32,3}; obj_id=convert(UInt32, 77605))
neuron = Neuron(nodeNet)
swc = SWC(neuron)
SWCs.save(swc, "output.swc")
```

## Morphological Features
- Whole-neuron: path length, Sholl analysis, hull area/volume, tortuosity, asymmetry, fractal dimension
- Per-segment: order, length, branching angle, curvature, radius, distance to root

## Relevance to Connectomics
The primary Julia tool for extracting and analyzing neuron morphologies from EM segmentations, including skeletonization and NBLAST-based cell type classification.
