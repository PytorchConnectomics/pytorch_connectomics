# ImageRegistration.jl

**GitHub:** https://github.com/seung-lab/ImageRegistration.jl
**Language:** Julia | **Stars:** 11

An image registration toolbox for Julia. Creates point set correspondences via block matching, calculates geometric transforms (affine, rigid, translation, piecewise affine), and renders images with those transforms.

## Key Features
- Block matching with parallel computation support
- Affine, rigid, translation, and piecewise affine transforms
- Mesh-based deformable registration (diffeomorphisms)
- `imwarp` (global transform) and `meshwarp` (piecewise transform) rendering

## API
```julia
matches = blockmatch(imgA, imgB, offsetA, offsetB, params)
img, offset = meshwarp(img, mesh, offset)
tform = calculate_affine(matches)
```

## Relevance to Connectomics
Core library for registering serial EM sections using block matching and elastic mesh transforms.
