# microviewer

**GitHub:** https://github.com/seung-lab/microviewer
**Language:** JavaScript | **Stars:** 16

Multiplatform browser-based 3D numpy image viewer with segmentation overlay and voxel painting support.

## Key Features
- 3-axis visualization of 3D images (grayscale, color, floating point, boolean, segmentation)
- Interactive segmentation overlay with brush selection tools
- Direct voxel painting with undo/redo
- Save segmentation as .npy or .ckl (crackle) format
- 3D object visualization: meshes, skeletons (osteoid), point clouds, bounding boxes
- Supports .npy, .ckl, .nrrd, .nii formats

## API
```python
from microviewer import view, hyperview, objects
view(numpy_image)                    # grayscale/color
view(numpy_image, seg=True)          # segmentation labels
hyperview(image, labels)             # interactive overlay
objects([mesh, skeleton, bbox])      # 3D object visualization
```

## Relevance to Connectomics
Quick browser-based inspection of EM image volumes, segmentation labels, and reconstructed neuron meshes/skeletons without requiring heavy visualization tools.
