# data-cube-x

**GitHub:** https://github.com/seung-lab/data-cube-x
**Language:** JavaScript | **Stars:** 9

Hackable 3D volumetric image library for the web. Pure JavaScript CPU rendering engine used in EyeWire.org for EM visualization. Renders axial slices of 3D volumes to canvas with segmentation overlay support.

## Key Features
- Pure JavaScript 3D image stack representation (no WebGL required)
- Fast CPU rendering of axial slices (up to 55 FPS at 1024x1024)
- Segmentation overlay via `Volume` object (image + label layers)
- Supports uint8, uint16, uint32 data types
- Insert images/canvases into volumetric stack at arbitrary offsets

## API
```javascript
var dc = new DataCube({ bytes: 1, size: { x: 256, y: 256, z: 256 } });
dc.insertImage(img, 0, 0, 0);
var slice = dc.slice('z', 128);
dc.renderGrayImageSlice(context, 'z', 128);
```

## Relevance to Connectomics
Provides lightweight browser-based EM image volume viewing, the rendering core behind the EyeWire citizen science neuron tracing platform.
