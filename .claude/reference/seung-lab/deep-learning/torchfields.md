# torchfields

**GitHub:** https://github.com/seung-lab/torchfields
**Language:** Python | **Stars:** 51

A PyTorch add-on for working with image mappings and displacement fields, including Spatial Transformers. Provides displacement field abstraction that encapsulates functionality for Spatial Transformer Networks and Optical Flow estimation.

## Key Features
- Displacement fields as first-class PyTorch tensors (inherits from `torch.Tensor`)
- Eulerian (pull) and Lagrangian (push) warping conventions
- Field composition, inversion, and sampling
- Pixel and half-image unit conversions
- Installs as `torch.Field` via monkey patching

## API
```python
import torchfields
field = torch.Field.identity(shape)
warped = field.sample(image)          # warp image using field
composed = field1(field2)             # compose fields
inv_field = field.inverse()           # invert field
pixel_field = field.pixels()          # convert to pixel units
```

## Relevance to Connectomics
Used for image alignment/registration of EM serial sections and applying spatial transformations during volume reconstruction.
