# pznet

**GitHub:** https://github.com/seung-lab/pznet
**Language:** C++ | **Stars:** 0

Efficient 3D convolutional network inference on manycore CPUs (Intel Xeon/Xeon Phi). Compiles Caffe models into optimized C++ for CPU-only inference.

## Key Features
- Compile Caffe prototxt + HDF5 weights into optimized CPU inference code
- Compile-time optimized and statically scheduled ND convnet primitives
- Intel Parallel Studio integration for multi-core/many-core performance
- Python wrapper for inference (`PZNet` class)
- Docker image available

## API
```python
from pznet.pznet import PZNet
net = PZNet('my/compiled/net')
output_patch = net.forward(np.random.rand(20, 256, 256).astype('float32'))
```

## Relevance to Connectomics
Enables fast CPU-based inference of 3D segmentation networks on EM volumes, useful for environments without GPU access.
