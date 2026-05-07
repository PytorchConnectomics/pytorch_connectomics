# PyTorch EMVision Reference

**EMVision** is a collection of RSUNet variants for 3D EM segmentation by Kisuk Lee (MIT).

**Location**: `/projects/weilab/weidf/lib/seg/pytorch-emvision`

## Core Models

| Model | Feature |
|-------|---------|
| `RSUNet` | Base model, BN+ReLU, anisotropic (1,2,2) downsampling |
| `isoRSUNet` | Isotropic (2,2,2) downsampling |
| `rsunet_gn` | Group Normalization (better for small batches) |
| `rsunet_act` | Configurable activation (ReLU/LeakyReLU/PReLU/ELU) |
| `rsunet_2d3d` | 2D conv at shallow layers, 3D at deeper layers |
| `dynamic_rsunet` | Recurrent training with multiple BN layers |

## Design Principles

1. **Anisotropic convolutions**: (1,3,3) kernels respect EM data resolution differences
2. **Pre-activation residual blocks**: BN->ReLU->Conv for better gradient flow
3. **Addition-based skip connections**: More efficient than concatenation
4. **BilinearUp**: Caffe-style bilinear upsampling (no checkerboard artifacts)
5. **Kaiming initialization**: He init for ReLU networks

## Integration with PyTC

PyTC's `rsunet` and `rsunet_iso` architectures are based on EMVision's design but reimplemented in PyTC's model framework. Key file: `connectomics/models/arch/rsunet.py`.
