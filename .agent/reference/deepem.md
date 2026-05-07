# DeepEM Reference

**DeepEM** is a PyTorch framework for superhuman connectomics segmentation (Lee et al. 2017, Harvard/Princeton).

**Location**: `/projects/weilab/weidf/lib/pytorch_connectomics/lib/DeepEM`

## Key Architecture

- **RSUNet** (Residual Symmetric U-Net): Primary model with anisotropic convolutions
- Built on `emvision` library (Kisuk Lee)
- Pre-activation residual blocks, addition-based skip connections
- (1,2,2) downsampling for anisotropic EM data

## What PyTC Adopted from DeepEM

1. **EM augmentations**: Misalignment, missing sections, motion blur (now MONAI transforms)
2. **RSUNet architecture**: Registered as `rsunet` and `rsunet_iso`
3. **Train/val splitting**: Spatial Z-axis splitting approach
4. **Anisotropic convolution design**: Mixed (1,3,3) and (3,3,3) kernels

## What PyTC Did NOT Adopt

- Custom training loop (Lightning is superior)
- Module-based config (YAML is better for reproducibility)
- Custom data loaders (MONAI CacheDataset is more efficient)
- dataprovider3 dependency (unmaintained)

## Design Patterns Worth Knowing

- **Margin-based BCE loss**: Ignores easy examples near decision boundary
- **Iteration-based training**: Fixed iterations, not epochs
- **Multi-task output**: Dict-based multi-head architecture
- **Module-based config**: Python modules for dataset-specific logic

## References

- Lee et al. 2017: "Superhuman Accuracy on SNEMI3D" (https://arxiv.org/abs/1706.00120)
