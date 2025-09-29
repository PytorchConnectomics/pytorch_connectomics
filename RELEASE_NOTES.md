# PyTorch Connectomics Release Notes

## Version 1.0 - Latest Development Build

*Release Date: September 2025*

### 🚀 Major Features

#### Tensorstore Integration
- **NEW**: Added support for Google Tensorstore for scalable data I/O
- Enhanced data loading pipeline with tensorstore backend for handling large-scale neuromorphic datasets
- Improved memory efficiency for processing high-resolution EM data volumes
- Files: `connectomics/engine/trainer.py`, `connectomics/data/utils/data_io.py`

#### Skeleton-aware Processing
- **NEW**: Added skeleton magnitude prediction capabilities
- Enhanced curvilinear structure analysis tools
- Improved skeletonization workflow for neuron reconstruction
- Files: `connectomics/data/utils/data_segmentation.py`, `connectomics/utils/visualizer.py`

#### Advanced Instance Segmentation
- **ENHANCED**: Updated polarity-to-instance conversion with CC3D (Connected Components 3D)
- Improved memory handling for large-scale instance segmentation tasks
- Better connected component analysis for complex neuronal structures
- Files: `connectomics/utils/process.py`

### 🔧 Improvements & Bug Fixes

#### Distance Transform Enhancements
- Fixed background value handling in distance transform computations (`bg=-1`)
- Added erosion support to distance transform pipeline
- Improved skeleton-aware distance transform accuracy
- Enhanced numerical stability for distance-based loss functions

#### Training & Inference Stability
- **FIXED**: Ensured inference block results are non-negative
- **FIXED**: Resolved erosion option bugs in data preprocessing
- **FIXED**: Visualization improvements for multi-target outputs (`topt=5`)
- **FIXED**: Removed negative artifacts in output saving pipeline

#### Model Architecture Updates
- Added pretrained model dictionary validation
- Enhanced test-time augmentation support (`num_aug` parameter flexibility)
- Improved backward compatibility for single inference mode
- Better support for multi-channel input processing in VolumeDataset

#### Configuration & Dataset Management
- **UPDATED**: Enhanced YACS configuration validation
- **UPDATED**: Improved dataset building with better error handling
- **UPDATED**: Streamlined configuration files for SNEMI, MitoEM, and NucMM datasets
- **UPDATED**: Better support for distributed training setups

### 📊 Benchmark & Documentation Updates

#### Benchmark Notebooks
- **NEW**: Added SNEMI benchmark notebooks with Colab integration
- **NEW**: Added MitoEM benchmark notebooks for mitochondria segmentation
- Migrated existing benchmarks to Google Colab for better accessibility
- Updated tutorial notebooks with latest API changes

#### Documentation Improvements
- **UPDATED**: Comprehensive documentation overhaul
- **UPDATED**: Improved installation instructions and dependency management
- **UPDATED**: Enhanced API documentation with clearer examples
- **UPDATED**: Updated neuron segmentation tutorials

### 🔄 Data Pipeline Enhancements

#### Augmentation & Preprocessing
- Enhanced valid mask sampling for improved training data quality
- Added configurable rejection sampling trials
- Improved memory management for large volume processing
- Better handling of multi-scale data inputs

#### I/O & Format Support
- Enhanced support for various EM data formats
- Improved HDF5 and TIFF stack handling
- Better integration with cloud storage solutions
- Optimized data loading for distributed training

### 🛠️ Development & Testing

#### Code Quality
- **UPDATED**: Cython dependency bumped to 0.29.22 for Python 3.10 compatibility
- **FIXED**: Resolved PyGen_Send compatibility issues
- Enhanced error handling and logging throughout the codebase
- Improved code formatting and documentation standards

#### Testing Infrastructure
- Enhanced test coverage for model blocks and loss functions
- Improved augmentation testing suite
- Better integration testing for end-to-end workflows
- Updated test configurations for new features

### 📦 Dependencies & Compatibility

#### Updated Requirements
- Cython >= 0.29.22 (Python 3.10 compatibility)
- Enhanced MONAI integration (>= 0.9.1)
- Improved PyTorch compatibility
- Better support for latest CUDA versions

#### System Requirements
- Python 3.8+ support maintained
- PyTorch 1.8+ compatibility
- Enhanced GPU memory management
- Improved distributed training support

### 🔧 Configuration Changes

#### New Configuration Options
- Tensorstore backend configuration
- Enhanced distance transform options
- Improved augmentation parameters
- Better multi-GPU training settings

#### Deprecated Features
- Legacy distance transform implementations
- Old-style configuration formats
- Deprecated augmentation interfaces

### 🚨 Breaking Changes

- **IMPORTANT**: Distance transform background value changed to `-1`
- **IMPORTANT**: Some legacy configuration options removed
- **IMPORTANT**: Updated API for skeleton-aware processing
- **IMPORTANT**: Modified polarity-to-instance conversion interface

### 🔮 Coming Soon

- Enhanced 3D visualization tools
- Improved multi-task learning capabilities
- Better integration with neuromorphic data standards
- Advanced active learning features

---

### 📝 Technical Details

**Total Changes**: 295 insertions, 231 deletions across 24 files
**Key Contributors**: donglaiw, zengyuy, linok-bc, jasonkena
**Testing**: All existing tests pass with new functionality
**Documentation**: Comprehensive updates to user guides and API documentation

### 🙏 Acknowledgments

Special thanks to the Harvard Visual Computing Group and all contributors who made this release possible. The framework continues to benefit from the broader connectomics and computer vision communities.

For detailed technical specifications and usage examples, please refer to the updated documentation at [connectomics.readthedocs.io](https://connectomics.readthedocs.io).

---

*For support and questions, join our [Slack community](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w) or check the [GitHub issues](https://github.com/zudi-lin/pytorch_connectomics/issues).*