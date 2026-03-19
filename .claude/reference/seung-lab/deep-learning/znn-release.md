# znn-release

**GitHub:** https://github.com/seung-lab/znn-release
**Language:** C++ | **Stars:** 94 | **Archived**

Multi-core CPU implementation of deep learning for 2D and 3D sliding window convolutional networks. Uses FFT-based convolutions, which are advantageous for 3D kernels >= 5x5x5. Currently deprecated in favor of GPU-based frameworks.

## Key Features
- FFT-based convolutions for efficient large-kernel 3D ConvNets
- Multi-core CPU parallelism (no GPU required)
- Dense output / sliding window inference
- Python interface for network training and inference

## Publications
- Zlateski, Lee & Seung (2015) -- ZNN: Fast and Scalable Training on Multi-Core
- Lee, Zlateski, Vishwanathan & Seung (2015) -- Recursive Training for Neuronal Boundary Detection

## Relevance to Connectomics
Historical predecessor to modern GPU-based EM segmentation networks; pioneered dense 3D ConvNet inference for neuronal boundary prediction.
