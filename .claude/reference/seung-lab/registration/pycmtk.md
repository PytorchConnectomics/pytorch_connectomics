# pycmtk

**GitHub:** https://github.com/seung-lab/pycmtk
**Language:** C++ | **Stars:** 2

Unofficial Python bindings for CMTK (Computational Morphometry Toolkit) functions, using pybind11 to wrap C++ code and avoid syscalls/filesystem IO.

## Key Features
- Python bindings for CMTK registration functions
- Avoids subprocess calls by directly wrapping C++
- Built with pybind11 and CMake

## Relevance to Connectomics
Provides image registration capabilities for aligning EM volume sections during reconstruction.
