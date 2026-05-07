# pyzfp

**GitHub:** https://github.com/seung-lab/pyzfp
**Language:** Cython | **Stars:** 0

Python wrapper over the ZFP floating-point compression library. Cython-based rewrite for better performance than the original ctypes version. Wraps zfp version 0.5.5.

## Key Features
- Lossy floating-point array compression via ZFP
- Configurable tolerance-based compression
- Parallel compression support
- Cython-based for performance

## API
```python
from pyzfp import compress, decompress
import numpy as np

a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))
compressed = compress(a, tolerance=1e-7, parallel=True)
recovered = decompress(compressed, a.shape, a.dtype, tolerance=1e-7)
```

## Relevance to Connectomics
Enables lossy compression of floating-point volumetric data (e.g., affinity maps, displacement fields) for storage and transfer efficiency.
