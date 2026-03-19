# ostd

**GitHub:** https://github.com/seung-lab/ostd
**Language:** C++ | **Stars:** 1

OSTD skeleton format based on linked paths. A compact binary format for neuron skeletons with fast header parsing and format conversion.

## Key Features
- Read/write OSTD skeleton files via osteoid library
- Memory-mapped file access for fast header queries
- CLI tool for info, conversion (SWC <-> OSTD), and viewing
- Stores cable length, component count, vertex/edge counts in header

## API
```python
import osteoid
skeleton = osteoid.load("example.ostd")
skeleton.save("example.ostd")
```

```bash
ostd info example.ostd
ostd convert example.swc example.ostd
ostd view example.ostd
```

## Relevance to Connectomics
Efficient binary skeleton format for storing and exchanging neuron morphology data from connectomics reconstructions.
