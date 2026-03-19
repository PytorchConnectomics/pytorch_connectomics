# mapbuffer

**GitHub:** https://github.com/seung-lab/mapbuffer
**Language:** Python | **Stars:** 10

Serializable map of integers to bytes with near-zero parsing. Designed for fast random access to individual items in large serialized dictionaries without deserializing the entire structure.

## Key Features
- Serialize dict of int->bytes with minimal parse overhead
- Eytzinger binary search for fast index lookup
- Optional gzip/brotli compression per item
- CRC32c integrity checking
- IntMap specialization for u64->u64 mapping
- mmap support for memory-efficient file access

## API
```python
from mapbuffer import MapBuffer, IntMap

mb = MapBuffer({2848: b'abc', 12939: b'123'})
binary = mb.tobytes()
mb = MapBuffer(binary)
print(mb[2848])  # b'abc'

im = IntMap({1: 2, 3: 4})
print(im[1])  # 2
```

## Relevance to Connectomics
Used for efficient serialization of skeleton fragments and mesh data in the connectomics reconstruction pipeline, replacing slow pickle-based approaches.
