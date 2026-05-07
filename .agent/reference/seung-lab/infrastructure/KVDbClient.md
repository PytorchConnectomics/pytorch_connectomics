# KVDbClient

**GitHub:** https://github.com/seung-lab/KVDbClient
**Language:** Python | **Stars:** 0

Generic key-value database client providing a unified interface for Google Cloud BigTable and Apache HBase backends. Used as the storage layer for the PyChunkedGraph.

## Key Features
- Unified API for BigTable and HBase backends
- Row-level locking for concurrency control
- Atomic unique ID generation
- Configurable column families with per-attribute serializers (NumPy, JSON, Pickle)
- Operation logging and auditing

## API
```python
from kvdbclient import get_client_class, BigTableConfig

config = BigTableConfig(PROJECT="my-project", INSTANCE="my-instance")
client = get_client_class("bigtable")("my_table", config)
```

## Relevance to Connectomics
Provides the database abstraction layer for the PyChunkedGraph, which stores dynamic neuron segmentations for proofreading.
