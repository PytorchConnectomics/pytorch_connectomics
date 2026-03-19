# shard-computer

**GitHub:** https://github.com/seung-lab/shard-computer
**Language:** C++ | **Stars:** 4

Accelerated shard hash computation for Neuroglancer Precomputed shards using MurmurHash3.

## Key Features
- Compute shard number for a single label
- Compute unique shard numbers from a numpy array of labels
- Assign labels to shards (returns shard-to-labels mapping)

## API
```python
import shardcomputer
shard_no = shardcomputer.shard_number(label, preshift_bits, shard_bits, minishard_bits)
shard_set = shardcomputer.unique_shard_numbers(labels, preshift_bits, shard_bits, minishard_bits)
mapping = shardcomputer.assign_labels_to_shards(labels, preshift_bits, shard_bits, minishard_bits)
```

## Relevance to Connectomics
Enables efficient Neuroglancer Precomputed shard organization for serving large-scale EM segmentation volumes.
