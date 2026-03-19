# cloud-files

**GitHub:** https://github.com/seung-lab/cloud-files
**Language:** Python | **Stars:** 44

Fast, threaded Python client for cloud storage (GCS, S3, local FS, HTTP). Designed for petabyte-scale access to millions of files in parallel across thousands of cores.

## Key Features
- Transparent threading and optional multi-process access
- Google Cloud Storage, AWS S3, local filesystem, and HTTP support
- gzip, brotli, bz2, zstd, xz compression
- HTTP Range reads for partial file access
- Green thread support for virtualized servers
- Exponential random-window retries for cluster robustness
- Resumable bulk transfers
- Bundled CLI tool

## API
```python
from cloudfiles import CloudFiles, CloudFile, dl

cf = CloudFiles('gs://bucket', progress=True)
results = cf.get(['file1', 'file2', ...])     # threaded batch read
cf.put('filename', content)                    # write
cf.puts([{'path': 'f', 'content': b}])        # batch write
cf['file1', 0:30]                              # range read
list(cf.list(prefix='abc'))                    # list files
cf.delete(['f1', 'f2'])                        # batch delete
```

## Relevance to Connectomics
Core I/O library for accessing large-scale EM datasets and segmentation volumes stored in cloud object storage.
