# SimpleTasks.jl

**GitHub:** https://github.com/seung-lab/SimpleTasks.jl
**Language:** Julia | **Stars:** 1

Julia framework for creating and operating on cloud-distributed tasks. Provides queue-based task scheduling (AWS SQS) and bucket-based data I/O (AWS S3, GCS) for parallelizing computation in the cloud.

## Key Features
- DaemonTask interface (prepare/execute/finalize lifecycle)
- AWS SQS queue service for task scheduling
- AWS S3 and GCS bucket services for data I/O
- File system caching layer
- JSON-serializable task definitions

## Relevance to Connectomics
Generic task distribution framework used to parallelize EM volume processing (watershed, agglomeration, meshing) across cloud clusters.
