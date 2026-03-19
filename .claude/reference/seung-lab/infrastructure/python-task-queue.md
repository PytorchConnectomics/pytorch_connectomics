# python-task-queue

**GitHub:** https://github.com/seung-lab/python-task-queue
**Language:** Python | **Stars:** 39

Asynchronous serverless task queue with timed leasing. Supports AWS SQS and local filesystem queues for distributed, dependency-free task execution.

## Key Features
- SQS-backed cloud queue and filesystem-based local queue
- Timed task leasing (auto-recycle on timeout)
- `@queueable` decorator for simple function-based tasks
- `RegisteredTask` class for complex task definitions
- JSON messaging format
- Multi-process local execution via `LocalTaskQueue`
- Green thread support

## API
```python
from taskqueue import queueable, LocalTaskQueue, TaskQueue
from functools import partial

@queueable
def my_task(x):
    process(x)

# Local execution
tq = LocalTaskQueue(parallel=5)
tq.insert_all(partial(my_task, i) for i in range(1000))

# Cloud/filesystem queue
tq = TaskQueue("fq:///path/to/queue")  # or "sqs://queue-name"
tq.insert(partial(my_task, 42))
```

## Installation
```bash
pip install task-queue
```

## Relevance to Connectomics
The task orchestration backbone for Igneous and other seung-lab pipelines; manages distributed execution of meshing, downsampling, skeletonization, and other large-scale processing jobs.
