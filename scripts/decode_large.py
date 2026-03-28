"""Large-volume parallel waterz decoding using file-backed orchestrator.

Usage:
    # Serial (single process, all stages)
    python scripts/decode_large.py --config tutorials/waterz_decoding_large.yaml

    # Initialize workflow only (for parallel launch)
    python scripts/decode_large.py --config tutorials/waterz_decoding_large.yaml --init-only

    # Run as a worker (claims tasks from shared workflow dir)
    python scripts/decode_large.py --config tutorials/waterz_decoding_large.yaml --worker

    # Wait for all workers to finish, then assemble output
    python scripts/decode_large.py --config tutorials/waterz_decoding_large.yaml --wait --assemble
"""

import argparse
import os
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Large-volume waterz decoding")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--init-only", action="store_true", help="Initialize workflow and exit")
    parser.add_argument("--worker", action="store_true", help="Run as a worker (claim tasks)")
    parser.add_argument("--wait", action="store_true", help="Wait for all tasks to complete")
    parser.add_argument("--assemble", action="store_true", help="Assemble final output volume")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Run N worker processes on this machine (e.g. --parallel 8)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks per worker")
    parser.add_argument("--idle-timeout", type=float, default=60.0, help="Worker idle timeout (seconds)")
    parser.add_argument("--worker-id", type=str, default=None, help="Worker identifier")
    parser.add_argument("--job-id", type=str, default=None, help="SLURM job ID")
    # Allow CLI overrides in key=value format
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    large_cfg = cfg.get("large_decode", {})

    # Apply CLI overrides
    for override in args.overrides:
        if "=" not in override:
            print(f"Warning: skipping invalid override '{override}' (expected key=value)")
            continue
        key, value = override.split("=", 1)
        # Try numeric conversion
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        large_cfg[key] = value

    if not large_cfg.get("affinity_path"):
        print("Error: large_decode.affinity_path is required")
        sys.exit(1)
    if not large_cfg.get("workflow_root"):
        print("Error: large_decode.workflow_root is required")
        sys.exit(1)

    os.environ.setdefault("CCACHE_DISABLE", "1")

    from waterz import LargeDecodeRunner

    # Parse config
    chunk_shape = large_cfg.get("chunk_shape", [256, 512, 512])
    if isinstance(chunk_shape, list):
        chunk_shape = tuple(chunk_shape)

    thresholds = large_cfg.get("thresholds", [0.5])
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds]

    runner = LargeDecodeRunner.create(
        affinity_path=large_cfg["affinity_path"],
        workflow_root=large_cfg["workflow_root"],
        chunk_shape=chunk_shape,
        thresholds=thresholds,
        merge_function=large_cfg.get("merge_function", "aff85_his256"),
        aff_threshold_low=float(large_cfg.get("aff_threshold_low", 0.1)),
        aff_threshold_high=float(large_cfg.get("aff_threshold_high", 0.999)),
        channel_order=large_cfg.get("channel_order", "xyz"),
        write_output=bool(large_cfg.get("write_output", True)),
        output_path=large_cfg.get("output_path") or None,
        border_min_overlap=int(large_cfg.get("border_min_overlap", 1)),
        border_one_sided_threshold=float(large_cfg.get("border_one_sided_threshold", 0.9)),
        border_iou_threshold=float(large_cfg.get("border_iou_threshold", 0.0)),
        border_affinity_threshold=float(large_cfg.get("border_affinity_threshold", 0.0)),
        compression=large_cfg.get("compression", "gzip"),
        compression_level=int(large_cfg.get("compression_level", 4)),
    )

    chunks = runner.chunks
    borders = runner.borders
    print(f"Volume shape: {runner.config.volume_shape}")
    print(f"Chunk shape:  {runner.config.chunk_shape}")
    print(f"Chunks:       {len(chunks)}")
    print(f"Borders:      {len(borders)}")
    print(f"Workflow:     {runner.config.workflow_root}")

    if args.init_only:
        print("Workflow initialized. Launch workers to execute tasks.")
        return

    if args.worker:
        worker_id = args.worker_id or os.environ.get("SLURM_JOB_ID", None)
        job_id = args.job_id or os.environ.get("SLURM_ARRAY_TASK_ID", None)
        print(f"Starting worker: {worker_id or 'auto'} (job={job_id or 'none'})")
        n = runner.run_worker(
            worker_id=worker_id,
            max_tasks=args.max_tasks,
            idle_timeout=args.idle_timeout,
            job_id=job_id,
        )
        print(f"Worker completed {n} tasks.")
        return

    if args.wait:
        print("Waiting for all tasks to complete...")
        runner.wait(timeout=None)
        print("All tasks completed.")
        if args.assemble and runner.config.write_output:
            print("Assembling output...")
            runner.handle_assemble_output(None)
            print(f"Output: {runner.config.resolved_output_path}")
        return

    if args.parallel and args.parallel > 1:
        # Multi-process on one machine
        import multiprocessing as mp

        n_workers = args.parallel
        print(f"Running parallel decode with {n_workers} workers...")

        def _worker_fn(worker_idx):
            os.environ["CCACHE_DISABLE"] = "1"
            # Each process loads its own runner from disk
            from waterz import LargeDecodeRunner as _LDR
            w = _LDR.load(large_cfg["workflow_root"])
            return w.run_worker(
                worker_id=f"local-{worker_idx}",
                idle_timeout=args.idle_timeout or 120,
                max_tasks=args.max_tasks,
            )

        with mp.Pool(n_workers) as pool:
            counts = pool.map(_worker_fn, range(n_workers))
        n = sum(counts)
        print(f"Completed {n} tasks across {n_workers} workers.")
    else:
        # Default: run serial (all stages in one process)
        print("Running serial decode...")
        n = runner.run_serial()
        print(f"Completed {n} tasks.")

    status = runner.orchestrator.stage_counts()
    for stage, counts in sorted(status.items()):
        print(f"  {stage}: {counts}")


if __name__ == "__main__":
    main()
