"""
Profile dataloader performance to identify bottlenecks.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connectomics.config import load_config, resolve_data_paths  # noqa: E402
from scripts.main import create_datamodule  # noqa: E402


def profile_dataloader(config_path: str, num_batches: int = 10):
    """Profile dataloader performance."""

    print("=" * 60)
    print("DATALOADER PROFILING")
    print("=" * 60)

    # Load config
    cfg = load_config(config_path)
    cfg = resolve_data_paths(cfg)
    print(f"Config: {config_path}")
    print(f"Batch size: {cfg.system.training.batch_size}")
    print(f"Num workers: {cfg.system.training.num_workers}")
    print(f"Iter num per epoch: {cfg.data.iter_num_per_epoch}")
    print()

    # Create datamodule
    print("Creating datamodule...")
    start = time.time()
    datamodule = create_datamodule(cfg)
    setup_time = time.time() - start
    print(f"✓ Setup time: {setup_time:.2f}s")
    print()

    # Get dataloader
    train_loader = datamodule.train_dataloader()
    print(f"Total batches: {len(train_loader)}")
    print()

    # Profile first epoch
    print(f"Profiling first {num_batches} batches...")
    print("-" * 60)

    batch_times = []
    total_start = time.time()

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        batch_start = time.time()

        # Simulate some processing
        image = batch["image"]
        label = batch["label"]

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        print(f"Batch {i + 1}/{num_batches}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Label shape: {label.shape}")
        print(f"  Time: {batch_time:.3f}s")
        print()

    total_time = time.time() - total_start

    # Statistics
    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average batch time: {sum(batch_times) / len(batch_times):.3f}s")
    print(f"Min batch time: {min(batch_times):.3f}s")
    print(f"Max batch time: {max(batch_times):.3f}s")
    print(f"Throughput: {num_batches / total_time:.2f} batches/sec")
    print(
        f"Samples/sec: {num_batches * cfg.system.training.batch_size / total_time:.2f}"
    )
    print()

    # Recommendations
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    avg_time = sum(batch_times) / len(batch_times)

    if avg_time > 1.0:
        print("⚠️  SLOW: Average batch time > 1s")
        print("   Recommendations:")
        print(
            f"   - Increase num_workers (current: {cfg.system.training.num_workers})"
        )
        print("   - Enable caching if possible")
        print("   - Check disk I/O performance")
        print("   - Simplify transform pipeline")
    elif avg_time > 0.5:
        print("⚠️  MODERATE: Average batch time 0.5-1.0s")
        print("   Could be improved with:")
        print(f"   - More workers (current: {cfg.system.training.num_workers})")
        print("   - Caching strategy")
    else:
        print("✓ GOOD: Average batch time < 0.5s")

    print()


if __name__ == "__main__":
    config_path = "tutorials/mito_mitoEM_common.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    profile_dataloader(config_path, num_batches=10)
