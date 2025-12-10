#!/usr/bin/env python3
"""
CellMap training script using PyTC's Lightning framework.

This script provides a thin wrapper that:
1. Creates CellMap dataloaders using cellmap-data package
2. Wraps them in PyTC's Lightning DataModule interface
3. Reuses all PyTC training infrastructure (model building, checkpointing, logging)

Usage:
    python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml
    python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_mito.yaml

Requirements:
    pip install cellmap-data cellmap-segmentation-challenge
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
PYTC_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PYTC_ROOT))

import torch
import pytorch_lightning as pl

from connectomics.config import Config
from connectomics.training.lit import (
    ConnectomicsModule,
    cleanup_run_directory,
    create_trainer,
    modify_checkpoint_state,
    parse_args,
    setup_config,
    setup_run_directory,
    setup_seed_everything,
)

# CellMap data loading (official)
try:
    from cellmap_segmentation_challenge.utils import get_dataloader, make_datasplit_csv
except ImportError:
    print("❌ Error: cellmap-data not installed")
    print("   Please run: pip install cellmap-data cellmap-segmentation-challenge")
    sys.exit(1)

# Setup seed_everything with version fallback
seed_everything = setup_seed_everything()


class CellMapDataModule(pl.LightningDataModule):
    """
    Lightning DataModule wrapper for CellMap dataloaders.

    This class bridges CellMap's get_dataloader() with PyTC's Lightning framework.
    """

    class _KeyMappingLoader:
        """Adapter to rename CellMap batch keys to PyTC conventions."""

        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for batch in self.loader:
                yield self._map_batch(batch)

        def __len__(self):
            return len(self.loader)

        @property
        def dataset(self):
            return getattr(self.loader, "dataset", None)

        @property
        def batch_size(self):
            return getattr(self.loader, "batch_size", None)

        def _map_batch(self, batch):
            mapped = {}
            if "input" in batch:
                mapped["image"] = batch["input"]
            if "output" in batch:
                label = batch["output"]
                # Replace any NaNs/infs coming from upstream data transforms
                if torch.isnan(label).any() or torch.isinf(label).any():
                    label = torch.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
                mapped["label"] = label.clamp_(0.0, 1.0)
            for key, value in batch.items():
                if key in {"input", "output"}:
                    continue
                if key == "__metadata__":
                    mapped["metadata"] = value
                else:
                    mapped[key] = value
            return mapped

    def __init__(
        self,
        cfg: Config,
        mode: str = "train",
    ):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare_data(self):
        """Prepare data (download, generate datasplit, etc.)"""
        cellmap_cfg = self.cfg.data.cellmap

        # Ensure datasplit exists
        datasplit_path = Path(cellmap_cfg["datasplit_path"])
        if not datasplit_path.exists():
            print(f"🔧 Generating datasplit: {datasplit_path}")
            datasplit_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract scale from input_array_info
            scale = cellmap_cfg["input_array_info"]["scale"]

            # Build search path from data_root
            data_root = cellmap_cfg["data_root"]
            search_path = f"{data_root}/{{dataset}}/{{dataset}}.zarr/recon-1/{{name}}"

            # Use CellMap's official datasplit generator
            make_datasplit_csv(
                csv_path=str(datasplit_path),
                classes=cellmap_cfg["classes"],
                scale=scale,
                force_all_classes=cellmap_cfg["force_all_classes"],
                search_path=search_path,
            )
            print(f"✅ Datasplit generated: {datasplit_path}")

    @staticmethod
    def _unwrap_loader(loader):
        """Return the underlying PyTorch DataLoader if wrapped by CellMapDataLoader."""
        if loader is None:
            return None
        return getattr(loader, "loader", loader)

    def setup(self, stage: str = None):
        """Setup train/val/test dataloaders"""
        cellmap_cfg = self.cfg.data.cellmap

        # Get system config based on mode
        if stage == "fit" or stage is None:
            system_cfg = self.cfg.system.training
        else:
            system_cfg = self.cfg.system.inference

        # Use CUDA only when there are no multiprocessing workers (safe on main process);
        # otherwise force CPU to avoid CUDA init in forked workers.
        dataloader_device = (
            "cuda"
            if torch.cuda.is_available()
            and system_cfg.num_gpus > 0
            and system_cfg.num_workers == 0
            else "cpu"
        )

        # Get absolute path to datasplit
        from pathlib import Path as PathLib
        csv_path_abs = str(PathLib(cellmap_cfg["datasplit_path"]).absolute())
        print(f"📂 Loading datasplit from: {csv_path_abs}")

        # Common dataloader kwargs
        dataloader_kwargs = {
            "batch_size": system_cfg.batch_size,
            "datasplit_path": csv_path_abs,
            "classes": cellmap_cfg["classes"],
            "input_array_info": cellmap_cfg["input_array_info"],
            "target_array_info": cellmap_cfg["target_array_info"],
            "num_workers": system_cfg.num_workers,
            "device": dataloader_device,
        }

        if stage == "fit" or stage is None:
            print("📦 Creating CellMap train/val dataloaders...")
            train_loader, val_loader = get_dataloader(
                **dataloader_kwargs,
                spatial_transforms=cellmap_cfg["spatial_transforms"],
                iterations_per_epoch=self.cfg.data.iter_num_per_epoch,
            )
            self.train_loader = self._KeyMappingLoader(self._unwrap_loader(train_loader))
            self.val_loader = (
                self._KeyMappingLoader(self._unwrap_loader(val_loader))
                if val_loader is not None
                else None
            )
            if self.train_loader is not None:
                print(f"  Train batches per epoch: {len(self.train_loader)}")
            if self.val_loader is not None:
                print(f"  Val batches: {len(self.val_loader)}")

        if stage == "test":
            print("📦 Creating CellMap test dataloader...")
            # Note: For CellMap challenge submission, you'd need test crops
            # This is a placeholder for when test data has labels
            if hasattr(self.cfg, "test") and self.cfg.test.data.test_image:
                test_loader, _ = get_dataloader(
                    **dataloader_kwargs,
                    spatial_transforms=None,
                    iterations_per_epoch=self.cfg.data.iter_num_per_epoch,
                )
                self.test_loader = self._KeyMappingLoader(self._unwrap_loader(test_loader))
                if self.test_loader is not None:
                    print(f"  Test batches: {len(self.test_loader)}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def main():
    """Main training function (reuses main.py logic)"""
    # Parse arguments (same as main.py)
    args = parse_args()

    # Validate config is provided
    if not args.config:
        print("❌ Error: --config is required")
        print("\nUsage:")
        print("  python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_cos7.yaml")
        sys.exit(1)

    # Setup config (same as main.py)
    print("\n" + "=" * 60)
    print("🚀 CellMap Training with PyTC Lightning Framework")
    print("=" * 60)
    cfg = setup_config(args)

    # Validate CellMap config
    if not hasattr(cfg.data, "cellmap"):
        print("❌ Error: Config must have data.cellmap section")
        print("   See tutorials/cellmap_cos7.yaml for example")
        sys.exit(1)

    # Setup run directory (same as main.py)
    dirpath = cfg.monitor.checkpoint.dirpath
    run_dir = setup_run_directory(args.mode, cfg, dirpath)
    output_base = run_dir.parent

    # Set random seed (same as main.py)
    if cfg.system.seed is not None:
        print(f"🎲 Random seed set to: {cfg.system.seed}")
        seed_everything(cfg.system.seed, workers=True)

    # Create model (same as main.py)
    print(f"Creating model: {cfg.model.architecture}")
    model = ConnectomicsModule(cfg)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    # Handle checkpoint (same as main.py)
    ckpt_path = modify_checkpoint_state(
        args.checkpoint,
        run_dir,
        reset_optimizer=args.reset_optimizer,
        reset_scheduler=args.reset_scheduler,
        reset_epoch=args.reset_epoch,
        reset_early_stopping=args.reset_early_stopping,
    )

    # Create trainer (same as main.py)
    trainer = create_trainer(
        cfg,
        run_dir=run_dir,
        fast_dev_run=args.fast_dev_run,
        ckpt_path=ckpt_path,
        mode=args.mode,
    )

    # Create CellMap datamodule (custom for CellMap)
    datamodule = CellMapDataModule(cfg, mode=args.mode)

    # Training/testing workflow (same as main.py)
    try:
        if args.mode == "train":
            print("\n" + "=" * 60)
            print("🏃 STARTING TRAINING")
            print("=" * 60)

            trainer.fit(
                model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )
            print("\n✅ Training completed successfully!")

        elif args.mode == "test":
            print("\n" + "=" * 60)
            print("🧪 RUNNING TEST")
            print("=" * 60)

            trainer.test(
                model,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )

    except Exception as e:
        mode_name = args.mode.capitalize() if args.mode else "Operation"
        print(f"\n❌ {mode_name} failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup (same as main.py)
        if args.mode == "train":
            cleanup_run_directory(output_base)


if __name__ == "__main__":
    main()
