"""
Validation Reseeding Callback for PyTorch Lightning.

This callback ensures validation datasets are properly reseeded at the start
of each validation epoch to prevent frozen validation losses caused by
sampling identical patches across epochs.

The callback handles:
- Single or multiple validation dataloaders
- Wrapped datasets (Subset, ConcatDataset, etc.)
- DDP training (logs only on rank 0)
- Sanity check validation (logs but doesn't interfere)
- Datasets without set_epoch() method (gracefully skipped)
"""

from __future__ import annotations
import logging
from typing import Any, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
try:
    from torch.utils.data import ChainDataset
    HAS_CHAIN_DATASET = True
except ImportError:
    HAS_CHAIN_DATASET = False

logger = logging.getLogger(__name__)


class ValidationReseedingCallback(Callback):
    """
    Callback to reseed validation datasets at the start of each validation epoch.
    
    This ensures validation samples different patches each epoch while maintaining
    determinism, preventing the model from memorizing fixed validation patches.
    
    Args:
        base_seed: Base random seed (typically from cfg.system.seed)
        log_fingerprint: If True, log a sampling fingerprint for verification
        log_all_ranks: If True, log from all DDP ranks (otherwise rank 0 only)
        verbose: If True, log detailed information about dataset reseeding
    
    Example:
        >>> callback = ValidationReseedingCallback(base_seed=42, verbose=True)
        >>> trainer = pl.Trainer(callbacks=[callback])
        >>> trainer.fit(model, datamodule=datamodule)
    """
    
    def __init__(
        self,
        base_seed: int = 0,
        log_fingerprint: bool = True,
        log_all_ranks: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.base_seed = base_seed
        self.log_fingerprint = log_fingerprint
        self.log_all_ranks = log_all_ranks
        self.verbose = verbose
        
        # Track reseeding statistics
        self._reseeded_count = 0
        self._skipped_count = 0
        self._last_epoch = -1
    
    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Called at the start of each validation epoch.
        
        This is the CORRECT place to reseed validation datasets, as it runs
        before each validation epoch (unlike val_dataloader which is called once).
        """
        # Only log on rank 0 unless log_all_ranks is True
        if not self.log_all_ranks and trainer.global_rank != 0:
            return
        
        current_epoch = trainer.current_epoch
        global_step = trainer.global_step
        is_sanity_check = trainer.sanity_checking
        
        # Determine if this is a new epoch (not just sanity check repeat)
        if current_epoch == self._last_epoch and not is_sanity_check:
            return  # Already processed this epoch
        
        self._last_epoch = current_epoch
        
        # Log validation epoch start
        epoch_type = "SANITY CHECK" if is_sanity_check else f"EPOCH {current_epoch}"
        if self.verbose:
            logger.info(
                "[VAL RESEED] %s | Step %s | Rank %s",
                epoch_type,
                global_step,
                trainer.global_rank,
            )
        
        # Get validation dataloaders
        val_dataloaders = self._get_validation_dataloaders(trainer)
        
        if not val_dataloaders:
            logger.warning(
                "[VAL RESEED SKIPPED] %s | Reason: No validation dataloaders found",
                epoch_type,
            )
            return
        
        # Reseed each dataloader's dataset(s)
        total_reseeded = 0
        total_skipped = 0
        skipped_reasons = []
        
        for dl_idx, dataloader in enumerate(val_dataloaders):
            if self.verbose:
                logger.info("[VAL RESEED] Processing DataLoader %s", dl_idx)
            
            # Find all datasets in this dataloader
            datasets = self._find_all_datasets(dataloader)
            
            for ds_idx, dataset in enumerate(datasets):
                dataset_info = f"{type(dataset).__name__}@{id(dataset)}"
                
                # Check if dataset supports set_epoch
                if hasattr(dataset, 'set_epoch'):
                    try:
                        # Reseed: base_seed + current_epoch
                        # For sanity check, use negative epoch to avoid interfering with real training
                        seed_epoch = -1 if is_sanity_check else current_epoch
                        dataset.set_epoch(seed_epoch, self.base_seed)
                        
                        if self.verbose:
                            logger.info("[VAL RESEED]  Dataset %s: %s", ds_idx, dataset_info)
                            logger.info(
                                "[VAL RESEED]    set_epoch(epoch=%s, base_seed=%s)",
                                seed_epoch,
                                self.base_seed,
                            )
                            logger.info(
                                "[VAL RESEED]    Effective seed: %s",
                                self.base_seed + seed_epoch,
                            )
                        
                        total_reseeded += 1
                        
                        # Log fingerprint if requested and dataset supports it
                        if self.log_fingerprint and hasattr(dataset, 'get_sampling_fingerprint'):
                            fingerprint = dataset.get_sampling_fingerprint()
                            logger.info("[VAL RESEED]    Fingerprint: %s", fingerprint)
                        
                    except Exception as e:
                        logger.warning(
                            "[VAL RESEED SKIPPED] Dataset %s: %s | Reason: "
                            "Exception during set_epoch: %s",
                            ds_idx,
                            dataset_info,
                            e,
                        )
                        total_skipped += 1
                        skipped_reasons.append(f"Exception: {e}")
                else:
                    logger.warning(
                        "[VAL RESEED SKIPPED] Dataset %s: %s | Reason: no set_epoch() method",
                        ds_idx,
                        dataset_info,
                    )
                    total_skipped += 1
                    skipped_reasons.append("no set_epoch() method")
        
        # Update statistics
        self._reseeded_count += total_reseeded
        self._skipped_count += total_skipped
        
        # Summary
        logger.info("[VAL RESEED] Summary for %s:", epoch_type)
        logger.info("[VAL RESEED]   Datasets reseeded: %s", total_reseeded)
        logger.info("[VAL RESEED]   Datasets skipped:  %s", total_skipped)
        logger.info("[VAL RESEED]   Total dataloaders: %s", len(val_dataloaders))
        if skipped_reasons:
            logger.info("[VAL RESEED]   Skip reasons: %s", ", ".join(set(skipped_reasons)))
    
    def _get_validation_dataloaders(
        self,
        trainer: pl.Trainer,
    ) -> List[DataLoader]:
        """
        Get all validation dataloaders from the trainer.
        
        Handles:
        - Single dataloader
        - List of dataloaders
        - CombinedLoader (Lightning 2.0+)
        """
        val_dataloaders = trainer.val_dataloaders
        
        if val_dataloaders is None:
            return []
        
        # Handle different return types
        if isinstance(val_dataloaders, DataLoader):
            return [val_dataloaders]
        elif isinstance(val_dataloaders, list):
            return val_dataloaders
        else:
            # CombinedLoader or other wrapper
            # Try to extract dataloaders
            if hasattr(val_dataloaders, 'loaders'):
                # CombinedLoader has .loaders attribute
                loaders = val_dataloaders.loaders
                if isinstance(loaders, dict):
                    return list(loaders.values())
                elif isinstance(loaders, list):
                    return loaders
            
            # Fallback: treat as single dataloader
            return [val_dataloaders]
    
    def _find_all_datasets(
        self,
        dataloader: DataLoader,
    ) -> List[Dataset]:
        """
        Recursively find all "real" datasets in a dataloader.
        
        Handles:
        - Direct dataset
        - Subset wrapping a dataset
        - ConcatDataset containing multiple datasets
        - ChainDataset containing multiple datasets
        - Nested wrappers (Subset(ConcatDataset(...)))
        """
        datasets = []
        
        if not hasattr(dataloader, 'dataset'):
            return datasets
        
        dataset = dataloader.dataset
        self._extract_datasets_recursive(dataset, datasets)
        
        return datasets
    
    def _extract_datasets_recursive(
        self,
        dataset: Any,
        result: List[Dataset],
    ) -> None:
        """Recursively extract all leaf datasets."""
        
        # Handle Subset
        if isinstance(dataset, Subset):
            # Subset wraps another dataset
            self._extract_datasets_recursive(dataset.dataset, result)
            return
        
        # Handle ConcatDataset
        if isinstance(dataset, ConcatDataset):
            # ConcatDataset contains multiple datasets
            for sub_dataset in dataset.datasets:
                self._extract_datasets_recursive(sub_dataset, result)
            return
        
        # Handle ChainDataset
        if HAS_CHAIN_DATASET and isinstance(dataset, ChainDataset):
            # ChainDataset contains multiple datasets
            for sub_dataset in dataset.datasets:
                self._extract_datasets_recursive(sub_dataset, result)
            return
        
        # Leaf dataset - add to result
        result.append(dataset)
    
    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log final statistics at end of training."""
        if trainer.global_rank != 0:
            return

        logger.info("[VAL RESEED] Final statistics")
        logger.info("[VAL RESEED]   Total datasets reseeded: %s", self._reseeded_count)
        logger.info("[VAL RESEED]   Total datasets skipped:  %s", self._skipped_count)


__all__ = ["ValidationReseedingCallback"]
