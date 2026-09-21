"""Periodic checkpoints remain optimizer-boundary artifacts after accumulated resume."""

import hashlib

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from connectomics.training.lightning.callbacks import EMAWeightsCallback, OptimizerStepCheckpoint


class TrainingProbe(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1, bias=False)

    def training_step(self, batch, batch_idx):
        return self.model(batch).sum()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class BatchCounter(pl.Callback):
    def __init__(self):
        self.microbatches = 0
        self.observed = []

    def state_dict(self):
        return {"microbatches": self.microbatches}

    def load_state_dict(self, state_dict):
        self.microbatches = state_dict["microbatches"]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.microbatches += 1
        self.observed.append((trainer.global_step, self.microbatches))


def trainer_for(directory, max_steps, counter):
    checkpoint = OptimizerStepCheckpoint(
        dirpath=directory,
        filename="{step:08d}",
        monitor=None,
        every_n_train_steps=10,
        every_n_epochs=0,
        save_top_k=-1,
        save_last=False,
        save_on_train_epoch_end=False,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[counter, EMAWeightsCallback(decay=0.9), checkpoint],
        max_steps=max_steps,
        max_epochs=-1,
        accumulate_grad_batches=4,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    return trainer


def test_periodic_resume_preserves_step20_until_new_optimizer_steps_reach30(tmp_path):
    # Stop within an epoch, as staged production runs do.
    loader = DataLoader(torch.ones(160, 1), batch_size=1)
    original_counter = BatchCounter()
    trainer = trainer_for(tmp_path, 20, original_counter)
    trainer.fit(TrainingProbe(), train_dataloaders=loader)
    path20 = tmp_path / "step=00000020.ckpt"
    original_hash = hashlib.sha256(path20.read_bytes()).hexdigest()
    state20 = torch.load(path20, map_location="cpu", weights_only=False)
    assert state20["callbacks"]["BatchCounter"]["microbatches"] == 80

    resumed_counter = BatchCounter()
    resumed = trainer_for(tmp_path, 30, resumed_counter)
    save = resumed.save_checkpoint
    saves = []

    def observed_save(filepath, *args, **kwargs):
        saves.append((resumed.global_step, resumed_counter.microbatches))
        return save(filepath, *args, **kwargs)

    resumed.save_checkpoint = observed_save
    resumed.fit(TrainingProbe(), train_dataloaders=loader, ckpt_path=path20)

    assert resumed_counter.observed[0] == (20, 81)
    assert saves == [(30, 120)]
    assert hashlib.sha256(path20.read_bytes()).hexdigest() == original_hash
    state30 = torch.load(tmp_path / "step=00000030.ckpt", map_location="cpu", weights_only=False)
    assert state30["global_step"] == 30
    assert state30["callbacks"]["BatchCounter"]["microbatches"] == 120
    assert state30["callbacks"]["EMAWeightsCallback"]["updates"] == 30
    assert state30["lr_schedulers"][0]["last_epoch"] == 30
    assert state30["lr_schedulers"][0]["T_max"] == 200000
    assert {int(state["step"]) for state in state30["optimizer_states"][0]["state"].values()} == {
        30
    }
