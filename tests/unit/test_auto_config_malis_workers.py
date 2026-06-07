"""MalisLoss worker auto-default + CPU-budget reservation in the runtime resolver."""

import unittest

from omegaconf import OmegaConf

from connectomics.config.hardware import auto_config
from connectomics.config.hardware.auto_config import resolve_runtime_resource_sentinels


def _make_cfg(num_workers, batch_size=2, losses=None):
    return OmegaConf.create(
        {
            "system": {"num_gpus": 1, "num_workers": num_workers},
            "data": {"dataloader": {"batch_size": batch_size}},
            "model": {"loss": {"losses": losses if losses is not None else []}},
        }
    )


class TestMalisWorkerResolver(unittest.TestCase):
    def setUp(self):
        # Pin CPU/GPU detection so the math is deterministic.
        self._orig_cpus = auto_config._available_cpus_for_current_run
        self._orig_gpu = auto_config.get_gpu_info
        self._orig_proc = auto_config._infer_local_process_count
        auto_config._available_cpus_for_current_run = lambda: 16
        auto_config.get_gpu_info = lambda: {"cuda_available": True, "num_gpus": 1}
        auto_config._infer_local_process_count = lambda **kw: 1

    def tearDown(self):
        auto_config._available_cpus_for_current_run = self._orig_cpus
        auto_config.get_gpu_info = self._orig_gpu
        auto_config._infer_local_process_count = self._orig_proc

    def _malis(self, **kwargs):
        item = {"function": "MalisLoss", "weight": 1.0}
        if kwargs:
            item["kwargs"] = dict(kwargs)
        return item

    def test_auto_default_injected_batch_times_two(self):
        cfg = _make_cfg(num_workers=4, batch_size=2, losses=[self._malis()])
        resolve_runtime_resource_sentinels(cfg, print_results=False)
        self.assertEqual(cfg.model.loss.losses[0].kwargs.malis_num_workers, 4)
        # num_workers explicit -> not reduced.
        self.assertEqual(cfg.system.num_workers, 4)

    def test_auto_default_capped_at_8(self):
        cfg = _make_cfg(num_workers=4, batch_size=8, losses=[self._malis()])
        resolve_runtime_resource_sentinels(cfg, print_results=False)
        self.assertEqual(cfg.model.loss.losses[0].kwargs.malis_num_workers, 8)

    def test_minus_one_reserves_malis_from_dataloader(self):
        cfg = _make_cfg(num_workers=-1, batch_size=2, losses=[self._malis()])
        resolve_runtime_resource_sentinels(cfg, print_results=False)
        # 16 cpus / 1 proc - 4 malis = 12 dataloader workers.
        self.assertEqual(cfg.system.num_workers, 12)
        self.assertEqual(cfg.model.loss.losses[0].kwargs.malis_num_workers, 4)

    def test_explicit_value_preserved_and_reserved(self):
        cfg = _make_cfg(num_workers=-1, batch_size=2, losses=[self._malis(malis_num_workers=2)])
        resolve_runtime_resource_sentinels(cfg, print_results=False)
        self.assertEqual(cfg.model.loss.losses[0].kwargs.malis_num_workers, 2)
        self.assertEqual(cfg.system.num_workers, 14)  # 16 - 2

    def test_serial_optout_reserves_nothing(self):
        cfg = _make_cfg(num_workers=-1, batch_size=2, losses=[self._malis(malis_num_workers=1)])
        resolve_runtime_resource_sentinels(cfg, print_results=False)
        self.assertEqual(cfg.model.loss.losses[0].kwargs.malis_num_workers, 1)
        self.assertEqual(cfg.system.num_workers, 16)  # full budget, nothing reserved

    def test_no_malis_unaffected(self):
        cfg = _make_cfg(
            num_workers=-1,
            batch_size=2,
            losses=[{"function": "PerChannelBCEWithLogitsLoss", "weight": 1.0}],
        )
        resolve_runtime_resource_sentinels(cfg, print_results=False)
        self.assertEqual(cfg.system.num_workers, 16)  # unchanged behavior


if __name__ == "__main__":
    unittest.main()
