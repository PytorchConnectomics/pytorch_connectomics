"""Reducer coverage for the opt-in MedNeXt dummy parameter freeze."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from connectomics.config.pipeline import from_dict
from connectomics.models.architectures.mednext_models import MEDNEXT_AVAILABLE, build_mednext_custom
from connectomics.training.lightning.trainer import _select_ddp_strategy_kwargs


def _worker(rank, port, checkpoint, static):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        cfg = from_dict(
            {
                "model": {
                    "arch": {"type": "mednext_custom"},
                    "mednext_freeze_dummy_tensor": True,
                    "mednext": {
                        "base_channels": 2,
                        "exp_r": 2,
                        "block_counts": [1] * 9,
                        "checkpoint_style": checkpoint,
                    },
                    "loss": {"deep_supervision": False},
                },
                "optimization": {"ddp_static_graph": static},
            }
        )
        model = torch.nn.parallel.DistributedDataParallel(
            build_mednext_custom(cfg), **_select_ddp_strategy_kwargs(cfg)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        torch.manual_seed(rank)
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.randn(1, 1, 32, 32, 32)).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        for p in model.parameters():
            gathered = [torch.empty_like(p) for _ in range(2)]
            dist.all_gather(gathered, p.detach())
            assert torch.equal(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.skipif(not MEDNEXT_AVAILABLE, reason="nnunet_mednext unavailable")
@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo unavailable")
@pytest.mark.parametrize("checkpoint", [None, "outside_block"])
@pytest.mark.parametrize("static", [False, True])
def test_ddp_freeze(checkpoint, static):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, checkpoint, static), nprocs=2, join=True)
