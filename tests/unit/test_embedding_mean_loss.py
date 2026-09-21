"""DeepEM parity and embedding supervision contracts.

Oracle below is trimmed from the planner-cached /tmp/deepem_mean.py, source:
https://github.com/ZettaAI/DeepEM/blob/feature/sr-zero-pad/deepem/loss/mean.py
Only unused downsampling is removed and scalar accumulators retain means.dtype
instead of upstream fp32 (required to compare both sides in fp64).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cc3d
import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

from connectomics.config import load_config, resolve_default_profiles
from connectomics.models.losses.build import create_loss
from connectomics.models.losses.embedding import EmbeddingMeanLoss
from connectomics.training.losses import LossOrchestrator


def create_mapping(trgt: npt.NDArray, splt: npt.NDArray, mask: npt.NDArray) -> list[list[int]]:
    trgt, splt = trgt.astype(np.uint64), splt.astype(np.uint64)
    encoded = (2**32) * trgt + splt
    encoded[mask == 0] = 0
    unq = np.unique(encoded)
    mapping: dict[int, list[int]] = {}
    for unq_id in unq:
        trgt_id, splt_id = int(unq_id // (2**32)), int(unq_id % (2**32))
        mapping[trgt_id] = mapping.get(trgt_id, []) + [splt_id]
    result = list(mapping.values())
    return result


class MeanLoss(nn.Module):
    """
    Means-based loss for metric embeddings.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        delta_v: float = 0.0,
        delta_d: float = 1.5,
        recompute_ext: bool = False,
        mask_background: bool = True,
        loss_scale_factor: tuple[float, float, float] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_v = delta_v  # Variance (intra-cluster pull force) hinge
        self.delta_d = delta_d  # Distance (inter-cluster push force) hinge
        self.recompute_ext = recompute_ext
        self.mask_background = mask_background
        self.loss_scale_factor = loss_scale_factor

    def forward(
        self,
        embd: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
        splt: torch.Tensor | None = None,
        sr_scale_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        :param embd: Embeddings
        :param trgt: Target segmentation
        :param mask: Segmentation mask
        :param splt: Connected components of the target segmentation
        :param sr_scale_z: SR scale factor in Z (>0 for zero-padded aniso samples)
        """
        device = embd.device

        groups = None
        if self.recompute_ext:
            assert splt is not None
            trgt = torch.squeeze(trgt)
            splt = torch.squeeze(splt)
            mask = torch.squeeze(mask)
            groups = create_mapping(trgt.cpu().numpy(), splt.cpu().numpy(), mask.cpu().numpy())
            trgt = splt

        trgt = trgt.to(torch.int)

        # Filter out background and get unique IDs
        masked_trgt = trgt[mask > 0]
        if self.mask_background:
            masked_trgt = masked_trgt[masked_trgt != 0]
        ids = torch.unique(masked_trgt).tolist()

        # Recompute external matrix
        mext = self.compute_ext_matrix(ids, groups, self.recompute_ext, device)
        vecs = self.generate_vecs(embd, trgt, mask, ids)
        means = [torch.mean(vec, dim=0) for vec in vecs]
        weights = [1.0] * len(vecs)

        # Dummy nmsk
        nmsk = torch.tensor([1]).to(device, dtype=torch.float)

        # Compute loss
        loss_int = self.compute_loss_int(vecs, means, weights, device)
        loss_ext = self.compute_loss_ext(means, weights, mext, device)
        loss_nrm = self.compute_loss_nrm(means, device)

        loss = (self.alpha * loss_int) + (self.beta * loss_ext) + (self.gamma * loss_nrm)
        # Only handle the empty case. Keep loss intact otherwise.
        if not vecs:
            # Graph-connected zero during training. Safe no-op during eval.
            loss = (embd * mask).sum() * 0
        return loss, nmsk

    def compute_loss_int(
        self,
        vecs: list[torch.Tensor],
        means: list[torch.Tensor],
        weights: list[float],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the internal term of the loss."""
        assert len(vecs) == len(means) == len(weights)
        zero = lambda: torch.zeros(
            1, dtype=means[0].dtype if means else torch.float64, device=device
        ).squeeze()
        loss = zero()
        for vec, mean, weight in zip(vecs, means, weights):
            margin = torch.norm(vec - mean, p=1, dim=1) - self.delta_v
            loss += weight * torch.mean(torch.max(margin, zero()) ** 2)
        loss /= max(1.0, len(vecs))
        return loss

    def compute_loss_ext(
        self,
        means: list[torch.Tensor],
        weights: list[float],
        mext: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the external term of the loss."""
        assert len(means) == len(weights)
        zero = lambda: torch.zeros(
            1, dtype=means[0].dtype if means else torch.float64, device=device
        ).squeeze()
        loss = zero()
        count = len(means)
        if (count > 1) and (mext is not None):
            means0 = torch.stack(means)
            means1 = means0.unsqueeze(0)  # 1 x N x Dim
            means2 = means0.unsqueeze(1)  # N x 1 x Dim
            margin = 2 * self.delta_d - torch.norm(means2 - means1, p=1, dim=2)
            margin = margin[mext.to(device)]
            loss = torch.sum(torch.max(margin, zero()) ** 2)
            loss /= max(1.0, count * (count - 1.0))  # Normalize
        return loss

    def compute_loss_nrm(self, means: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        """Compute the regularization term of the loss."""
        zero = lambda: torch.zeros(
            1, dtype=means[0].dtype if means else torch.float64, device=device
        ).squeeze()
        loss = zero()
        if len(means) > 0:
            loss = torch.mean(torch.norm(torch.stack(means), p=1, dim=1))
        return loss

    def generate_vecs(
        self,
        embd: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
        ids: Sequence[int],
    ) -> list[torch.Tensor]:
        """
        Generate a list of vectorized embeddings for each ground truth object.
        """
        if self.mask_background and 0 in ids:
            raise ValueError("ID '0' is not allowed when mask_background is enabled.")

        mask_bool = mask.bool() if not self.mask_background else None
        result = []

        for obj_id in ids:
            obj_mask = (
                (trgt == int(obj_id)) & mask_bool
                if mask_bool is not None
                else (trgt == int(obj_id))
            )
            idx = torch.nonzero(obj_mask, as_tuple=True)

            if idx[0].numel() == 0:
                # If there are no indices for this ID, skip to the next one
                continue

            vec = embd[0, :, idx[-3], idx[-2], idx[-1]].transpose(0, 1)  # Count x Dim
            result.append(vec)

        return result

    def compute_ext_matrix(
        self,
        ids: Sequence[int],
        groups: Sequence[Sequence[int]] | None = None,
        recompute_ext: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor | None:
        """
        Compute a matrix that indicates the presence of 'external' interaction
        between objects.
        """
        num_ids = len(ids)

        # Recompute external matrix
        if recompute_ext:
            assert groups is not None
            mext_np = np.ones((num_ids, num_ids)) - np.eye(num_ids)
            idmap = {x: i for i, x in enumerate(ids)}
            for group in groups:
                for i, id_i in enumerate(group):
                    for id_j in group[i + 1 :]:
                        mext_np[idmap[id_i], idmap[id_j]] = 0
                        mext_np[idmap[id_j], idmap[id_i]] = 0
            mext = torch.from_numpy(mext_np).to(device, dtype=torch.bool)
        else:
            mext = ~torch.eye(num_ids, dtype=torch.bool, device=device)

        # Safeguard
        if mext.sum() == 0:
            return None

        return mext


def fixture_labels():
    ids = torch.zeros(1, 8, 16, 16, dtype=torch.long)
    ids[:, 1:4, 1:6, 1:6] = 1
    ids[:, 4:7, 9:15, 9:15] = 1
    ids[:, 1:6, 7:10, 1:7] = 2
    ids[:, 2:7, 1:6, 10:15] = 3
    return ids


def value_grad(loss, pred, ids, mask=None):
    value = loss(pred, None, gt_seg=ids, mask=mask)
    return value.detach(), torch.autograd.grad(value, pred)[0]


@pytest.mark.parametrize("recompute", [False, True])
def test_noncontiguous_prediction(recompute):
    torch.manual_seed(17)
    ids = fixture_labels()
    storage = torch.randn(1, 16, 8, 16, 12, dtype=torch.float64, requires_grad=True)
    pred = storage.permute(0, 4, 2, 3, 1)
    assert not pred.is_contiguous()
    contiguous = pred.detach().contiguous().requires_grad_()
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, :, 3, :] = False
    loss = EmbeddingMeanLoss(recompute_ext=recompute)
    actual = loss(pred, None, gt_seg=ids, mask=mask)
    grad = torch.autograd.grad(actual, storage)[0].permute(0, 4, 2, 3, 1)
    expected, expected_grad = value_grad(loss, contiguous, ids, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(grad, expected_grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("delta_v", [0, 0.5])
@pytest.mark.parametrize("connectivity", [6, 26])
@pytest.mark.parametrize("random_ids", [False, True])
def test_deepem_parity(recompute, delta_v, connectivity, random_ids):
    torch.manual_seed(17)
    ids = torch.randint(0, 21, (1, 8, 16, 16)) if random_ids else fixture_labels()
    # Extrude a random 20-ID plane to bound the oracle's component count.
    if random_ids:
        ids = torch.randint(1, 21, (1, 1, 16, 16)).expand(1, 8, 16, 16).clone()
    pred = (torch.randn(1, 12, 8, 16, 16, dtype=torch.float64) * 0.2).requires_grad_()
    mask = torch.ones_like(ids, dtype=torch.bool)
    splt = torch.from_numpy(cc3d.connected_components(ids[0].numpy(), connectivity=connectivity))
    oracle = MeanLoss(recompute_ext=recompute, delta_v=delta_v)
    expected, _ = oracle(pred, ids, mask, splt=splt[None])
    expected_grad = torch.autograd.grad(expected, pred)[0]
    loss = EmbeddingMeanLoss(recompute_ext=recompute, delta_v=delta_v, connectivity=connectivity)
    actual, actual_grad = value_grad(loss, pred, ids, mask)
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-12)
    fp32 = loss(pred.float(), None, gt_seg=ids)
    torch.testing.assert_close(fp32.double(), actual, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("recompute,expected", [(True, 6.5 / 6), (False, 19 / 6)])
def test_ordered_pair_denominator(recompute, expected):
    ids = torch.tensor([1, 0, 1 if recompute else 2, 0, 3]).reshape(1, 1, 1, 5)
    pred = torch.tensor([0.0, 0.0, 0.5, 0.0, 2.0], dtype=torch.float64).reshape(1, 1, 1, 1, 5)
    loss = EmbeddingMeanLoss(alpha=0, gamma=0, recompute_ext=recompute)
    assert loss(pred, None, gt_seg=ids).item() == pytest.approx(expected)


def test_diagonal_connectivity():
    ids = torch.zeros(1, 1, 4, 5, dtype=torch.long)
    ids[0, 0, 0, :2] = 1
    ids[0, 0, 1, 2:] = 1
    pred = torch.zeros(1, 1, 1, 4, 5, dtype=torch.float64)
    pred[0, 0, 0, 0, :2] = torch.tensor([0.0, 0.2])
    pred[0, 0, 0, 1, 2:] = torch.tensor([2.0, 2.4, 2.8])
    means = [pred[0, 0, 0, 0, :2], pred[0, 0, 0, 1, 2:]]
    separate = torch.stack([(v - v.mean()).square().mean() for v in means]).mean()
    combined = torch.cat(means)
    together = (combined - combined.mean()).square().mean()
    for connectivity, expected in [(6, separate), (26, together)]:
        actual = EmbeddingMeanLoss(connectivity=connectivity, gamma=0)(pred, None, gt_seg=ids)
        torch.testing.assert_close(actual, expected)


def test_batch_reduction_and_empty():
    torch.manual_seed(2)
    pred = torch.randn(2, 12, 8, 16, 16, requires_grad=True)
    ids = fixture_labels().expand(2, -1, -1, -1).clone()
    loss = EmbeddingMeanLoss()
    samples = [loss(pred[b : b + 1], None, gt_seg=ids[b : b + 1]) for b in range(2)]
    torch.testing.assert_close(loss(pred, None, gt_seg=ids), torch.stack(samples).mean())
    ids[1] = 0
    torch.testing.assert_close(loss(pred, None, gt_seg=ids), samples[0])
    ids.zero_()
    zero = loss(pred, None, gt_seg=ids)
    assert zero.requires_grad and zero.item() == 0
    zero.backward()
    assert torch.count_nonzero(pred.grad) == 0


@pytest.mark.parametrize("shape", [(1, 8, 8), (8, 1, 8), (8, 8, 1)])
def test_rank_normalization(shape):
    pred = torch.randn(2, 12, *shape)
    ids = torch.ones(2, *shape)
    mask = torch.ones_like(ids, dtype=torch.bool)
    loss = EmbeddingMeanLoss()
    expected = loss(pred, None, gt_seg=ids, mask=mask)
    actual = loss(pred, None, gt_seg=ids[:, None], mask=mask[:, None].expand(-1, 3, *shape))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "name,shape",
    [
        ("gt_seg", (2, 3, 4)),
        ("gt_seg", (2, 2, 3, 4, 5)),
        ("gt_seg", (1, 3, 4, 5)),
        ("gt_seg", (2, 3, 4, 6)),
        ("mask", (3, 4, 5)),
        ("mask", (1, 3, 4, 5)),
        ("mask", (2, 1, 3, 4, 6)),
        ("mask", (2, 1, 1, 3, 4, 5)),
    ],
)
def test_invalid_shapes(name, shape):
    kwargs = {"gt_seg": torch.ones(2, 3, 4, 5), name: torch.ones(shape)}
    with pytest.raises(ValueError, match=name):
        EmbeddingMeanLoss()(torch.zeros(2, 12, 3, 4, 5), None, **kwargs)


def test_missing_gt_and_invalid_prediction():
    with pytest.raises(ValueError, match="emit_gt_seg: true"):
        EmbeddingMeanLoss()(torch.zeros(1, 12, 2, 2, 2), None)
    with pytest.raises(ValueError, match="pred"):
        EmbeddingMeanLoss()(torch.zeros(12, 2, 2, 2), None)


@pytest.mark.parametrize(
    "kwargs", [{"connectivity": 4}, {"delta_d": 0}, {"delta_v": -1}, {"pair_chunk_size": 0}]
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        EmbeddingMeanLoss(**kwargs)


@pytest.mark.parametrize("split", [False, True])
def test_no_external_pairs(split):
    ids = torch.ones(1, 1, 1, 5, dtype=torch.long)
    if split:
        ids[..., 2] = 0
    pred = torch.randn(1, 12, 1, 1, 5, requires_grad=True)
    value, grad = value_grad(EmbeddingMeanLoss(alpha=0, gamma=0), pred, ids)
    assert value == 0 and torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("recompute", [False, True])
def test_ignored_background_and_mask(recompute):
    ids = torch.tensor([1, 1, 0, -1, 2, 2, 2]).reshape(1, 1, 1, 7)
    mask = torch.ones(1, 3, 1, 1, 7, dtype=torch.bool)
    mask[:, 1, ..., 5] = False
    pred = torch.randn(1, 12, 1, 1, 7, dtype=torch.float64, requires_grad=True)
    changed = pred.detach().clone()
    changed[..., [2, 3, 5]] += 100
    changed.requires_grad_()
    loss = EmbeddingMeanLoss(recompute_ext=recompute)
    a, ga = value_grad(loss, pred, ids, mask)
    b, gb = value_grad(loss, changed, ids, mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)
    assert torch.count_nonzero(ga[..., [2, 3, 5]]) == 0


@pytest.mark.parametrize("ignored", [False, True])
def test_invalid_voxels_cannot_bridge_components(ignored):
    ids = torch.ones(1, 1, 1, 5, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    if ignored:
        ids[..., 2] = -1
    else:
        mask[..., 2] = False
    pred = torch.tensor([0.0, 0.0, 100.0, 2.0, 2.0]).reshape(1, 1, 1, 1, 5)
    loss = EmbeddingMeanLoss(gamma=0)
    assert loss(pred, None, gt_seg=ids, mask=mask).item() == 0


def test_float_ids_round():
    ids = torch.tensor([1.1, 1.9, -1.1, 0.1]).reshape(1, 1, 1, 4)
    pred = torch.randn(1, 12, 1, 1, 4)
    loss = EmbeddingMeanLoss()
    torch.testing.assert_close(
        loss(pred, None, gt_seg=ids), loss(pred, None, gt_seg=ids.round().long())
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_work_dtype_under_autocast(dtype):
    pred = torch.randn(1, 12, 2, 2, 2).to(dtype).requires_grad_()
    ids = torch.ones(1, 2, 2, 2, dtype=torch.long)
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    loss_fn = EmbeddingMeanLoss()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss_fn(pred, None, gt_seg=ids)
    expected = loss_fn(pred.to(work_dtype), None, gt_seg=ids)
    assert actual.dtype == work_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(torch.autograd.grad(actual, pred)[0]).all()


@pytest.mark.parametrize("chunk", [1, 64, 1024])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_chunk_value_and_gradient(chunk, checkpoint):
    torch.manual_seed(4)
    ids = torch.arange(1, 301).reshape(1, 3, 10, 10)
    pred = (torch.randn(1, 12, 3, 10, 10) * 0.05).requires_grad_()
    a, ga = value_grad(EmbeddingMeanLoss(), pred, ids)
    b, gb = value_grad(
        EmbeddingMeanLoss(pair_chunk_size=chunk, pair_checkpoint=checkpoint), pred, ids
    )
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ga, gb, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fp16 GradScaler")
def test_cuda_mixed_precision():
    torch.manual_seed(43)
    conv = nn.Conv3d(1, 12, 1).cuda()
    optimizer = torch.optim.AdamW(conv.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()
    x = torch.randn(1, 1, 16, 16, 16, device="cuda") * 5
    ids = torch.zeros(1, 16, 16, 16, dtype=torch.long, device="cuda")
    ids[:, :8] = 1
    ids[0, 10, 1, :1] = 2
    ids[0, 12, 1, :2] = 3
    ids[0, 14, 1, :3] = 4
    loss_fn = EmbeddingMeanLoss()
    for iteration in range(30):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = conv(x)
            loss = loss_fn(pred, None, gt_seg=ids)
        torch.testing.assert_close(
            loss, loss_fn(pred.float(), None, gt_seg=ids), rtol=1e-5, atol=1e-5
        )
        assert loss.dtype == torch.float32
        scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if iteration >= 10:
            assert all(torch.isfinite(p.grad).all() for p in conv.parameters())
        scaler.step(optimizer)
        scaler.update()
        if iteration >= 10:
            assert scaler.get_scale() >= scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory stress")
def test_cuda_memory_stress():
    # Planner sets this from code_v0's real-data probe before running GPU gates.
    n = max(4000, 2 * int(os.environ.get("EMBEDDING_PROBE_MAX_N", "0")))
    assert n <= 64**3
    ids = torch.zeros(128, 128, 128, dtype=torch.long)
    points = torch.cartesian_prod(*(torch.arange(0, 128, 2) for _ in range(3)))[:n]
    ids[points[:, 0], points[:, 1], points[:, 2]] = torch.arange(1, n + 1)
    ids = ids[None].cuda()
    peaks = {}
    for use_checkpoint in (False, True):
        pred = (torch.randn(1, 12, 128, 128, 128, device="cuda") * 0.05).requires_grad_()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        EmbeddingMeanLoss(pair_checkpoint=use_checkpoint)(pred, None, gt_seg=ids).backward()
        torch.cuda.synchronize()
        peaks[use_checkpoint] = torch.cuda.max_memory_allocated() - baseline
        print(
            f"N={n} pair_checkpoint={use_checkpoint} peak_above_input_bytes={peaks[use_checkpoint]}"
        )
        del pred
    cfg = resolve_default_profiles(
        load_config(Path(__file__).parents[2] / "tutorials/neuron_nisb/base_banis+_embed12.yaml")
    )
    configured = cfg.model.loss.losses[1]["kwargs"].get("pair_checkpoint", False)
    assert peaks[configured] < 1.5e9, "Training pair_checkpoint must satisfy the memory gate"


def test_orchestrator_embedding_routing():
    cfg = resolve_default_profiles(
        load_config(Path(__file__).parents[2] / "tutorials/neuron_nisb/base_banis+_embed12.yaml")
    )
    terms = cfg.model.loss.losses
    losses = nn.ModuleList([create_loss(term["function"], **term["kwargs"]) for term in terms])
    captured = {}

    def hook(module, args, kwargs):
        captured.update(pred=args[0], mask=kwargs["mask"], gt_seg=kwargs["gt_seg"])

    handle = losses[1].register_forward_pre_hook(hook, with_kwargs=True)
    orchestrator = LossOrchestrator(
        cfg=cfg,
        loss_functions=losses,
        loss_weights=[1.0, 1.0],
        enable_nan_detection=False,
        debug_on_nan=False,
    )
    pred = torch.randn(1, 18, 4, 4, 4, requires_grad=True)
    labels = torch.ones(1, 6, 4, 4, 4)
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[..., 0] = False
    ids = torch.ones(1, 4, 4, 4, dtype=torch.long)
    total, _ = orchestrator.compute_standard_loss(
        pred, labels, stage="train", target_mask=mask, gt_seg=ids
    )
    handle.remove()
    assert torch.isfinite(total)
    torch.testing.assert_close(captured["pred"], pred[:, 6:18], rtol=0, atol=0)
    assert captured["gt_seg"] is ids
    torch.testing.assert_close(captured["mask"].bool(), mask[:, :3])
    embedding = losses[1](captured["pred"], None, mask=captured["mask"], gt_seg=ids)
    grad = torch.autograd.grad(embedding, pred)[0]
    assert torch.count_nonzero(grad[:, :6]) == 0
    assert torch.count_nonzero(grad[:, 6:]) > 0
