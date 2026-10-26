"""Equivalence to the pre-optimization loss, preserved verbatim below."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from connectomics.models.losses.losses import PerChannelBCEWithLogitsLoss


def _reduce_weighted_tensor(
    loss_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    reduction: str,
) -> torch.Tensor:
    """Reduce weighted loss values while excluding invalid (weight<=0) voxels for mean."""
    if reduction == "none":
        return loss_tensor

    if reduction == "sum":
        return loss_tensor.sum()

    # reduction == "mean"
    if weight is None:
        return loss_tensor.mean()

    valid = weight > 0
    if valid.shape != loss_tensor.shape:
        try:
            valid = torch.broadcast_to(valid, loss_tensor.shape)
        except RuntimeError as e:
            raise ValueError(
                "Weight mask shape is not broadcastable to loss tensor shape: "
                f"weight={tuple(weight.shape)}, loss={tuple(loss_tensor.shape)}"
            ) from e
    if not torch.any(valid):
        return loss_tensor.new_tensor(0.0)
    return loss_tensor[valid].mean()


class OldLoss(nn.Module):
    """BCE loss computed independently per channel with per-channel class balancing.

    Equivalent to summing C separate WeightedBCEWithLogitsLoss instances (one
    per output channel) but expressed as a single loss entry for config brevity.
    Matches the DeepEM per-edge loss structure where each affinity channel has
    its own class-balanced BCE.

    Args:
        auto_pos_weight: If True, compute per-channel pos_weight as
            min(n_neg / n_pos, max_pos_weight) from the current batch.
        max_pos_weight: Cap for auto pos_weight to avoid extreme values.
        reduction: Per-channel reduction before summing ('mean' or 'sum').
    """

    def __init__(
        self,
        auto_pos_weight: bool = True,
        max_pos_weight: float = 10.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.auto_pos_weight = auto_pos_weight
        self.max_pos_weight = max_pos_weight
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute per-channel BCE loss.

        Args:
            input: Logits [B, C, ...].
            target: Ground truth [B, C, ...].
            weight: Optional spatial mask [B, C, ...] (e.g. affinity valid mask).

        Returns:
            Sum of per-channel reduced BCE losses.
        """
        C = input.shape[1]

        # Vectorised per-channel pos_weight computation.
        pos_weight_tensor = None
        if self.auto_pos_weight:
            valid = weight > 0 if weight is not None else torch.ones_like(target, dtype=torch.bool)
            pos = (target > 0) & valid
            neg = (target <= 0) & valid

            # Reduce over batch + spatial dims → (C,)
            reduce_dims = (0,) + tuple(range(2, target.ndim))
            pos_counts = pos.sum(dim=reduce_dims).float()
            neg_counts = neg.sum(dim=reduce_dims).float()

            pw = torch.ones(C, device=input.device, dtype=torch.float32)
            nonzero = pos_counts > 0
            pw[nonzero] = torch.clamp(
                neg_counts[nonzero] / pos_counts[nonzero],
                max=self.max_pos_weight,
            )
            # (1, C, 1, ..., 1) for broadcasting; cast to input dtype for AMP
            pos_weight_tensor = pw.reshape(1, C, *([1] * (target.ndim - 2))).to(input.dtype)

        bce = F.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=pos_weight_tensor,
            reduction="none",
        )

        if weight is not None:
            bce = bce * weight

        # Per-channel reduction, then sum across channels.
        total = input.new_tensor(0.0)
        for c in range(C):
            ch_bce = bce[:, c : c + 1, ...]
            ch_w = weight[:, c : c + 1, ...] if weight is not None else None
            total = total + _reduce_weighted_tensor(ch_bce, ch_w, self.reduction)

        return total


@pytest.mark.parametrize(
    "device,autocast",
    [
        ("cpu", False),
        ("cpu", True),
        pytest.param(
            "cuda",
            False,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
        pytest.param(
            "cuda",
            True,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["none", "full", "batch", "empty_channel", "empty", "lower_rank"])
def test_reference_value_and_gradient(device, autocast, dtype, case):
    torch.manual_seed(42)
    shape = (1, 2, 2, 2) if case == "lower_rank" else (2, 3, 3, 4, 5)
    x = torch.randn(shape, device=device, dtype=dtype)
    y = torch.randint(0, 2, shape, device=device).to(dtype)
    w = (torch.rand_like(x) > 0.3).to(dtype)
    if case == "none":
        w = None
    elif case == "batch":
        w = w[:1]
    elif case == "empty_channel":
        w[:, 0] = 0
    elif case == "empty":
        w.zero_()
    elif case == "lower_rank":
        w = torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype)
    a, b = [x.detach().clone().requires_grad_() for _ in range(2)]
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        old = OldLoss()(a, y, w)
        new = PerChannelBCEWithLogitsLoss()(b, y, w)
    assert new.dtype == old.dtype
    torch.testing.assert_close(new, old)
    if case == "empty":
        # The sync-free reduction keeps the graph even when every weight is invalid.
        new.backward()
        torch.testing.assert_close(b.grad, torch.zeros_like(b))
    else:
        assert new.requires_grad == old.requires_grad
    if old.requires_grad:
        old.backward()
        new.backward()
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_single_channel_weight_errors_match(reduction):
    x = torch.randn(2, 3, 4, 4)
    y = torch.rand_like(x)
    w = torch.ones(2, 1, 4, 4)
    try:
        expected = OldLoss(reduction=reduction)(x, y, w)
    except Exception as exc:
        with pytest.raises(type(exc)):
            PerChannelBCEWithLogitsLoss(reduction=reduction)(x, y, w)
    else:
        torch.testing.assert_close(
            PerChannelBCEWithLogitsLoss(reduction=reduction)(x, y, w), expected
        )


@pytest.mark.parametrize("reduction", ["sum", "none"])
def test_other_reductions_unchanged(reduction):
    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    y = torch.randint(0, 2, x.shape).float()
    w = torch.rand_like(x)
    torch.testing.assert_close(
        PerChannelBCEWithLogitsLoss(reduction=reduction)(x, y, w),
        OldLoss(reduction=reduction)(x, y, w),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("case", ["none", "full", "batch", "empty_channel", "empty"])
def test_vectorized_mean_has_no_host_sync(case):
    x = torch.randn(2, 3, 3, 4, 5, device="cuda", requires_grad=True)
    y = torch.randint(0, 2, x.shape, device="cuda").float()
    weight = torch.ones_like(x)
    if case == "none":
        weight = None
    elif case == "batch":
        weight = weight[:1]
    elif case == "empty_channel":
        weight[:, 0] = 0
    elif case == "empty":
        weight.zero_()
    loss = PerChannelBCEWithLogitsLoss()
    try:
        torch.cuda.set_sync_debug_mode("error")
        # Include automatic pos_weight computation and the full backward pass.
        result = loss(x, y, weight)
        result.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert x.grad is not None
    assert torch.isfinite(result)
    assert torch.isfinite(x.grad).all()
    if case == "empty":
        torch.testing.assert_close(result, torch.zeros_like(result))
        torch.testing.assert_close(x.grad, torch.zeros_like(x))
