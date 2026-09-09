from __future__ import annotations

import torch
import torch.nn as nn

from connectomics.models.architectures.dual_view_mednext import (
    DualViewMedNeXt,
    ReliabilityGate3d,
)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv3(x)


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class _FakeMedNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [2, 4, 8, 16, 32]
        self.stem = nn.Conv3d(1, widths[0], kernel_size=1)
        self.enc_block_0 = nn.Sequential(nn.Conv3d(2, 2, 3, padding=1), nn.GELU())
        self.down_0 = _Down(2, 4)
        self.enc_block_1 = nn.Sequential(nn.Conv3d(4, 4, 3, padding=1), nn.GELU())
        self.down_1 = _Down(4, 8)
        self.enc_block_2 = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1), nn.GELU())
        self.down_2 = _Down(8, 16)
        self.enc_block_3 = nn.Sequential(nn.Conv3d(16, 16, 3, padding=1), nn.GELU())
        self.down_3 = _Down(16, 32)
        self.bottleneck = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.GELU())
        self.up_3 = _Up(32, 16)
        self.dec_block_3 = nn.Identity()
        self.up_2 = _Up(16, 8)
        self.dec_block_2 = nn.Identity()
        self.up_1 = _Up(8, 4)
        self.dec_block_1 = nn.Identity()
        self.up_0 = _Up(4, 2)
        self.dec_block_0 = nn.Identity()
        self.out_0 = nn.Conv3d(2, 9, 1)
        self.out_1 = nn.Conv3d(4, 9, 1)
        self.out_2 = nn.Conv3d(8, 9, 1)
        self.out_3 = nn.Conv3d(16, 9, 1)
        self.out_4 = nn.Conv3d(32, 9, 1)


def _build(design: str, angle_conditioning: bool = False):
    return DualViewMedNeXt(
        _FakeMedNeXt(),
        design=design,
        deep_supervision=True,
        view_angles_deg=[45.0, 135.0],
        angle_conditioning=angle_conditioning,
        gate_reduction=4,
        checkpoint_style=None,
    )


def test_dual_view_mednext_output_contract():
    image = torch.randn(1, 2, 16, 32, 32)
    for model in (_build("siamese_mean"), _build("partial_gated", True)):
        outputs = model(image)
        assert list(outputs) == ["output", "ds_4", "ds_3", "ds_2", "ds_1"]
        assert outputs["output"].shape == (1, 9, 16, 32, 32)
        assert outputs["ds_1"].shape == (1, 9, 8, 16, 16)
        assert outputs["ds_4"].shape == (1, 9, 1, 2, 2)


def test_partial_design_duplicates_only_the_shallow_path():
    model = _build("partial_gated", True)
    assert model.right_stem is not model.trunk.stem
    assert model.right_enc_block_0 is not model.trunk.enc_block_0
    assert model.right_down_0 is not model.trunk.down_0
    assert not hasattr(model, "right_enc_block_1")


def test_reliability_gate_initializes_to_exact_mean():
    gate = ReliabilityGate3d(4)
    left = torch.randn(2, 4, 3, 5, 7)
    right = torch.randn_like(left)
    torch.testing.assert_close(gate(left, right), 0.5 * (left + right))


def test_dual_view_model_rejects_nonpaired_input():
    model = _build("siamese_mean")
    try:
        model(torch.randn(1, 1, 16, 32, 32))
    except ValueError as exc:
        assert "[B,2,Z,Y,X]" in str(exc)
    else:
        raise AssertionError("expected non-paired input to be rejected")

