"""RSUNet interpolation and stage-depth contracts on small CPU volumes."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from connectomics.config.schema.model import ModelConfig
from connectomics.models.architectures.rsunet import (
    BilinearUp3d,
    ResBlock,
    RSUNet,
    build_rsunet,
    build_rsunet_iso,
)


@pytest.mark.parametrize("factor", [(1, 1, 1), (1, 2, 2), (2, 2, 1), (2, 2, 2), (3, 1, 2)])
def test_interpolation_matches_zero_extended_trilinear(factor):
    x = torch.randn(1, 2, 4, 5, 6, dtype=torch.float64, requires_grad=True)
    up = BilinearUp3d(2, 2, factor).double()
    actual = up(x)
    reference = F.interpolate(
        F.pad(x, (1, 1, 1, 1, 1, 1)),
        scale_factor=factor,
        mode="trilinear",
        align_corners=False,
    )
    reference = reference[
        :, :, factor[0] : -factor[0], factor[1] : -factor[1], factor[2] : -factor[2]
    ]
    assert actual.shape == (1, 2, *(size * scale for size, scale in zip(x.shape[2:], factor)))
    torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-6)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert not list(up.parameters())


@pytest.mark.parametrize("factor", [(1, 2, 2), (2, 2, 1), (2, 2, 2)])
def test_interpolation_preserves_interior_constants_and_documents_boundary(factor):
    output = BilinearUp3d(1, 1, factor)(torch.ones(1, 1, 5, 5, 5))
    interior = output[:, :, 2:-2, 2:-2, 2:-2]
    torch.testing.assert_close(interior, torch.ones_like(interior))
    # Zero extension contributes a quarter of each interpolated boundary sample.
    expected_corner = 0.75 ** sum(scale == 2 for scale in factor)
    assert output[0, 0, 0, 0, 0].item() == expected_corner


def test_isotropic_impulse_is_symmetric_in_every_axis_and_keeps_channels_separate():
    impulse = torch.zeros(1, 2, 5, 5, 5)
    impulse[0, 0, 2, 2, 2] = 1
    response = BilinearUp3d(2, 2, (2, 2, 2))(impulse)
    torch.testing.assert_close(response[:, 0], response[:, 0].transpose(-1, -2))
    torch.testing.assert_close(response[:, 0], response[:, 0].transpose(-1, -3))
    assert response[:, 0].sum().item() == 8
    assert response[:, 0].max().item() == 0.75**3
    assert torch.count_nonzero(response[:, 1]) == 0


@pytest.mark.parametrize("factor", [(2, 2), (2, 2, 2, 2), (2, 0, 2), (-1, 2, 2), (1.5, 2, 2)])
def test_interpolation_rejects_invalid_factors(factor):
    with pytest.raises(ValueError, match="three positive integers"):
        BilinearUp3d(1, 1, factor)


def test_default_depth_matches_explicit_single_block_stages():
    kwargs = {"in_channels": 1, "out_channels": 6, "width": [4, 8, 12]}
    torch.manual_seed(12)
    default = RSUNet(**kwargs).eval()
    torch.manual_seed(12)
    explicit = RSUNet(**kwargs, residual_blocks_per_stage=[1, 1, 1]).eval()
    assert default.residual_blocks_per_stage == [1, 1, 1]
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert "input_conv.res.conv1.weight" in default.state_dict()
    assert not any("extra_res" in key for key in default.state_dict())
    x = torch.randn(1, 1, 8, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(default(x), explicit(x), atol=0, rtol=0)


@pytest.mark.parametrize("counts", [[1, 1, 1], [1, 2, 3]])
@pytest.mark.parametrize("deep_supervision", [False, True])
def test_stage_depth_preserves_output_shapes_and_finite_gradients(counts, deep_supervision):
    model = RSUNet(
        1,
        6,
        width=[4, 8, 12],
        down_factors=[(2, 2, 1), (2, 2, 2)],
        residual_blocks_per_stage=counts,
        norm="group",
        num_groups=4,
        deep_supervision=deep_supervision,
    )
    x = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    output = model(x)
    if deep_supervision:
        assert set(output) == {"output", "ds_1", "ds_2"}
        assert output["ds_1"].shape == (1, 6, 2, 2, 4)
        assert output["ds_2"].shape == (1, 6, 4, 4, 8)
        loss = sum(value.square().mean() for value in output.values())
        output = output["output"]
    else:
        loss = output.square().mean()
    assert output.shape == (1, 6, 8, 8, 8)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("builder", [build_rsunet, build_rsunet_iso])
def test_structured_config_builds_depth_at_matching_encoder_decoder_resolutions(builder):
    model_cfg = OmegaConf.structured(ModelConfig)
    model_cfg.rsunet.width = [4, 8, 12]
    model_cfg.rsunet.residual_blocks_per_stage = [1, 2, 3]
    model = builder(SimpleNamespace(model=model_cfg))
    stages = [model.input_conv, *(block.conv for block in model.down_blocks)]
    stages += [block.conv for block in model.up_blocks]
    actual_counts = [
        sum(isinstance(module, ResBlock) for module in stage.modules()) for stage in stages
    ]
    assert actual_counts == [1, 2, 3, 2, 1]
    model_cfg.rsunet.residual_blocks_per_stage = None
    shallow = builder(SimpleNamespace(model=model_cfg))
    assert model.get_model_info()["parameters"] > shallow.get_model_info()["parameters"]


@pytest.mark.parametrize("counts", [[], [1], [1, 1, 1], [0, 1], [1, -1], [1, 1.5], [True, 1]])
def test_stage_depth_rejects_invalid_counts(counts):
    with pytest.raises(ValueError, match="residual_blocks_per_stage"):
        RSUNet(1, 6, width=[4, 8], residual_blocks_per_stage=counts)
