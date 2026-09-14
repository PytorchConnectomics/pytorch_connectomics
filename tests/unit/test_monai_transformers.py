import pytest
import torch
from omegaconf.errors import ConfigKeyError

from connectomics.config.pipeline import from_dict
from connectomics.models.architectures.monai_models import build_swin_unetr, build_unetr


def _config(input_size, **transformer):
    return from_dict(
        {
            "model": {
                "input_size": list(input_size),
                "in_channels": 1,
                "out_channels": 6,
                "transformer": transformer,
            }
        }
    )


@pytest.mark.parametrize("input_size", [(32, 32), (32, 32, 32)])
@pytest.mark.parametrize("proj_type", ["conv", "perceptron"])
def test_unetr_current_api_forward_backward(input_size, proj_type):
    cfg = _config(
        input_size,
        feature_size=4,
        hidden_size=24,
        mlp_dim=48,
        num_heads=4,
        proj_type=proj_type,
    )
    model = build_unetr(cfg)
    image = torch.randn(1, 1, *input_size, requires_grad=True)

    logits = model(image)
    logits.square().mean().backward()

    assert logits.shape == (1, 6, *input_size)
    assert torch.isfinite(logits).all()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0


@pytest.mark.parametrize("input_size", [(32, 64), (32, 32, 64)])
def test_swin_unetr_current_api_forward_backward(input_size):
    cfg = _config(input_size, feature_size=12)
    model = build_swin_unetr(cfg)
    image = torch.randn(1, 1, *input_size, requires_grad=True)

    logits = model(image)
    logits.square().mean().backward()

    assert logits.shape == (1, 6, *input_size)
    assert torch.isfinite(logits).all()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0


@pytest.mark.parametrize("builder", [build_unetr, build_swin_unetr])
@pytest.mark.parametrize("input_size", [(32,), (32, 32, 32, 32), (32, 0, 32)])
def test_transformer_rejects_invalid_spatial_dimensions(builder, input_size):
    with pytest.raises(ValueError, match="model.input_size"):
        builder(_config(input_size, feature_size=12))


@pytest.mark.parametrize(
    "builder,input_size,divisor",
    [(build_unetr, (32, 32, 33), 16), (build_swin_unetr, (32, 32, 48), 32)],
)
def test_transformer_rejects_incompatible_crop(builder, input_size, divisor):
    with pytest.raises(ValueError, match=f"positive multiples of {divisor}"):
        builder(_config(input_size, feature_size=12))


def test_swin_unetr_rejects_illegal_feature_size():
    with pytest.raises(ValueError, match="feature_size should be divisible by 12"):
        build_swin_unetr(_config((32, 32, 64), feature_size=16))


def test_transformer_schema_rejects_removed_pos_embed():
    with pytest.raises(ConfigKeyError, match="pos_embed"):
        _config((32, 32, 32), pos_embed="perceptron")
