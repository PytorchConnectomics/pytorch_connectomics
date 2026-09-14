import pytest
import torch

from connectomics.config.pipeline import from_dict
from connectomics.models.architectures.mednext_models import build_mednext, build_mednext_custom

pytest.importorskip("nnunet_mednext")


@pytest.mark.parametrize("size", ["S", "B", "M", "L"])
@pytest.mark.parametrize("checkpoint_style", [None, "outside_block"])
def test_preset_checkpointing_obeys_config(size, checkpoint_style):
    cfg = from_dict({"model": {"mednext": {"size": size, "checkpoint_style": checkpoint_style}}})

    model = build_mednext(cfg)

    assert model.model.outside_block_checkpointing is (checkpoint_style == "outside_block")


def test_preset_checkpointing_rejects_unknown_style():
    cfg = from_dict({"model": {"mednext": {"checkpoint_style": "inside_block"}}})

    with pytest.raises(ValueError, match="checkpoint_style must be None or 'outside_block'"):
        build_mednext(cfg)


def test_checkpointing_preserves_logits_and_gradients():
    cfg = from_dict(
        {
            "model": {
                "out_channels": 6,
                "mednext": {
                    "base_channels": 4,
                    "exp_r": 2,
                    "block_counts": [1] * 9,
                    "checkpoint_style": None,
                },
            }
        }
    )
    plain = build_mednext_custom(cfg)
    cfg.model.mednext.checkpoint_style = "outside_block"
    checkpointed = build_mednext_custom(cfg)
    checkpointed.load_state_dict(plain.state_dict())
    image = torch.randn(1, 1, 32, 32, 32)

    plain_logits = plain(image)
    checkpointed_logits = checkpointed(image)
    plain_logits.square().mean().backward()
    checkpointed_logits.square().mean().backward()

    torch.testing.assert_close(checkpointed_logits, plain_logits)
    for plain_parameter, checkpointed_parameter in zip(
        plain.parameters(), checkpointed.parameters()
    ):
        if plain_parameter.grad is None:
            assert checkpointed_parameter.grad is None
        else:
            assert checkpointed_parameter.grad is not None
            assert torch.isfinite(checkpointed_parameter.grad).all()
            torch.testing.assert_close(checkpointed_parameter.grad, plain_parameter.grad)
