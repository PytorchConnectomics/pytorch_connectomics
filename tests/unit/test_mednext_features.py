import pytest
import torch


nnunet_mednext = pytest.importorskip("nnunet_mednext")
from nnunet_mednext import MedNeXt  # noqa: E402


def _build_tiny_mednext(deep_supervision: bool = False) -> MedNeXt:
    return MedNeXt(
        in_channels=1,
        n_channels=4,
        n_classes=3,
        exp_r=2,
        kernel_size=3,
        deep_supervision=deep_supervision,
        do_res=True,
        do_res_up_down=True,
        block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        checkpoint_style=None,
        dim="3d",
        grn=False,
    )


def test_mednext_forward_features_matches_main_projection():
    model = _build_tiny_mednext(deep_supervision=False)
    model.eval()

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        features = model.forward_features(x)
        projected = model.forward_output(features)
        logits = model(x)

    assert features.shape == (1, 4, 32, 32, 32)
    assert projected.shape == (1, 3, 32, 32, 32)
    assert torch.allclose(projected, logits)


def test_mednext_forward_features_preserves_deep_supervision_behavior():
    model = _build_tiny_mednext(deep_supervision=True)
    model.eval()

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        features = model.forward_features(x)
        outputs = model(x)

    assert features.shape == (1, 4, 32, 32, 32)
    assert isinstance(outputs, list)
    assert len(outputs) == 5
    assert outputs[0].shape == (1, 3, 32, 32, 32)
