import pytest
import torch

from connectomics.config import from_dict
from connectomics.models.build import build_model
from connectomics.models.architectures.mednext_models import MedNeXtMultiHeadWrapper


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


def test_mednext_multi_head_wrapper_returns_named_outputs():
    trunk = _build_tiny_mednext(deep_supervision=False)
    model = MedNeXtMultiHeadWrapper(
        trunk,
        {
            "affinity": {"out_channels": 9, "num_blocks": 1},
            "sdt": {"out_channels": 1, "num_blocks": 0},
        },
    )
    model.eval()

    x = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        features = model.forward_features(x)
        outputs = model(x)

    assert features.shape == (1, 4, 32, 32, 32)
    assert isinstance(outputs, dict)
    assert "output" in outputs
    assert set(outputs["output"].keys()) == {"affinity", "sdt"}
    assert outputs["output"]["affinity"].shape == (1, 9, 32, 32, 32)
    assert outputs["output"]["sdt"].shape == (1, 1, 32, 32, 32)
    assert model.head_specs["affinity"]["num_blocks"] == 1
    assert model.head_specs["sdt"]["num_blocks"] == 0


def test_mednext_multi_head_wrapper_rejects_deep_supervision_trunk():
    trunk = _build_tiny_mednext(deep_supervision=True)

    with pytest.raises(ValueError, match="does not support deep supervision yet"):
        MedNeXtMultiHeadWrapper(trunk, {"affinity": {"out_channels": 9, "num_blocks": 1}})


def test_build_model_creates_mednext_multi_head_wrapper_from_config():
    cfg = from_dict(
        {
            "model": {
                "arch": {"type": "mednext_custom"},
                "in_channels": 1,
                "out_channels": 10,
                "primary_head": "affinity",
                "heads": {
                    "affinity": {"out_channels": 9, "num_blocks": 1},
                    "sdt": {"out_channels": 1, "num_blocks": 0},
                },
                "mednext": {
                    "base_channels": 4,
                    "exp_r": 2,
                    "kernel_size": 3,
                    "block_counts": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "do_res": True,
                    "do_res_up_down": True,
                    "dim": "3d",
                    "grn": False,
                },
                "loss": {"deep_supervision": False},
            }
        }
    )

    model = build_model(cfg)
    assert isinstance(model, MedNeXtMultiHeadWrapper)
    assert model.primary_head == "affinity"

    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        outputs = model(x)

    assert outputs["output"]["affinity"].shape == (1, 9, 32, 32, 32)
    assert outputs["output"]["sdt"].shape == (1, 1, 32, 32, 32)
