"""NumbaAffineResample must reproduce MONAI's float32 Resample for RandAffined."""

import math

import pytest
import torch
from monai.transforms import RandAffined, Resample

from connectomics.config.schema import AugmentationConfig
from connectomics.data.augmentation.build import _build_augmentations
from connectomics.data.augmentation.fast_resample import NUMBA_AVAILABLE, NumbaAffineResample

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")

KEYS = ("image", "label", "label_aux")


def _affine(resampler, padding_mode="reflection"):
    t = RandAffined(
        keys=KEYS,
        prob=1.0,
        rotate_range=(math.pi,) * 3,
        scale_range=(0.2,) * 3,
        shear_range=(0.5,) * 3,
        mode=("bilinear", "nearest", "nearest"),
        padding_mode=padding_mode,
    )
    t.rand_affine.resampler = resampler
    return t


def _data(seed, shape=(1, 23, 30, 27)):
    g = torch.Generator().manual_seed(seed)
    return {
        "image": torch.rand(shape, generator=g),
        "label": torch.randint(0, 5000, shape, generator=g, dtype=torch.int32),
        "label_aux": torch.randint(0, 3, shape, generator=g, dtype=torch.int32),
    }


@pytest.mark.parametrize("padding_mode", ["reflection", "zeros"])
def test_matches_monai_float32_over_consecutive_grids(padding_mode):
    ref = _affine(Resample(dtype=torch.float32), padding_mode)
    fast = _affine(NumbaAffineResample(dtype=torch.float32), padding_mode)
    # consecutive calls create new grids; stale cached coordinates would break equality
    for seed in range(6):
        data = _data(seed)
        ref.set_random_state(seed=seed)
        fast.set_random_state(seed=seed)
        a, b = ref(dict(data)), fast(dict(data))
        for key in KEYS:
            assert b[key].dtype == a[key].dtype == torch.float32
            assert torch.equal(b[key].as_tensor(), a[key].as_tensor()), (seed, key)


def test_labels_keep_exact_ids():
    fast = _affine(NumbaAffineResample(dtype=torch.float32))
    data = _data(7)
    fast.set_random_state(seed=7)
    out = fast(dict(data))
    assert set(out["label"].unique().long().tolist()) <= set(data["label"].unique().tolist())


@pytest.mark.parametrize("enabled", [False, True])
def test_build_flag_swaps_resampler(enabled):
    aug = AugmentationConfig()
    aug.affine.enabled, aug.affine.prob, aug.affine.fast_resample = True, 0.5, enabled
    transforms = _build_augmentations(aug, ["image", "label"])
    affine = [t for t in transforms if isinstance(t, RandAffined)]
    assert len(affine) == 1
    resampler = affine[0].rand_affine.resampler
    assert isinstance(resampler, NumbaAffineResample) is enabled
    if enabled:
        assert resampler.dtype == torch.float32


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
@pytest.mark.parametrize("shape", [(1, 3, 4, 5), (1, 1, 4, 5)])
def test_dtype_and_singleton_equivalence(dtype, mode, shape):
    from monai.transforms.utils import create_grid

    image = torch.randint(0, 10, shape).to(dtype)
    grid = torch.as_tensor(create_grid(shape[1:])).float() + 0.5
    fast = NumbaAffineResample(dtype=torch.float32)
    ref = Resample(dtype=torch.float32)
    args = {"mode": mode, "padding_mode": "reflection"}
    assert fast._is_fast_path(image, grid, mode, "reflection", False) == (
        dtype not in (torch.float16, torch.bfloat16)
    )
    assert torch.equal(fast(image, grid, **args), ref(image, grid, **args))
    grid.add_(17)
    assert torch.equal(fast(image, grid, **args), ref(image, grid, **args))


@pytest.mark.parametrize(
    "condition", ["dtype", "image_grad", "grid_grad", "align", "padding", "norm"]
)
def test_fallback_conditions(condition, monkeypatch):
    from monai.transforms.utils import create_grid

    image = torch.rand(1, 3, 4, 5)
    grid = torch.as_tensor(create_grid(image.shape[1:])).float()
    options = {"dtype": torch.float32, "padding_mode": "reflection", "align_corners": False}
    if condition == "dtype":
        options["dtype"] = torch.float64
    if condition == "image_grad":
        image.requires_grad_()
    if condition == "grid_grad":
        grid.requires_grad_()
    if condition == "align":
        options["align_corners"] = True
    if condition == "padding":
        options["padding_mode"] = "zeros"
    if condition == "norm":
        options["norm_coords"] = False
    fast = NumbaAffineResample(**options)
    ref = Resample(**options)
    monkeypatch.setattr(fast, "_coords", lambda *args: pytest.fail("unexpected fast path"))
    torch.testing.assert_close(fast(image, grid), ref(image, grid), rtol=0, atol=0)
