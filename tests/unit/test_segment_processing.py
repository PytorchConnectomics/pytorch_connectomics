import numpy as np
import torch
from monai.data import MetaTensor

from connectomics.data.processing.segment import seg_erosion_instance


def _sample_seg_3d():
    seg = np.zeros((3, 8, 8), dtype=np.int32)
    seg[:, 1:4, 1:4] = 1
    seg[:, 4:7, 4:7] = 2
    return seg


def _sample_seg_2d():
    seg = np.zeros((8, 8), dtype=np.int32)
    seg[1:4, 1:4] = 1
    seg[4:7, 4:7] = 2
    return seg


def test_seg_erosion_instance_torch_3d_matches_numpy():
    seg_np = _sample_seg_3d()
    out_np = seg_erosion_instance(seg_np.copy(), tsz_h=1)

    seg_torch = torch.from_numpy(seg_np.copy())
    out_torch = seg_erosion_instance(seg_torch.clone(), tsz_h=1)

    assert isinstance(out_torch, torch.Tensor)
    assert out_torch.dtype == seg_torch.dtype
    np.testing.assert_array_equal(out_torch.cpu().numpy(), out_np)


def test_seg_erosion_instance_torch_2d_matches_numpy():
    seg_np = _sample_seg_2d()
    out_np = seg_erosion_instance(seg_np.copy(), tsz_h=1)

    seg_torch = torch.from_numpy(seg_np.copy())
    out_torch = seg_erosion_instance(seg_torch.clone(), tsz_h=1)

    assert isinstance(out_torch, torch.Tensor)
    assert out_torch.dtype == seg_torch.dtype
    np.testing.assert_array_equal(out_torch.cpu().numpy(), out_np)


def test_seg_erosion_instance_metatensor_3d_matches_numpy():
    seg_np = _sample_seg_3d()
    out_np = seg_erosion_instance(seg_np.copy(), tsz_h=1)

    seg_meta = MetaTensor(torch.from_numpy(seg_np.copy()))
    out_meta = seg_erosion_instance(seg_meta.clone(), tsz_h=1)

    assert isinstance(out_meta, MetaTensor)
    assert out_meta.dtype == seg_meta.dtype
    np.testing.assert_array_equal(out_meta.cpu().numpy(), out_np)


def _sample_seg_3d_with_ignore_ring():
    """One labelled object surrounded by an unlabeled (-1) ring, as in the
    zebrafinch padded GT cubes (~40% of each cube is the -1 ring)."""
    seg = np.full((5, 9, 9), -1, dtype=np.int32)
    seg[:, 3:6, 3:6] = 7
    return seg


def test_seg_erosion_instance_preserves_ignore_sentinel():
    """Erosion must not relabel the -1 ignore sentinel as background.

    `seg * keep` turned -1 into 0, which silently converts "unlabeled, mask this
    edge out of the loss" into "supervised background". AffinityTarget keys its
    mask off `seg != -1`, so the whole padded ring became a negative target.
    """
    seg = _sample_seg_3d_with_ignore_ring()
    n_ignore = int((seg == -1).sum())

    for tsz_h in (1, 2):
        out = seg_erosion_instance(seg.copy(), tsz_h=tsz_h)
        assert int((out == -1).sum()) == n_ignore, f"ignore voxels lost at tsz_h={tsz_h}"
        assert not ((seg == -1) & (out == 0)).any(), "ignore became background"

    out_torch = seg_erosion_instance(torch.from_numpy(seg.copy()), tsz_h=2)
    np.testing.assert_array_equal(
        out_torch.cpu().numpy(), seg_erosion_instance(seg.copy(), tsz_h=2)
    )


def test_seg_erosion_instance_still_erodes_between_touching_objects():
    """Guard the fix: preserving negatives must not disable erosion itself."""
    seg = np.zeros((3, 7, 7), dtype=np.int32)
    seg[:, :, :3] = 1
    seg[:, :, 3:] = 2

    out = seg_erosion_instance(seg.copy(), tsz_h=1)

    assert int((out > 0).sum()) < seg.size, "erosion removed nothing"
    assert set(np.unique(out)) - {0} == {1, 2}, "labels must survive erosion"
    # the shared boundary between the two objects is what gets dropped
    assert (out[:, :, 2:4] == 0).all()
