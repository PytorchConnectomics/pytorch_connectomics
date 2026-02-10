import numpy as np
import torch
from monai.data import MetaTensor

from connectomics.data.process.segment import seg_erosion_instance


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
