import torch

from connectomics.utils.visualizer import get_visualization_mask


def test_get_visualization_mask_prefers_valid_mask():
    mask = torch.zeros(1, 1, 8, 8)
    valid_mask = torch.ones(1, 1, 8, 8)
    batch = {"mask": mask, "valid_mask": valid_mask}

    selected = get_visualization_mask(batch)

    assert selected is valid_mask


def test_get_visualization_mask_falls_back_to_mask():
    mask = torch.zeros(1, 1, 8, 8)
    batch = {"mask": mask}

    selected = get_visualization_mask(batch)

    assert selected is mask


def test_get_visualization_mask_returns_none_if_missing():
    batch = {"image": torch.zeros(1, 1, 8, 8)}

    selected = get_visualization_mask(batch)

    assert selected is None
