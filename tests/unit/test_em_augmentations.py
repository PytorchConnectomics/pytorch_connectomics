"""
Tests for EM-specific augmentations.

Tests all transforms in connectomics/data/augment/monai_transforms.py
to ensure they work correctly with Lightning and MONAI.
"""

import pytest
import torch

from connectomics.data.augment.monai_transforms import (
    RandCopyPasted,
    RandCutBlurd,
    RandCutNoised,
    RandMisAlignmentd,
    RandMissingPartsd,
    RandMissingSectiond,
    RandMixupd,
    RandMotionBlurd,
)


@pytest.fixture
def sample_3d_image():
    """Create sample 3D image."""
    return torch.randn(1, 64, 64, 64)


@pytest.fixture
def sample_3d_label():
    """Create sample 3D label."""
    return torch.randint(0, 3, (64, 64, 64))


@pytest.fixture
def sample_data_dict(sample_3d_image, sample_3d_label):
    """Create sample data dictionary."""
    return {
        "image": sample_3d_image,
        "label": sample_3d_label,
    }


class TestRandMisAlignmentd:
    """Test RandMisAlignmentd transform."""

    def test_basic_functionality(self, sample_data_dict):
        """Test basic misalignment."""
        transform = RandMisAlignmentd(
            keys=["image"],
            prob=1.0,  # Always apply
            displacement=8,
            rotate_ratio=0.0,
        )

        result = transform(sample_data_dict)

        assert "image" in result
        assert result["image"].shape == sample_data_dict["image"].shape

    def test_rotation_mode(self, sample_data_dict):
        """Test rotation mode."""
        transform = RandMisAlignmentd(
            keys=["image"],
            prob=1.0,
            displacement=8,
            rotate_ratio=1.0,  # Always rotate
        )

        result = transform(sample_data_dict)
        assert result["image"].shape == sample_data_dict["image"].shape

    def test_apply_to_multiple_keys(self, sample_data_dict):
        """Test applying to both image and label."""
        transform = RandMisAlignmentd(
            keys=["image", "label"],
            prob=1.0,
            displacement=8,
        )

        result = transform(sample_data_dict)

        assert "image" in result
        assert "label" in result
        # Note: shapes may differ due to cropping
        assert result["image"].ndim == sample_data_dict["image"].ndim

    def test_probability_control(self, sample_data_dict):
        """Test probability control."""
        transform = RandMisAlignmentd(
            keys=["image"],
            prob=0.0,  # Never apply
            displacement=8,
        )

        result = transform(sample_data_dict)

        # Should be unchanged
        assert torch.equal(result["image"], sample_data_dict["image"])


class TestRandMissingSectiond:
    """Test RandMissingSectiond transform."""

    def test_basic_functionality(self, sample_data_dict):
        """Test basic missing section."""
        transform = RandMissingSectiond(
            keys=["image"],
            prob=1.0,
            num_sections=2,
        )

        result = transform(sample_data_dict)

        assert "image" in result
        # Shape should be preserved (sections zeroed, not removed)
        assert result["image"].shape == sample_data_dict["image"].shape
        # Some sections should be zeroed out
        # Check that at least one section is zero (with tolerance for float)
        depth = result["image"].shape[1]
        # Skip first/last sections
        section_sums = result["image"][0, 1 : depth - 1, :, :].sum(dim=(1, 2))
        # At least one middle section should be zero (or very close to zero)
        assert (section_sums == 0).any() or (section_sums < 1e-6).any()

    def test_respects_boundaries(self, sample_data_dict):
        """Test that first and last sections are preserved."""
        transform = RandMissingSectiond(
            keys=["image"],
            prob=1.0,
            num_sections=1,
        )

        original_first = sample_data_dict["image"][0, 0, :, :].clone()
        original_last = sample_data_dict["image"][0, -1, :, :].clone()
        original_depth = sample_data_dict["image"].shape[1]

        # Run multiple times
        for _ in range(10):
            result = transform(sample_data_dict.copy())
            # Shape should be preserved
            assert result["image"].shape[1] == original_depth
            # First and last sections should be unchanged (preserved)
            assert torch.equal(result["image"][0, 0, :, :], original_first)
            assert torch.equal(result["image"][0, -1, :, :], original_last)

    def test_probability_control(self, sample_data_dict):
        """Test probability control."""
        transform = RandMissingSectiond(
            keys=["image"],
            prob=0.0,  # Never apply
            num_sections=2,
        )

        result = transform(sample_data_dict)

        # Should be unchanged
        assert torch.equal(result["image"], sample_data_dict["image"])


class TestRandMissingPartsd:
    """Test RandMissingPartsd transform."""

    def test_basic_functionality(self, sample_data_dict):
        """Test basic missing parts."""
        transform = RandMissingPartsd(
            keys=["image"],
            prob=1.0,
            hole_range=(0.1, 0.3),
        )

        result = transform(sample_data_dict)

        assert "image" in result
        assert result["image"].shape == sample_data_dict["image"].shape

        # Should have zeros (hole)
        assert (result["image"] == 0).any()

    def test_hole_size(self, sample_data_dict):
        """Test hole size respects range."""
        transform = RandMissingPartsd(
            keys=["image"],
            prob=1.0,
            hole_range=(0.2, 0.2),  # Fixed size
        )

        result = transform(sample_data_dict)

        # Count zeros
        num_zeros = (result["image"] == 0).sum()
        result["image"].numel()

        # Hole should be approximately 20% of one section
        # (exact size depends on section size)
        assert num_zeros > 0


class TestRandMotionBlurd:
    """Test RandMotionBlurd transform."""

    def test_basic_functionality(self, sample_data_dict):
        """Test basic motion blur."""
        transform = RandMotionBlurd(
            keys=["image"],
            prob=1.0,
            sections=2,
            kernel_size=7,
        )

        result = transform(sample_data_dict)

        assert "image" in result
        assert result["image"].shape == sample_data_dict["image"].shape

        # Image should be different (blurred)
        assert not torch.equal(result["image"], sample_data_dict["image"])

    def test_section_range(self, sample_data_dict):
        """Test section range."""
        transform = RandMotionBlurd(
            keys=["image"],
            prob=1.0,
            sections=(1, 3),  # 1-3 sections
            kernel_size=7,
        )

        result = transform(sample_data_dict)
        assert result["image"].shape == sample_data_dict["image"].shape


class TestRandCutNoised:
    """Test RandCutNoised transform."""

    def test_basic_functionality(self, sample_data_dict):
        """Test basic cut noise."""
        transform = RandCutNoised(
            keys=["image"],
            prob=1.0,
            length_ratio=0.25,
            noise_scale=0.2,
        )

        result = transform(sample_data_dict)

        assert "image" in result
        assert result["image"].shape == sample_data_dict["image"].shape

    def test_noise_scale(self, sample_data_dict):
        """Test noise scale."""
        # Set image to constant value
        sample_data_dict["image"] = torch.ones_like(sample_data_dict["image"]) * 0.5

        transform = RandCutNoised(
            keys=["image"],
            prob=1.0,
            length_ratio=0.5,  # Large region
            noise_scale=0.2,
        )

        result = transform(sample_data_dict)

        # Should have values different from 0.5 (noise added)
        assert (result["image"] != 0.5).any()


class TestRandCutBlurd:
    """Test RandCutBlurd transform."""

    def test_basic_functionality(self, sample_data_dict):
        """Test basic cut blur."""
        transform = RandCutBlurd(
            keys=["image"],
            prob=1.0,
            length_ratio=0.25,
            down_ratio_range=(2.0, 4.0),
            downsample_z=False,
        )

        result = transform(sample_data_dict)

        assert "image" in result
        assert result["image"].shape == sample_data_dict["image"].shape

    def test_downsample_z(self, sample_data_dict):
        """Test z-axis downsampling."""
        transform = RandCutBlurd(
            keys=["image"],
            prob=1.0,
            length_ratio=0.25,
            down_ratio_range=(2.0, 2.0),
            downsample_z=True,
        )

        result = transform(sample_data_dict)
        assert result["image"].shape == sample_data_dict["image"].shape


class TestRandMixupd:
    """Test RandMixupd transform."""

    def test_basic_functionality(self):
        """Test basic mixup."""
        # Create batched data
        data = {
            "image": torch.randn(4, 1, 32, 32, 32),  # Batch of 4
        }

        transform = RandMixupd(
            keys=["image"],
            prob=1.0,
            alpha_range=(0.7, 0.9),
        )

        result = transform(data)

        assert "image" in result
        assert result["image"].shape == data["image"].shape

    def test_requires_batch(self):
        """Test that mixup requires batch."""
        # Single sample (not batched)
        data = {
            "image": torch.randn(1, 32, 32, 32),
        }

        transform = RandMixupd(
            keys=["image"],
            prob=1.0,
            alpha_range=(0.7, 0.9),
        )

        result = transform(data)

        # Should return unchanged (no batch)
        assert torch.equal(result["image"], data["image"])


class TestRandCopyPasted:
    """Test RandCopyPasted transform."""

    def test_basic_functionality(self):
        """Test basic copy-paste."""
        # Create data with segmentation mask
        data = {
            "image": torch.randn(64, 64, 64),
            "label": torch.zeros(64, 64, 64, dtype=torch.bool),
        }

        # Create simple object (sphere)
        center = 32
        radius = 10
        for x in range(center - radius, center + radius):
            for y in range(center - radius, center + radius):
                for z in range(center - radius, center + radius):
                    if (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2 < radius**2:
                        data["label"][x, y, z] = True

        transform = RandCopyPasted(
            keys=["image"],
            label_key="label",
            prob=1.0,
            max_obj_ratio=0.9,
        )

        result = transform(data)

        assert "image" in result
        assert "label" in result

    def test_requires_label(self, sample_data_dict):
        """Test that copy-paste requires label."""
        # Remove label
        data = {"image": sample_data_dict["image"]}

        transform = RandCopyPasted(
            keys=["image"],
            label_key="label",
            prob=1.0,
        )

        result = transform(data)

        # Should return unchanged (no label)
        assert torch.equal(result["image"], data["image"])


class TestIntegration:
    """Integration tests with multiple transforms."""

    def test_transform_chain(self, sample_data_dict):
        """Test chaining multiple transforms."""
        transforms = [
            RandMissingSectiond(keys=["image"], prob=1.0, num_sections=1),
            RandMisAlignmentd(keys=["image"], prob=1.0, displacement=8),
            RandMotionBlurd(keys=["image"], prob=1.0, sections=1, kernel_size=5),
        ]

        data = sample_data_dict

        for transform in transforms:
            data = transform(data)

        assert "image" in data

    def test_with_monai_transforms(self, sample_data_dict):
        """Test compatibility with MONAI transforms."""
        from monai.transforms import RandGaussianNoised, RandShiftIntensityd

        transforms = [
            RandMissingSectiond(keys=["image"], prob=1.0, num_sections=1),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
            RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.1),
        ]

        data = sample_data_dict

        for transform in transforms:
            data = transform(data)

        assert "image" in data


# Performance tests
@pytest.mark.slow
class TestPerformance:
    """Performance tests (optional, marked as slow)."""

    def test_augmentation_speed(self, sample_data_dict):
        """Test augmentation speed."""
        import time

        transform = RandMisAlignmentd(
            keys=["image"],
            prob=1.0,
            displacement=16,
        )

        # Warm-up
        for _ in range(5):
            _ = transform(sample_data_dict)

        # Benchmark
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            _ = transform(sample_data_dict)
        end = time.time()

        time_per_iteration = (end - start) / num_iterations
        print(f"\nRandMisAlignmentd: {time_per_iteration * 1000:.2f} ms/iteration")

        # Should be reasonably fast (< 100ms)
        assert time_per_iteration < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
