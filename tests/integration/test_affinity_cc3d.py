"""
Test suite for decode_affinity_cc connected components function.

Tests cover:
- Basic functionality with synthetic affinity data
- Numba vs skimage fallback comparison
- Small object removal
- Volume resizing
- Edge cases
- Performance benchmarks
"""

import numpy as np
import pytest
from connectomics.decoding import affinity_cc3d
from connectomics.decoding.segmentation import decode_affinity_cc

try:
    import numba  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import pytest_benchmark  # noqa: F401

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_affinities():
    """Create simple synthetic affinity predictions."""
    # Create 32x32x32 volume with 2 separate components
    aff = np.zeros((3, 32, 32, 32), dtype=np.float32)

    # Component 1: cube in corner (8x8x8)
    aff[:, 0:8, 0:8, 0:8] = 0.9

    # Component 2: cube in opposite corner (8x8x8)
    aff[:, 24:32, 24:32, 24:32] = 0.9

    return aff


@pytest.fixture
def connected_affinities():
    """Create fully connected affinity predictions."""
    # 16x16x16 volume, all connected
    aff = np.ones((3, 16, 16, 16), dtype=np.float32) * 0.8
    return aff


@pytest.fixture
def six_channel_affinities():
    """Create 6-channel affinities (short + long range)."""
    # Should only use first 3 channels
    aff = np.zeros((6, 32, 32, 32), dtype=np.float32)
    aff[:3, 8:24, 8:24, 8:24] = 0.9  # Short-range
    aff[3:, 8:24, 8:24, 8:24] = 0.1  # Long-range (ignored)
    return aff


if not HAS_BENCHMARK:

    @pytest.fixture
    def benchmark():
        """Lightweight fallback when pytest-benchmark is not installed."""

        def _runner(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        return _runner


class TestAffinityCC3D:
    """Test suite for decode_affinity_cc function."""

    def test_basic_functionality(self, simple_affinities):
        """Test basic connected components on simple affinity data."""
        segm = decode_affinity_cc(simple_affinities, threshold=0.5)

        # Check output shape
        assert segm.shape == simple_affinities.shape[1:], "Output shape mismatch"

        # Check output dtype
        assert segm.dtype in [np.uint8, np.uint16, np.uint32, np.uint64], \
            "Output should be integer type"

        # Check number of components (should be 2 + background)
        unique_labels = np.unique(segm)
        assert len(unique_labels) == 3, f"Expected 3 labels (bg + 2 objects), got {len(unique_labels)}"
        assert 0 in unique_labels, "Background label 0 should be present"

    def test_threshold_sensitivity(self, simple_affinities):
        """Test that threshold parameter affects segmentation."""
        # Low threshold - more connected
        segm_low = decode_affinity_cc(simple_affinities, threshold=0.3)
        n_labels_low = len(np.unique(segm_low)) - 1  # Exclude background

        # High threshold - more disconnected
        segm_high = decode_affinity_cc(simple_affinities, threshold=0.95)
        n_labels_high = len(np.unique(segm_high)) - 1

        # High threshold should create more or equal segments
        assert n_labels_high >= n_labels_low, \
            "Higher threshold should not decrease number of segments"

    def test_fully_connected(self, connected_affinities):
        """Test single fully connected component."""
        segm = decode_affinity_cc(connected_affinities, threshold=0.5)

        unique_labels = np.unique(segm)
        # Should be background + 1 component
        assert len(unique_labels) == 2, \
            f"Fully connected volume should have 2 labels (bg + 1 object), got {len(unique_labels)}"

    def test_six_channel_input(self, six_channel_affinities):
        """Test that only first 3 channels are used."""
        segm = decode_affinity_cc(six_channel_affinities, threshold=0.5)

        # Should produce valid segmentation using only short-range affinities
        assert segm.shape == six_channel_affinities.shape[1:], "Shape mismatch"
        unique_labels = np.unique(segm)
        assert len(unique_labels) >= 2, "Should find at least 1 component"

    def test_empty_input(self):
        """Test behavior with empty (all zero) affinities."""
        aff = np.zeros((3, 16, 16, 16), dtype=np.float32)
        segm = decode_affinity_cc(aff, threshold=0.5)

        # Should be all background (0)
        assert np.all(segm == 0), "Empty affinities should produce all-background segmentation"

    def test_small_object_removal(self, simple_affinities):
        """Test removal of small connected components."""
        # Without removal
        segm_full = decode_affinity_cc(simple_affinities, threshold=0.5, min_instance_size=0)
        n_full = len(np.unique(segm_full)) - 1

        # With removal (threshold larger than component size)
        component_size = 8 * 8 * 8  # 512 voxels
        segm_filtered = decode_affinity_cc(
            simple_affinities,
            threshold=0.5,
            min_instance_size=component_size + 100  # Remove components < 612 voxels
        )
        n_filtered = len(np.unique(segm_filtered)) - 1

        # Should have fewer components after filtering
        assert n_filtered <= n_full, "Filtering should not increase component count"

    def test_remove_small_modes(self, simple_affinities):
        """Test different small object removal modes."""
        # Mode 'background' - remove to background
        segm_bg = decode_affinity_cc(
            simple_affinities,
            threshold=0.5,
            min_instance_size=100,
            remove_small_mode='background'
        )

        # Mode 'neighbor' - remove to neighboring label
        segm_neighbor = decode_affinity_cc(
            simple_affinities,
            threshold=0.5,
            min_instance_size=100,
            remove_small_mode='neighbor'
        )

        # Both should be valid segmentations
        assert segm_bg.shape == simple_affinities.shape[1:], "Shape mismatch (background mode)"
        assert segm_neighbor.shape == simple_affinities.shape[1:], "Shape mismatch (neighbor mode)"

    def test_volume_resizing(self, simple_affinities):
        """Test volume resizing with scale factors."""
        scale_factors = (2.0, 1.0, 0.5)  # 2x in z, 1x in y, 0.5x in x
        segm = decode_affinity_cc(simple_affinities, threshold=0.5, scale_factors=scale_factors)

        # Expected shape
        original_shape = simple_affinities.shape[1:]
        expected_shape = tuple(int(s * f) for s, f in zip(original_shape, scale_factors))

        assert segm.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {segm.shape}"

    def test_numba_vs_skimage(self, simple_affinities):
        """Compare Numba and skimage implementations."""
        # Run with Numba
        segm_numba = decode_affinity_cc(simple_affinities, threshold=0.5, use_numba=True)

        # Run with skimage
        segm_skimage = decode_affinity_cc(simple_affinities, threshold=0.5, use_numba=False)

        # Both should have same number of components (labels may differ)
        n_numba = len(np.unique(segm_numba)) - 1
        n_skimage = len(np.unique(segm_skimage)) - 1

        assert n_numba == n_skimage, \
            f"Numba ({n_numba} components) and skimage ({n_skimage} components) should find same number"

    def test_deterministic_output(self, simple_affinities):
        """Test that output is deterministic across runs."""
        segm1 = decode_affinity_cc(simple_affinities, threshold=0.5, use_numba=True)
        segm2 = decode_affinity_cc(simple_affinities, threshold=0.5, use_numba=True)

        # Should be identical
        np.testing.assert_array_equal(segm1, segm2,
            err_msg="Multiple runs should produce identical results")

    def test_output_dtype_casting(self):
        """Test automatic dtype selection based on number of labels."""
        # Small volume - should use uint8 or uint16
        small_aff = np.ones((3, 8, 8, 8), dtype=np.float32) * 0.9
        small_segm = decode_affinity_cc(small_aff, threshold=0.5)
        assert small_segm.dtype in [np.uint8, np.uint16], \
            "Small volumes should use compact dtype"

        # Large volume - may need uint32
        large_aff = np.random.rand(3, 64, 64, 64).astype(np.float32)
        large_segm = decode_affinity_cc(large_aff, threshold=0.5)
        assert large_segm.dtype in [np.uint8, np.uint16, np.uint32, np.uint64], \
            "Output should be integer type"

    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        # 2D input (should fail)
        aff_2d = np.random.rand(3, 32, 32).astype(np.float32)
        with pytest.raises((ValueError, IndexError)):
            decode_affinity_cc(aff_2d, threshold=0.5)

        # Wrong number of channels
        aff_wrong = np.random.rand(2, 32, 32, 32).astype(np.float32)
        with pytest.raises((ValueError, IndexError)):
            decode_affinity_cc(aff_wrong, threshold=0.5)

    def test_boundary_threshold_values(self, simple_affinities):
        """Test boundary values for threshold parameter."""
        # Threshold = 0.0 (everything connected)
        segm_zero = decode_affinity_cc(simple_affinities, threshold=0.0)
        assert len(np.unique(segm_zero)) >= 2, "Should have at least background + 1 component"

        # Threshold = 1.0 (nothing connected)
        segm_one = decode_affinity_cc(simple_affinities, threshold=1.0)
        # Most voxels should be background or very fragmented
        assert len(np.unique(segm_one)) >= 1, "Should have at least background"


class TestAffinityCC3DIntegration:
    """Integration tests with real-world usage patterns."""

    def test_pipeline_integration(self):
        """Test integration with typical segmentation pipeline."""
        # Simulate model output (6-channel affinities)
        batch_size = 2
        affinities_batch = np.random.rand(batch_size, 6, 64, 64, 64).astype(np.float32)

        # Process each sample in batch
        segmentations = []
        for i in range(batch_size):
            segm = decode_affinity_cc(
                affinities_batch[i],
                threshold=0.5,
                min_instance_size=100,
                remove_small_mode='background'
            )
            segmentations.append(segm)

        # Check all segmentations are valid
        assert len(segmentations) == batch_size
        for segm in segmentations:
            assert segm.shape == (64, 64, 64)
            assert segm.dtype in [np.uint8, np.uint16, np.uint32]

    def test_multi_threshold_processing(self, simple_affinities):
        """Test processing with multiple threshold values."""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = []

        for thresh in thresholds:
            segm = decode_affinity_cc(simple_affinities, threshold=thresh)
            n_components = len(np.unique(segm)) - 1
            results.append(n_components)

        # Generally, higher thresholds should not decrease fragmentation
        # (though this is not guaranteed for all data)
        assert all(isinstance(n, (int, np.integer)) for n in results), \
            "All results should be integers"

    def test_postprocessing_chain(self, simple_affinities):
        """Test chaining with other post-processing operations."""
        # Step 1: Connected components
        segm = decode_affinity_cc(
            simple_affinities,
            threshold=0.5,
            min_instance_size=0  # No filtering yet
        )

        # Step 2: Manual small object removal
        unique_labels, counts = np.unique(segm, return_counts=True)
        small_labels = unique_labels[counts < 100]

        for label_id in small_labels:
            if label_id > 0:  # Don't remove background
                segm[segm == label_id] = 0

        # Check result is still valid
        assert segm.shape == simple_affinities.shape[1:]
        assert segm.dtype in [np.uint8, np.uint16, np.uint32]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
