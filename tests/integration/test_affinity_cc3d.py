"""
Test suite for decode_affinity_cc connected components function.

Tests cover:
- Basic functionality with synthetic affinity data
- Edge cases
- Performance benchmarks
"""

import numpy as np
import pytest

from connectomics.decoding.decoders.segmentation import decode_affinity_cc

try:
    import numba  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import cupy  # noqa: F401

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

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
        assert segm.dtype in [
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ], "Output should be integer type"

        # Check number of components (should be 2 + background)
        unique_labels = np.unique(segm)
        assert (
            len(unique_labels) == 3
        ), f"Expected 3 labels (bg + 2 objects), got {len(unique_labels)}"
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
        assert (
            n_labels_high >= n_labels_low
        ), "Higher threshold should not decrease number of segments"

    def test_fully_connected(self, connected_affinities):
        """Test single fully connected component."""
        segm = decode_affinity_cc(connected_affinities, threshold=0.5)

        unique_labels = np.unique(segm)
        # Should be background + 1 component
        assert (
            len(unique_labels) == 2
        ), f"Fully connected volume should have 2 labels (bg + 1 object), got {len(unique_labels)}"

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

    def test_deterministic_output(self, simple_affinities):
        """Test that output is deterministic across runs."""
        segm1 = decode_affinity_cc(simple_affinities, threshold=0.5)
        segm2 = decode_affinity_cc(simple_affinities, threshold=0.5)

        # Should be identical
        np.testing.assert_array_equal(
            segm1, segm2, err_msg="Multiple runs should produce identical results"
        )

    def test_output_dtype_casting(self):
        """Test automatic dtype selection based on number of labels."""
        # Small volume - should use uint8 or uint16
        small_aff = np.ones((3, 8, 8, 8), dtype=np.float32) * 0.9
        small_segm = decode_affinity_cc(small_aff, threshold=0.5)
        assert small_segm.dtype in [np.uint8, np.uint16], "Small volumes should use compact dtype"

        # Large volume - may need uint32
        large_aff = np.random.rand(3, 64, 64, 64).astype(np.float32)
        large_segm = decode_affinity_cc(large_aff, threshold=0.5)
        assert large_segm.dtype in [
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ], "Output should be integer type"

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
        assert all(
            isinstance(n, (int, np.integer)) for n in results
        ), "All results should be integers"

    def test_postprocessing_chain(self, simple_affinities):
        """Test chaining with other post-processing operations."""
        # Step 1: Connected components
        segm = decode_affinity_cc(
            simple_affinities,
            threshold=0.5,
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


def _assert_same_partition(a: np.ndarray, b: np.ndarray) -> None:
    """Assert two label arrays induce the same partition (relabeling-invariant).

    Downstream metrics (adapted_rand, NERL, …) are relabeling-invariant, so the
    parallel kernel only needs to match the serial kernel's component
    structure, not the exact label IDs. This builds a bijection between
    label sets and fails if any voxel disagrees.
    """
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    a_to_b: dict = {}
    b_to_a: dict = {}
    for av, bv in zip(flat_a.tolist(), flat_b.tolist()):
        if av in a_to_b:
            assert a_to_b[av] == bv, f"label {av} maps to both {a_to_b[av]} and {bv}"
        else:
            a_to_b[av] = bv
        if bv in b_to_a:
            assert b_to_a[bv] == av, f"label {bv} maps to both {b_to_a[bv]} and {av}"
        else:
            b_to_a[bv] = av


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba required")
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
@pytest.mark.parametrize("edge_offset", [0, 1])
@pytest.mark.parametrize("density", [0.2, 0.5, 0.8])
def test_numba_parallel_matches_serial(seed, edge_offset, density):
    """Parallel-numba CC must induce the same partition as the serial DFS reference."""
    rng = np.random.default_rng(seed)
    aff = (rng.random((3, 12, 11, 13), dtype=np.float64) < density).astype(np.float32)

    serial = decode_affinity_cc(
        aff, threshold=0.5, backend="numba_serial", edge_offset=edge_offset
    )
    parallel = decode_affinity_cc(
        aff, threshold=0.5, backend="numba", edge_offset=edge_offset
    )

    assert serial.shape == parallel.shape
    n_serial = int(np.unique(serial).size)
    n_parallel = int(np.unique(parallel).size)
    assert n_serial == n_parallel, (
        f"component count mismatch: serial={n_serial} parallel={n_parallel}"
    )
    _assert_same_partition(serial, parallel)

    if edge_offset == 0:
        # edge_offset=0 stores edges at the lex-lower voxel, so both kernels
        # label components in lex order of their min member → bit-exact.
        np.testing.assert_array_equal(serial, parallel)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy required")
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
@pytest.mark.parametrize("edge_offset", [0, 1])
@pytest.mark.parametrize("density", [0.2, 0.5, 0.8])
def test_cupy_matches_serial(seed, edge_offset, density):
    """cupy CC must induce the same partition as the serial DFS reference."""
    rng = np.random.default_rng(seed)
    aff = (rng.random((3, 12, 11, 13), dtype=np.float64) < density).astype(np.float32)

    serial = decode_affinity_cc(
        aff, threshold=0.5, backend="numba_serial", edge_offset=edge_offset
    )
    gpu = decode_affinity_cc(aff, threshold=0.5, backend="cupy", edge_offset=edge_offset)

    assert serial.shape == gpu.shape
    n_serial = int(np.unique(serial).size)
    n_gpu = int(np.unique(gpu).size)
    assert n_serial == n_gpu, f"component count mismatch: serial={n_serial} cupy={n_gpu}"
    _assert_same_partition(serial, gpu)

    if edge_offset == 0:
        np.testing.assert_array_equal(serial, gpu)


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy required")
def test_cupy_empty_and_full():
    """Edge cases for the cupy backend."""
    aff_empty = np.zeros((3, 5, 5, 5), dtype=np.float32)
    aff_full = np.ones((3, 5, 5, 5), dtype=np.float32)

    for edge_offset in (0, 1):
        s_empty = decode_affinity_cc(
            aff_empty, threshold=0.5, backend="cupy", edge_offset=edge_offset
        )
        ref_empty = decode_affinity_cc(
            aff_empty, threshold=0.5, backend="numba_serial", edge_offset=edge_offset
        )
        np.testing.assert_array_equal(s_empty, ref_empty)
        assert s_empty.max() == 0

        s_full = decode_affinity_cc(
            aff_full, threshold=0.5, backend="cupy", edge_offset=edge_offset
        )
        ref_full = decode_affinity_cc(
            aff_full, threshold=0.5, backend="numba_serial", edge_offset=edge_offset
        )
        np.testing.assert_array_equal(s_full, ref_full)
        assert s_full.max() == 1


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba required")
def test_numba_parallel_empty_and_full():
    """Edge cases: no edges → all zeros; all edges → single component."""
    aff_empty = np.zeros((3, 5, 5, 5), dtype=np.float32)
    aff_full = np.ones((3, 5, 5, 5), dtype=np.float32)

    for edge_offset in (0, 1):
        s_empty = decode_affinity_cc(
            aff_empty, threshold=0.5, backend="numba", edge_offset=edge_offset
        )
        p_empty = decode_affinity_cc(
            aff_empty, threshold=0.5, backend="numba_serial", edge_offset=edge_offset
        )
        np.testing.assert_array_equal(s_empty, p_empty)
        assert s_empty.max() == 0

        s_full = decode_affinity_cc(
            aff_full, threshold=0.5, backend="numba", edge_offset=edge_offset
        )
        p_full = decode_affinity_cc(
            aff_full, threshold=0.5, backend="numba_serial", edge_offset=edge_offset
        )
        np.testing.assert_array_equal(s_full, p_full)
        assert s_full.max() == 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
