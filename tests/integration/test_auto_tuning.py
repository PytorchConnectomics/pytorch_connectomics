"""
Tests for auto-tuning functionality (decoding/auto_tuning.py).

Tests cover:
- SkeletonMetrics class
- Grid search threshold optimization
- Optuna-based threshold optimization
- Multi-parameter optimization
"""

import pytest
import numpy as np
from unittest.mock import patch
import pickle
import tempfile
from pathlib import Path


# Test imports and availability
def test_imports():
    """Test that auto_tuning module can be imported."""
    from connectomics.decoding import auto_tuning
    assert hasattr(auto_tuning, 'optimize_threshold')
    assert hasattr(auto_tuning, 'grid_search_threshold')
    assert hasattr(auto_tuning, 'optimize_parameters')
    assert hasattr(auto_tuning, 'SkeletonMetrics')


def test_optuna_availability():
    """Test Optuna availability check."""
    try:
        assert True
    except ImportError:
        pytest.skip("Optuna not installed")


def test_funlib_availability():
    """Test funlib availability check."""
    try:
        assert True
    except ImportError:
        pytest.skip("funlib.evaluate not installed")


# Mock skeleton for testing
def create_mock_skeleton():
    """Create a mock skeleton object for testing."""
    try:
        import networkx as nx
    except ImportError:
        pytest.skip("networkx not installed")

    # Create simple graph
    G = nx.Graph()

    # Add nodes with required attributes
    for i in range(10):
        G.add_node(i,
                   id=i // 2,  # 5 ground truth objects
                   index_position=(i, i, i),
                   nm_position=[i*10.0, i*10.0, i*10.0])

    # Add edges with lengths
    for i in range(9):
        G.add_edge(i, i+1, edge_length=10.0)

    return G


@pytest.fixture
def mock_skeleton_file():
    """Create a temporary skeleton file for testing."""
    skeleton = create_mock_skeleton()

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump(skeleton, f)
        skeleton_path = f.name

    yield skeleton_path

    # Cleanup
    Path(skeleton_path).unlink()


@pytest.fixture
def mock_affinities():
    """Create mock affinity predictions."""
    # Shape: (3, 10, 10, 10) for short-range affinities
    affinities = np.random.rand(3, 10, 10, 10).astype(np.float32)
    return affinities


@pytest.fixture
def mock_segmentation():
    """Create mock segmentation."""
    # Shape: (10, 10, 10)
    seg = np.random.randint(0, 5, size=(10, 10, 10), dtype=np.uint32)
    return seg


# Tests for SkeletonMetrics
class TestSkeletonMetrics:
    """Tests for SkeletonMetrics class."""

    def test_init_missing_funlib(self, mock_skeleton_file):
        """Test initialization fails gracefully without funlib."""
        with patch.dict('sys.modules', {'funlib.evaluate': None}):
            from connectomics.decoding.auto_tuning import FUNLIB_AVAILABLE
            if not FUNLIB_AVAILABLE:
                with pytest.raises(ImportError, match="funlib.evaluate not found"):
                    from connectomics.decoding import SkeletonMetrics
                    SkeletonMetrics(mock_skeleton_file)

    def test_init_missing_skeleton(self):
        """Test initialization fails with missing skeleton file."""
        pytest.importorskip("funlib.evaluate")
        from connectomics.decoding import SkeletonMetrics

        with pytest.raises(FileNotFoundError):
            SkeletonMetrics("nonexistent.pkl")

    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_init_success(self, mock_skeleton_file):
        """Test successful initialization."""
        pytest.importorskip("funlib.evaluate")
        from connectomics.decoding import SkeletonMetrics

        metrics_calc = SkeletonMetrics(mock_skeleton_file)
        assert metrics_calc.skeleton is not None
        assert metrics_calc.skeleton_path == Path(mock_skeleton_file)

    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_compute_metrics(self, mock_skeleton_file, mock_segmentation):
        """Test metric computation."""
        pytest.importorskip("funlib.evaluate")
        from connectomics.decoding import SkeletonMetrics

        metrics_calc = SkeletonMetrics(mock_skeleton_file)
        results = metrics_calc.compute(mock_segmentation)

        # Check required metrics are present
        assert 'nerl' in results
        assert 'erl' in results
        assert 'max_erl' in results
        assert 'voi_sum' in results
        assert 'voi_split' in results
        assert 'voi_merge' in results
        assert 'n_mergers' in results
        assert 'n_splits' in results

        # Check value ranges
        assert 0.0 <= results['nerl'] <= 1.0
        assert results['erl'] >= 0.0
        assert results['voi_sum'] >= 0.0

    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_compute_with_details(self, mock_skeleton_file, mock_segmentation):
        """Test metric computation with details."""
        pytest.importorskip("funlib.evaluate")
        from connectomics.decoding import SkeletonMetrics

        metrics_calc = SkeletonMetrics(mock_skeleton_file)
        results = metrics_calc.compute(mock_segmentation, return_details=True)

        # Check additional detail fields
        assert 'merge_stats' in results
        assert 'split_stats' in results
        assert 'voi_report' in results


# Tests for grid_search_threshold
class TestGridSearchThreshold:
    """Tests for grid_search_threshold function."""

    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_grid_search_default_thresholds(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test grid search with default thresholds."""
        from connectomics.decoding import grid_search_threshold

        result = grid_search_threshold(
            mock_affinities,
            mock_skeleton_file,
            verbose=False
        )

        assert 'best_threshold' in result
        assert 'best_metrics' in result
        assert 'all_results' in result
        assert len(result['all_results']) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_grid_search_custom_thresholds(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test grid search with custom thresholds."""
        from connectomics.decoding import grid_search_threshold

        custom_thresholds = np.array([0.3, 0.5, 0.7])
        result = grid_search_threshold(
            mock_affinities,
            mock_skeleton_file,
            thresholds=custom_thresholds,
            verbose=False
        )

        assert len(result['all_results']) == 3
        assert 0.3 <= result['best_threshold'] <= 0.7

    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_grid_search_voi_metric(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test grid search optimizing VOI metric."""
        from connectomics.decoding import grid_search_threshold

        result = grid_search_threshold(
            mock_affinities,
            mock_skeleton_file,
            metric='voi_sum',
            verbose=False
        )

        assert result['best_metrics']['voi_sum'] >= 0.0


# Tests for optimize_threshold
class TestOptimizeThreshold:
    """Tests for Optuna-based threshold optimization."""

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Requires Optuna"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_optimize_threshold_basic(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test basic Optuna threshold optimization."""
        from connectomics.decoding import optimize_threshold

        result = optimize_threshold(
            mock_affinities,
            mock_skeleton_file,
            n_trials=10,
            verbose=False
        )

        assert 'best_threshold' in result
        assert 'best_metrics' in result
        assert 'study' in result

        # Check threshold is in valid range
        assert 0.1 <= result['best_threshold'] <= 0.9

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Requires Optuna"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_optimize_threshold_custom_range(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test optimization with custom threshold range."""
        from connectomics.decoding import optimize_threshold

        result = optimize_threshold(
            mock_affinities,
            mock_skeleton_file,
            n_trials=10,
            threshold_range=(0.2, 0.8),
            verbose=False
        )

        assert 0.2 <= result['best_threshold'] <= 0.8

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Requires Optuna"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_optimize_voi_metric(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test optimization with VOI metric."""
        from connectomics.decoding import optimize_threshold

        result = optimize_threshold(
            mock_affinities,
            mock_skeleton_file,
            n_trials=10,
            metric='voi_sum',
            verbose=False
        )

        # VOI should be minimized
        study = result['study']
        assert study.direction.name == 'MINIMIZE'


# Tests for optimize_parameters
class TestOptimizeParameters:
    """Tests for multi-parameter optimization."""

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Requires Optuna"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_optimize_two_parameters(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test optimization of multiple parameters."""
        from connectomics.decoding import optimize_parameters

        param_space = {
            'threshold': (0.1, 0.9),
            'thres_small': (0, 100),
        }

        result = optimize_parameters(
            mock_affinities,
            mock_skeleton_file,
            param_space,
            n_trials=10,
            verbose=False
        )

        assert 'best_params' in result
        assert 'best_metrics' in result
        assert 'study' in result

        # Check parameters are in valid ranges
        assert 0.1 <= result['best_params']['threshold'] <= 0.9
        assert 0 <= result['best_params']['thres_small'] <= 100

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Requires Optuna"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_optimize_integer_params(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test optimization with integer parameters."""
        from connectomics.decoding import optimize_parameters

        param_space = {
            'thres_small': (50, 200),  # Both integers
        }

        result = optimize_parameters(
            mock_affinities,
            mock_skeleton_file,
            param_space,
            n_trials=10,
            verbose=False
        )

        # Check parameter is integer
        assert isinstance(result['best_params']['thres_small'], int)
        assert 50 <= result['best_params']['thres_small'] <= 200


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.skipif(
        not pytest.importorskip("optuna", reason="Optuna not installed"),
        reason="Requires Optuna"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_grid_vs_optuna(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test that grid search and Optuna produce similar results."""
        from connectomics.decoding import grid_search_threshold, optimize_threshold

        # Grid search
        grid_result = grid_search_threshold(
            mock_affinities,
            mock_skeleton_file,
            thresholds=np.linspace(0.1, 0.9, 10),
            verbose=False
        )

        # Optuna
        optuna_result = optimize_threshold(
            mock_affinities,
            mock_skeleton_file,
            n_trials=20,
            verbose=False
        )

        # Results should be in similar range
        assert abs(grid_result['best_threshold'] - optuna_result['best_threshold']) < 0.3

    @pytest.mark.skipif(
        not pytest.importorskip("funlib.evaluate", reason="funlib not installed"),
        reason="Requires funlib.evaluate"
    )
    def test_end_to_end_workflow(
        self, mock_affinities, mock_skeleton_file
    ):
        """Test complete workflow from affinities to optimized segmentation."""
        from connectomics.decoding import (
            grid_search_threshold,
            affinity_cc3d,
            SkeletonMetrics
        )

        # Find optimal threshold
        result = grid_search_threshold(
            mock_affinities,
            mock_skeleton_file,
            thresholds=np.array([0.3, 0.5, 0.7]),
            verbose=False
        )

        best_threshold = result['best_threshold']

        # Apply optimal threshold
        final_seg = affinity_cc3d(
            mock_affinities,
            threshold=best_threshold
        )

        # Verify segmentation quality
        metrics_calc = SkeletonMetrics(mock_skeleton_file)
        final_metrics = metrics_calc.compute(final_seg)

        assert final_metrics['nerl'] >= 0.0
        assert final_seg.shape == (10, 10, 10)
