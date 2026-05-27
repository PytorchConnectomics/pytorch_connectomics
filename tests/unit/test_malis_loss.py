"""Tests for MalisLoss wrapper around lib/malis."""

from contextlib import contextmanager
import unittest

import numpy as np
import torch

try:
    import malis as _malis_lib  # noqa: F401

    _HAS_MALIS = True
except Exception:
    _HAS_MALIS = False


def _reference_malis_pass(
    pred_affs: np.ndarray,
    gt_affs: np.ndarray,
    gt_seg: np.ndarray,
    nhood: np.ndarray,
    mask: np.ndarray | None,
    pos: int,
) -> np.ndarray:
    pass_affs = np.array(pred_affs, copy=True)
    class_edges = gt_affs == (1 - pos)
    if mask is None:
        constraint_edges = class_edges
    else:
        mask_b = np.ascontiguousarray(mask == 1, dtype=bool)
        constraint_edges = class_edges & mask_b
    pass_affs[constraint_edges] = 1 - pos

    node1, node2 = _malis_lib.nodelist_like(gt_seg.shape, nhood)
    weights = _malis_lib.malis_loss_weights(
        np.ascontiguousarray(gt_seg, dtype=np.uint64).ravel(),
        np.ascontiguousarray(node1, dtype=np.uint64).ravel(),
        np.ascontiguousarray(node2, dtype=np.uint64).ravel(),
        np.ascontiguousarray(pass_affs, dtype=np.float32).ravel(),
        pos,
    )
    weights = weights.reshape((-1,) + tuple(gt_seg.shape)).astype(np.float32, copy=False)
    weights[class_edges] = 0.0
    if mask is not None:
        weights[~mask_b] = 0.0

    num_pairs = weights.sum(dtype=np.float64)
    if num_pairs > 0:
        weights = weights / np.float32(num_pairs)
    return weights


def _reference_malis_weights(
    pred_affs: np.ndarray,
    target_affs: np.ndarray,
    nhood: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    gt_affs = np.ascontiguousarray(target_affs > 0.5, dtype=np.int32)
    gt_seg, _ = _malis_lib.connected_components_affgraph(gt_affs, nhood)
    weights_neg = _reference_malis_pass(pred_affs, gt_affs, gt_seg, nhood, mask, pos=0)
    weights_pos = _reference_malis_pass(pred_affs, gt_affs, gt_seg, nhood, mask, pos=1)
    return weights_neg + weights_pos


def _reference_malis_loss(
    pred_affs: np.ndarray,
    target_affs: np.ndarray,
    nhood: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.float32:
    weights = _reference_malis_weights(pred_affs, target_affs, nhood, mask)
    sqerr = np.square((pred_affs - target_affs).astype(np.float32, copy=False))
    return np.sum(weights * sqerr, dtype=np.float32)


@contextmanager
def _fixed_torch_randint(value: int):
    original_randint = torch.randint

    def _fake_randint(*args, **kwargs):
        return torch.tensor([value], dtype=torch.long)

    torch.randint = _fake_randint
    try:
        yield
    finally:
        torch.randint = original_randint


@unittest.skipUnless(_HAS_MALIS, "malis extension not built")
class TestMalisLoss(unittest.TestCase):
    def _crop_loss_shape(self, crop_size):
        from connectomics.models.losses.malis import MalisLoss

        loss_fn = MalisLoss(malis_crop_size=crop_size, reduction="none", sigmoid=False)
        C = loss_fn.nhood.shape[0]
        pred = torch.zeros(1, C, 8, 8, 8)
        target = torch.zeros_like(pred)
        loss_fn._compute_malis_weights = lambda pred_aff, target_aff, mask=None: torch.ones_like(
            pred_aff
        )
        return loss_fn(pred, target).shape, loss_fn.malis_crop_size, C

    def test_create_via_registry(self):
        from connectomics.models.losses import create_loss
        from connectomics.models.losses.malis import MalisLoss

        loss_fn = create_loss("MalisLoss")
        meta = getattr(loss_fn, "_connectomics_loss_metadata", None)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.name, "MalisLoss")
        self.assertEqual(meta.spatial_weight_arg, "mask")
        self.assertIsInstance(loss_fn, MalisLoss)

    def test_not_in_package_namespace(self):
        """MalisLoss is intentionally not re-exported from the package __init__."""
        import connectomics.models.losses as _pkg

        self.assertFalse(hasattr(_pkg, "MalisLoss"))

    def test_forward_simple_3d(self):
        from connectomics.models.losses.malis import MalisLoss

        torch.manual_seed(0)
        loss_fn = MalisLoss()
        nhood = loss_fn.nhood
        C = nhood.shape[0]
        pred = torch.randn(1, C, 6, 8, 8, requires_grad=True)
        gt_seg = np.zeros((6, 8, 8), dtype=np.int32)
        gt_seg[:4, :, :] = 1
        aff = _malis_lib.seg_to_affgraph(gt_seg, nhood).astype(np.float32)
        target = torch.from_numpy(aff).unsqueeze(0)
        loss = loss_fn(pred, target)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all().item())

    def test_loss_matches_numpy_reference_with_and_without_mask(self):
        from connectomics.models.losses.malis import MalisLoss

        loss_fn = MalisLoss(sigmoid=False)
        nhood = loss_fn.nhood
        C = nhood.shape[0]
        pred_np = np.linspace(0.05, 0.95, num=C * 3 * 4 * 5, dtype=np.float32).reshape(
            C,
            3,
            4,
            5,
        )
        gt_seg = np.zeros((3, 4, 5), dtype=np.int32)
        gt_seg[:, :2, :] = 1
        gt_seg[:, 2:, :] = 2
        target_np = _malis_lib.seg_to_affgraph(gt_seg, nhood).astype(np.float32)
        mask_np = np.ones((1, 3, 4, 5), dtype=np.float32)
        mask_np[:, :, 1:3, :] = 0.0

        pred = torch.from_numpy(pred_np).unsqueeze(0)
        target = torch.from_numpy(target_np).unsqueeze(0)
        mask = torch.from_numpy(mask_np).unsqueeze(0)

        expected_unmasked = _reference_malis_loss(pred_np, target_np, nhood)
        expected_masked = _reference_malis_loss(
            pred_np,
            target_np,
            nhood,
            np.broadcast_to(mask_np, target_np.shape),
        )

        torch.testing.assert_close(
            loss_fn(pred, target),
            torch.tensor(expected_unmasked),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            loss_fn(pred, target, mask=mask),
            torch.tensor(expected_masked),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_reduction_default_and_supported_modes(self):
        from connectomics.models.losses.malis import MalisLoss

        base_loss = MalisLoss(sigmoid=False)
        C = base_loss.nhood.shape[0]
        pred = torch.linspace(0.0, 0.9, steps=C * 2 * 2 * 2, dtype=torch.float32).reshape(
            1,
            C,
            2,
            2,
            2,
        )
        target = torch.zeros_like(pred)
        weights = torch.linspace(0.1, 1.0, steps=pred.numel(), dtype=torch.float32)
        weights = weights.reshape_as(pred)
        expected = weights * pred.square()

        for reduction, expected_value in (
            ("sum", expected.sum()),
            ("mean", expected.mean()),
            ("none", expected),
        ):
            if reduction == "sum":
                loss_fn = base_loss
            else:
                loss_fn = MalisLoss(reduction=reduction, sigmoid=False)
            loss_fn._compute_malis_weights = lambda pred_aff, target_aff, mask=None: weights
            torch.testing.assert_close(loss_fn(pred, target), expected_value)

    def test_mask_does_not_change_topology(self):
        """Regression: mask must NOT influence the CC reconstruction.

        Strategy: monkey-patch malis.connected_components_affgraph to record
        the int32 affinity array it is called with, then run forward() once
        without mask and once with a topology-splitting mask. Both calls must
        receive an identical CC input (i.e. mask is not folded in before CC).
        """
        from connectomics.models.losses import malis as malis_mod
        from connectomics.models.losses.malis import MalisLoss

        torch.manual_seed(0)
        loss_fn = MalisLoss()
        nhood = loss_fn.nhood
        C = nhood.shape[0]

        gt_seg = np.zeros((6, 8, 8), dtype=np.int32)
        gt_seg[:4, :, :] = 1
        aff = _malis_lib.seg_to_affgraph(gt_seg, nhood).astype(np.float32)
        target = torch.from_numpy(aff).unsqueeze(0)
        pred = torch.randn(1, C, 6, 8, 8)

        recorded: list[np.ndarray] = []
        real_cc = _malis_lib.connected_components_affgraph

        def _spy(aff_in, nh_in):
            recorded.append(np.array(aff_in, copy=True))
            return real_cc(aff_in, nh_in)

        # Patch at the lookup site used inside malis.py (it references the
        # module-level `_malis_lib`).
        orig = malis_mod._malis_lib.connected_components_affgraph
        malis_mod._malis_lib.connected_components_affgraph = _spy
        try:
            _ = loss_fn(pred, target)
            mask = torch.ones(1, 1, 6, 8, 8)
            mask[:, :, 1:3, :, :] = 0
            _ = loss_fn(pred, target, mask=mask)
        finally:
            malis_mod._malis_lib.connected_components_affgraph = orig

        self.assertEqual(len(recorded), 2)
        # The CC INPUT array must be identical between the two calls:
        # mask must not be folded in before CC.
        np.testing.assert_array_equal(recorded[0], recorded[1])

    def test_malis_crop_size_int_shape(self):
        shape, crop_size, C = self._crop_loss_shape(4)

        self.assertEqual(shape, (1, C, 4, 4, 4))
        self.assertEqual(crop_size, (4, 4, 4))

    def test_malis_crop_size_tuple_shape(self):
        shape, crop_size, C = self._crop_loss_shape((2, 4, 4))

        self.assertEqual(shape, (1, C, 2, 4, 4))
        self.assertEqual(crop_size, (2, 4, 4))

    def test_malis_crop_size_list_shape(self):
        shape, crop_size, C = self._crop_loss_shape([2, 4, 4])

        self.assertEqual(shape, (1, C, 2, 4, 4))
        self.assertEqual(crop_size, (2, 4, 4))

    def test_malis_crop_size_omegaconf_list_shape(self):
        try:
            from omegaconf import OmegaConf
        except Exception as e:
            self.skipTest(f"omegaconf not importable: {e}")

        shape, crop_size, C = self._crop_loss_shape(OmegaConf.create([2, 4, 4]))

        self.assertEqual(shape, (1, C, 2, 4, 4))
        self.assertEqual(crop_size, (2, 4, 4))

    def test_malis_crop_size_none_preserves_shape(self):
        shape, crop_size, C = self._crop_loss_shape(None)

        self.assertEqual(shape, (1, C, 8, 8, 8))
        self.assertIsNone(crop_size)

    def test_malis_crop_size_gradient_flows_with_fixed_origin(self):
        from connectomics.models.losses.malis import MalisLoss

        loss_fn = MalisLoss(malis_crop_size=4, sigmoid=False)
        C = loss_fn.nhood.shape[0]
        pred = torch.ones(1, C, 8, 8, 8, requires_grad=True)
        target = torch.zeros_like(pred)
        loss_fn._compute_malis_weights = lambda pred_aff, target_aff, mask=None: torch.ones_like(
            pred_aff
        )

        with _fixed_torch_randint(2):
            loss = loss_fn(pred, target)
        loss.backward()

        grad = pred.grad
        self.assertIsNotNone(grad)
        self.assertEqual(grad[..., :2, :, :].abs().sum().item(), 0.0)
        self.assertEqual(grad[..., 6:, :, :].abs().sum().item(), 0.0)
        self.assertEqual(grad[..., :, :2, :].abs().sum().item(), 0.0)
        self.assertEqual(grad[..., :, 6:, :].abs().sum().item(), 0.0)
        self.assertEqual(grad[..., :, :, :2].abs().sum().item(), 0.0)
        self.assertEqual(grad[..., :, :, 6:].abs().sum().item(), 0.0)
        self.assertGreater(grad[..., 2:6, 2:6, 2:6].abs().sum().item(), 0.0)

    def test_malis_crop_size_validation_construction(self):
        from connectomics.models.losses.malis import MalisLoss

        for bad_value in (0, -1, (2, 4), (2, 4, 5, 6), "foo", (2, 0, 4)):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ValueError):
                    MalisLoss(malis_crop_size=bad_value)
        for bad_value in (1.5, True, (True, 4, 4), [1.5, 2, 3]):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ValueError):
                    MalisLoss(malis_crop_size=bad_value)

    def test_malis_crop_size_too_large_at_forward(self):
        from connectomics.models.losses.malis import MalisLoss

        loss_fn = MalisLoss(malis_crop_size=99, sigmoid=False)
        C = loss_fn.nhood.shape[0]
        pred = torch.zeros(1, C, 4, 4, 4)
        target = torch.zeros_like(pred)

        with self.assertRaises(ValueError):
            loss_fn(pred, target)

    def test_malis_crop_does_not_break_mask_topology(self):
        from connectomics.models.losses import malis as malis_mod
        from connectomics.models.losses.malis import MalisLoss

        torch.manual_seed(0)
        loss_fn = MalisLoss(malis_crop_size=4)
        nhood = loss_fn.nhood
        C = nhood.shape[0]

        gt_seg = np.zeros((8, 8, 8), dtype=np.int32)
        gt_seg[:4, :, :] = 1
        aff = _malis_lib.seg_to_affgraph(gt_seg, nhood).astype(np.float32)
        target = torch.from_numpy(aff).unsqueeze(0)
        pred = torch.randn(1, C, 8, 8, 8)

        recorded: list[np.ndarray] = []
        real_cc = _malis_lib.connected_components_affgraph

        def _spy(aff_in, nh_in):
            recorded.append(np.array(aff_in, copy=True))
            return real_cc(aff_in, nh_in)

        orig_cc = malis_mod._malis_lib.connected_components_affgraph
        malis_mod._malis_lib.connected_components_affgraph = _spy
        try:
            with _fixed_torch_randint(2):
                _ = loss_fn(pred, target)
                mask = torch.ones(1, 1, 8, 8, 8)
                mask[:, :, 2:4, :, :] = 0
                _ = loss_fn(pred, target, mask=mask)
        finally:
            malis_mod._malis_lib.connected_components_affgraph = orig_cc

        self.assertEqual(len(recorded), 2)
        np.testing.assert_array_equal(recorded[0], recorded[1])

    def test_malis_crop_edge_list_cache_hit(self):
        from connectomics.models.losses import malis as malis_mod
        from connectomics.models.losses.malis import MalisLoss

        loss_fn = MalisLoss(malis_crop_size=4, sigmoid=False)
        C = loss_fn.nhood.shape[0]
        pred = torch.rand(1, C, 8, 8, 8)
        target = torch.zeros_like(pred)

        call_count = 0
        real_nodelist_like = _malis_lib.nodelist_like

        def _spy(shape, nhood):
            nonlocal call_count
            call_count += 1
            return real_nodelist_like(shape, nhood)

        orig = malis_mod._malis_lib.nodelist_like
        malis_mod._malis_lib.nodelist_like = _spy
        try:
            _ = loss_fn(pred, target)
            _ = loss_fn(pred, target)
        finally:
            malis_mod._malis_lib.nodelist_like = orig

        self.assertEqual(call_count, 1)

    def test_malis_crop_contiguous_output_path(self):
        from connectomics.models.losses.malis import MalisLoss

        torch.manual_seed(0)
        loss_fn = MalisLoss(malis_crop_size=4, sigmoid=False)
        C = loss_fn.nhood.shape[0]
        pred = torch.rand(1, C, 8, 8, 8, dtype=torch.float32)
        target = torch.zeros_like(pred)

        loss = loss_fn(pred, target)

        self.assertTrue(torch.isfinite(loss).item())

    def test_shape_mismatch_raises(self):
        from connectomics.models.losses.malis import MalisLoss

        loss_fn = MalisLoss()
        pred = torch.randn(1, loss_fn.nhood.shape[0], 4, 4, 4)
        target = torch.zeros(1, loss_fn.nhood.shape[0], 4, 4, 5)
        with self.assertRaises(ValueError):
            loss_fn(pred, target)

    def test_2d_not_supported(self):
        from connectomics.models.losses.malis import MalisLoss

        loss_fn = MalisLoss()
        pred = torch.randn(1, loss_fn.nhood.shape[0], 4, 4)
        target = torch.zeros_like(pred)
        with self.assertRaises(NotImplementedError):
            loss_fn(pred, target)


class TestMalisLossImportFallback(unittest.TestCase):
    @unittest.skipIf(_HAS_MALIS, "malis is available; cannot test ImportError path")
    def test_import_error_when_extension_missing(self):
        from connectomics.models.losses.malis import MalisLoss

        with self.assertRaises(ImportError):
            MalisLoss()


if __name__ == "__main__":
    unittest.main()
