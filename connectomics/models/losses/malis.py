"""PyTorch wrapper for the vendored MALIS affinity loss."""

from __future__ import annotations

import importlib
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

_malis_lib: Any = None
_MALIS_IMPORT_ERROR: BaseException | None = None

try:
    _malis_lib = importlib.import_module("malis")
except Exception as _malis_import_exc:
    _malis_lib = None
    _MALIS_IMPORT_ERROR = _malis_import_exc


class MalisLoss(nn.Module):
    """Constrained MALIS loss for 3D affinity tensors.

    v0 supports PyTC's canonical 3D affinity layout ``[B, C, Z, Y, X]`` only.
    2D tensors are rejected explicitly because the vendored MALIS helpers operate
    on 3D affinity graphs by default. See ``lib/malis/INVESTIGATION.md`` for
    GPU MALIS candidates and algorithm-level speedup follow-ups.

    Performance knobs (see ``docs/source/notes/malis.rst``):

    - ``malis_crop_size`` — random sub-volume crop on each forward call.
      ``64`` on a ``128^3`` patch gives ~4.6x measured step speedup vs
      the full-volume baseline (slurm 2505814 vs 2487040).
    - ``label_transform.emit_gt_seg: true`` (YAML, paired with this
      loss) — passes the eroded GT segmentation in via ``gt_seg=...``,
      skipping the per-step ``connected_components_affgraph`` call and
      preserving global instance IDs when ``malis_crop_size`` is set.
    """

    def __init__(
        self,
        nhood: Sequence[Sequence[int]] | np.ndarray | torch.Tensor | None = None,
        *,
        reduction: str = "sum",
        sigmoid: bool = True,
        malis_crop_size: int | Sequence[int] | None = None,
        malis_num_workers: int | None = None,
    ):
        """Initialize the MALIS affinity loss.

        Args:
            nhood: 3D affinity neighborhood with shape ``(n_edge, 3)``.
            reduction: ``"sum"``, ``"mean"``, or ``"none"``.
            sigmoid: Apply sigmoid to ``pred`` before computing the loss.
            malis_crop_size: Optional random shared spatial crop. ``None``
                preserves current full-volume behavior. An int ``K`` becomes
                ``(K, K, K)``; any length-3 integer sequence becomes
                ``(Kz, Ky, Kx)``.
            malis_num_workers: Optional number of worker threads for the
                per-sample/per-pass MALIS weight calls. ``None`` or ``1`` keeps
                the serial default path unchanged.
        """
        super().__init__()
        if _malis_lib is None:
            raise ImportError(
                "MalisLoss requires the optional 'malis' package/extension. "
                "Build or install the vendored library under lib/malis before "
                "constructing this loss."
            ) from _MALIS_IMPORT_ERROR
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction for MalisLoss: {reduction!r}")

        self.reduction = reduction
        self.sigmoid = sigmoid
        self.nhood = self._coerce_nhood(nhood)
        self.malis_crop_size = self._coerce_crop_size(malis_crop_size)
        self.malis_num_workers = self._coerce_num_workers(malis_num_workers)
        self._edge_list_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        gt_seg: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Compute MALIS-weighted squared affinity error.

        The default ``reduction="sum"`` matches the vendored TensorFlow MALIS
        implementation, which returns ``reduce_sum(weights * squared_error)``.
        ``reduction="mean"`` and ``"none"`` follow standard PyTorch reduction
        semantics over that weighted edge-loss tensor.

        Args:
            pred: Affinity logits by default, or affinities if ``sigmoid=False``.
            target: Ground-truth affinities with shape ``[B, C, Z, Y, X]``.
            mask: Optional binary edge mask for known ground-truth affinities.
                Masked-out edges are excluded from MALIS pass constraints and
                zeroed before per-pass normalization, but the mask does not
                change GT connected-component reconstruction.
            gt_seg: Optional ground-truth segmentation with shape ``[B, Z, Y, X]``
                or ``[B, 1, Z, Y, X]``. When supplied, MALIS uses these instance
                labels directly instead of reconstructing components from
                ``target`` affinities.
        """
        self._validate_inputs(pred, target)

        pred_aff = torch.sigmoid(pred) if self.sigmoid else pred
        target_aff = target.to(device=pred.device, dtype=pred_aff.dtype)
        mask_aff = None if mask is None else self._prepare_mask(mask, pred_aff)
        gt_seg_tensor = self._prepare_gt_seg(gt_seg, pred_aff)
        pred_aff, target_aff, mask_aff, gt_seg_tensor = self._apply_crop_if_configured(
            pred_aff,
            target_aff,
            mask_aff,
            gt_seg_tensor,
        )
        weight_kwargs = {}
        if gt_seg_tensor is not None:
            weight_kwargs["gt_seg"] = gt_seg_tensor.detach()
        weights = self._compute_malis_weights(
            pred_aff.detach(),
            target_aff.detach(),
            None if mask_aff is None else mask_aff.detach(),
            **weight_kwargs,
        )

        edge_loss = (pred_aff - target_aff) ** 2
        weighted_loss = weights * edge_loss

        if self.reduction == "none":
            return weighted_loss
        if self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss.mean()

    def _coerce_nhood(
        self,
        nhood: Sequence[Sequence[int]] | np.ndarray | torch.Tensor | None,
    ) -> np.ndarray:
        if nhood is None:
            arr = _malis_lib.mknhood3d(1)
        elif isinstance(nhood, torch.Tensor):
            arr = nhood.detach().cpu().numpy()
        else:
            arr = np.asarray(nhood)

        arr = np.ascontiguousarray(arr, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                "MalisLoss v0 expects a 3D neighborhood with shape (n_edge, 3); "
                f"got {tuple(arr.shape)}."
            )
        return arr

    def _coerce_crop_size(
        self,
        value: int | Sequence[int] | None,
    ) -> tuple[int, int, int] | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(
                "malis_crop_size must be a positive int or length-3 sequence; " f"got {value!r}."
            )
        if isinstance(value, (int, np.integer)):
            value_int = int(value)
            if value_int <= 0:
                raise ValueError(f"malis_crop_size scalar must be > 0; got {value!r}.")
            return (value_int, value_int, value_int)
        if isinstance(value, str):
            raise ValueError(
                "malis_crop_size must be int or length-3 int sequence, not str; " f"got {value!r}."
            )

        try:
            seq = list(value)
        except TypeError as e:
            raise ValueError(
                "malis_crop_size must be int or length-3 int sequence; " f"got {value!r}."
            ) from e

        if len(seq) != 3:
            raise ValueError(
                "malis_crop_size sequence must have length 3; "
                f"got len={len(seq)} value={value!r}."
            )

        out: list[int] = []
        for dim in seq:
            if isinstance(dim, bool):
                raise ValueError("malis_crop_size element must be a positive int; " f"got {dim!r}.")
            if not isinstance(dim, (int, np.integer)):
                raise ValueError(
                    "malis_crop_size element must be int; "
                    f"got {dim!r} (type {type(dim).__name__})."
                )
            dim_int = int(dim)
            if dim_int <= 0:
                raise ValueError(f"malis_crop_size element must be > 0; got {dim!r}.")
            out.append(dim_int)

        return (out[0], out[1], out[2])

    def _coerce_num_workers(self, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"malis_num_workers must be an int >= 1 or None; got {value!r}.")
        if not isinstance(value, (int, np.integer)):
            raise ValueError(
                "malis_num_workers must be an int >= 1 or None; "
                f"got {value!r} (type {type(value).__name__})."
            )

        value_int = int(value)
        if value_int < 1:
            raise ValueError(f"malis_num_workers must be >= 1; got {value!r}.")
        if value_int == 1:
            return None
        return value_int

    def _validate_inputs(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(
                "MalisLoss expects pred and target to have identical shapes; "
                f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}."
            )
        if pred.ndim == 4:
            raise NotImplementedError(
                "MalisLoss v0 supports 3D affinity tensors [B, C, Z, Y, X] only; "
                "2D support requires a dedicated mknhood2d/nodelist_like path."
            )
        if pred.ndim != 5:
            raise ValueError(
                "MalisLoss expects 3D affinity tensors with shape [B, C, Z, Y, X]; "
                f"got {tuple(pred.shape)}."
            )
        if pred.shape[1] != self.nhood.shape[0]:
            raise ValueError(
                "MalisLoss channel count must match nhood edges; "
                f"got C={pred.shape[1]}, nhood_edges={self.nhood.shape[0]}."
            )

    def _prepare_mask(self, mask: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        mask_tensor = mask.to(device=pred.device, dtype=pred.dtype)
        if mask_tensor.ndim == pred.ndim - 1 and mask_tensor.shape[0] == pred.shape[0]:
            mask_tensor = mask_tensor.unsqueeze(1)
        try:
            return torch.broadcast_to(mask_tensor, pred.shape)
        except RuntimeError as e:
            raise ValueError(
                "MalisLoss mask shape is not broadcastable to pred shape: "
                f"mask={tuple(mask.shape)}, pred={tuple(pred.shape)}."
            ) from e

    def _prepare_gt_seg(
        self,
        gt_seg: torch.Tensor | np.ndarray | None,
        pred: torch.Tensor,
    ) -> torch.Tensor | None:
        if gt_seg is None:
            return None

        gt_seg_tensor = torch.as_tensor(gt_seg, device=pred.device).detach()
        if gt_seg_tensor.ndim == pred.ndim and gt_seg_tensor.shape[1] == 1:
            gt_seg_tensor = gt_seg_tensor.squeeze(1)
        elif gt_seg_tensor.ndim == pred.ndim - 2 and pred.shape[0] == 1:
            gt_seg_tensor = gt_seg_tensor.unsqueeze(0)

        expected_shape = (pred.shape[0],) + tuple(pred.shape[-3:])
        if tuple(gt_seg_tensor.shape) != expected_shape:
            raise ValueError(
                "MalisLoss gt_seg must have shape [B, Z, Y, X] or [B, 1, Z, Y, X] "
                f"matching pred spatial dims; got gt_seg={tuple(gt_seg_tensor.shape)}, "
                f"expected={expected_shape}."
            )
        return gt_seg_tensor.contiguous()

    def _apply_crop_if_configured(
        self,
        pred_aff: torch.Tensor,
        target_aff: torch.Tensor,
        mask_aff: torch.Tensor | None,
        gt_seg: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Apply the configured random sub-volume crop, if any.

        Offset sampling stays on CPU. The returned tensors are contiguous copies
        of the narrowed views; this is required because the downstream CPU MALIS
        path eventually calls ``.cpu().numpy()``. Gradient still flows through
        the contiguous copies back to the original ``pred``. For B=2, C=3,
        crop=64, and fp16, pred + target + mask copies are about 9 MiB before
        overhead.

        Returns ``(pred_cropped, target_cropped, mask_cropped, gt_seg_cropped)``.
        If no crop is configured the inputs are returned unchanged.
        """
        if self.malis_crop_size is None:
            return pred_aff, target_aff, mask_aff, gt_seg

        k_z, k_y, k_x = self.malis_crop_size
        z_dim, y_dim, x_dim = pred_aff.shape[-3:]
        if k_z > z_dim or k_y > y_dim or k_x > x_dim:
            raise ValueError(
                "malis_crop_size exceeds spatial dims: "
                f"crop={self.malis_crop_size}, input_zyx=({z_dim},{y_dim},{x_dim})."
            )

        z0 = int(torch.randint(0, z_dim - k_z + 1, (1,)).item())
        y0 = int(torch.randint(0, y_dim - k_y + 1, (1,)).item())
        x0 = int(torch.randint(0, x_dim - k_x + 1, (1,)).item())

        pred_c = pred_aff.narrow(-3, z0, k_z).narrow(-2, y0, k_y).narrow(-1, x0, k_x).contiguous()
        target_c = (
            target_aff.narrow(-3, z0, k_z).narrow(-2, y0, k_y).narrow(-1, x0, k_x).contiguous()
        )
        mask_c = (
            None
            if mask_aff is None
            else mask_aff.narrow(-3, z0, k_z).narrow(-2, y0, k_y).narrow(-1, x0, k_x).contiguous()
        )
        gt_seg_c = (
            None
            if gt_seg is None
            else gt_seg.narrow(-3, z0, k_z).narrow(-2, y0, k_y).narrow(-1, x0, k_x).contiguous()
        )
        return pred_c, target_c, mask_c, gt_seg_c

    def _compute_malis_weights(
        self,
        pred_aff: torch.Tensor,
        target_aff: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        gt_seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_np = pred_aff.to(dtype=torch.float32).cpu().numpy()
        target_np = target_aff.to(dtype=torch.float32).cpu().numpy()
        mask_np = None if mask is None else mask.to(dtype=torch.float32).cpu().numpy()
        gt_seg_np = None if gt_seg is None else gt_seg.cpu().numpy()
        weights = np.empty_like(pred_np, dtype=np.float32)

        if self.malis_num_workers is None:
            for batch_idx in range(pred_np.shape[0]):
                gt_affs = np.ascontiguousarray(target_np[batch_idx] > 0.5, dtype=np.int32)
                pred_sample = np.ascontiguousarray(pred_np[batch_idx], dtype=np.float32)
                mask_sample = None
                if mask_np is not None:
                    mask_sample = np.ascontiguousarray(mask_np[batch_idx] == 1, dtype=bool)
                if gt_seg_np is None:
                    gt_seg_sample, _ = _malis_lib.connected_components_affgraph(gt_affs, self.nhood)
                else:
                    gt_seg_sample = np.ascontiguousarray(gt_seg_np[batch_idx], dtype=np.uint64)
                weights[batch_idx] = self._compute_sample_weights(
                    pred_sample,
                    gt_affs,
                    gt_seg_sample,
                    mask_sample,
                )

            return torch.from_numpy(weights).to(device=pred_aff.device, dtype=pred_aff.dtype)

        from concurrent.futures import ThreadPoolExecutor

        samples = [
            self._prepare_sample(pred_np, target_np, mask_np, gt_seg_np, batch_idx)
            for batch_idx in range(pred_np.shape[0])
        ]
        tasks = [(batch_idx, pos) for batch_idx in range(len(samples)) for pos in (0, 1)]

        def _run(task: tuple[int, int]) -> tuple[int, np.ndarray]:
            batch_idx, pos = task
            return batch_idx, self._malis_pass(*samples[batch_idx], pos=pos)

        with ThreadPoolExecutor(max_workers=self.malis_num_workers) as executor:
            results = list(executor.map(_run, tasks))

        acc: dict[int, np.ndarray] = {}
        for batch_idx, sample_weights in results:
            acc[batch_idx] = (
                sample_weights if batch_idx not in acc else acc[batch_idx] + sample_weights
            )
        for batch_idx in range(len(samples)):
            weights[batch_idx] = acc[batch_idx]

        return torch.from_numpy(weights).to(device=pred_aff.device, dtype=pred_aff.dtype)

    def _prepare_sample(
        self,
        pred_np: np.ndarray,
        target_np: np.ndarray,
        mask_np: np.ndarray | None,
        gt_seg_np: np.ndarray | None,
        batch_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        gt_affs = np.ascontiguousarray(target_np[batch_idx] > 0.5, dtype=np.int32)
        pred_sample = np.ascontiguousarray(pred_np[batch_idx], dtype=np.float32)
        mask_sample = None
        if mask_np is not None:
            mask_sample = np.ascontiguousarray(mask_np[batch_idx] == 1, dtype=bool)
        if gt_seg_np is None:
            gt_seg_sample, _ = _malis_lib.connected_components_affgraph(gt_affs, self.nhood)
        else:
            gt_seg_sample = np.ascontiguousarray(gt_seg_np[batch_idx], dtype=np.uint64)
        return pred_sample, gt_affs, gt_seg_sample, mask_sample

    def _compute_sample_weights(
        self,
        pred_affs: np.ndarray,
        gt_affs: np.ndarray,
        gt_seg: np.ndarray,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        weights_neg = self._malis_pass(pred_affs, gt_affs, gt_seg, mask, pos=0)
        weights_pos = self._malis_pass(pred_affs, gt_affs, gt_seg, mask, pos=1)
        return weights_neg + weights_pos

    def _malis_pass(
        self,
        pred_affs: np.ndarray,
        gt_affs: np.ndarray,
        gt_seg: np.ndarray,
        mask: np.ndarray | None,
        *,
        pos: int,
    ) -> np.ndarray:
        pass_affs = np.array(pred_affs, copy=True)
        class_edges = gt_affs == (1 - pos)
        constraint_edges = class_edges if mask is None else class_edges & mask
        pass_affs[constraint_edges] = 1 - pos

        node1, node2 = self._edge_list_for_shape(gt_seg.shape)
        weights = _malis_lib.malis_loss_weights(
            np.ascontiguousarray(gt_seg, dtype=np.uint64).ravel(),
            node1.ravel(),
            node2.ravel(),
            np.ascontiguousarray(pass_affs, dtype=np.float32).ravel(),
            pos,
        )
        weights = weights.reshape((-1,) + tuple(gt_seg.shape)).astype(np.float32, copy=False)
        weights[class_edges] = 0.0
        if mask is not None:
            weights[~mask] = 0.0

        num_pairs = weights.sum(dtype=np.float64)
        if num_pairs > 0:
            weights = weights / np.float32(num_pairs)
        return weights

    def _edge_list_for_shape(self, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        cached = self._edge_list_cache.get(shape)
        if cached is None:
            node1, node2 = _malis_lib.nodelist_like(shape, self.nhood)
            cached = (
                np.ascontiguousarray(node1, dtype=np.uint64),
                np.ascontiguousarray(node2, dtype=np.uint64),
            )
            self._edge_list_cache[shape] = cached
        return cached
