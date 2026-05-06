from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .system import SystemConfig


@dataclass
class InferenceDataConfig:
    """Inference data override configuration.

    Optional dataset/data loader overrides for inference jobs.
    """

    input_path: Optional[str] = None
    output_path: Optional[str] = None
    resolution: Optional[List[float]] = None


@dataclass
class SlidingWindowConfig:
    """Sliding window inference configuration."""

    enabled: bool = False
    window_size: Optional[List[int]] = None  # None = fallback to data.dataloader.patch_size
    sw_batch_size: Optional[int] = None  # If None, falls back to data.dataloader.batch_size
    overlap: float = 0.5
    blending: str = "gaussian"  # "constant" or "gaussian"
    sigma_scale: float = 0.125
    padding_mode: str = "reflect"
    cval: float = 0.0
    keep_input_on_cpu: bool = False  # Move full volume to CPU between sliding-window batches
    lazy_load: bool = False  # Stream ROIs from disk instead of materializing the full volume
    distributed_sharding: bool = False  # Split lazy sliding windows across DDP ranks
    distributed_reduce_chunk_mb: int = 128  # Chunk size for rank-0 accumulator reductions
    sw_device: Optional[str] = None
    output_device: Optional[str] = None
    # BANIS-style boundary handling (opt-in). When either is set, a custom
    # inferer replaces MONAI's SlidingWindowInferer.
    # snap_to_edge: place last window at image_size-roi_size so every window
    # fits inside the volume; no whole-volume padding.
    # target_context: per-axis voxels added on each side of the window before
    # forward; central roi_size of the prediction is accumulated. Per-window
    # context overhang is padded via padding_mode. Requires the model to
    # preserve input spatial shape.
    snap_to_edge: bool = False
    target_context: List[int] = field(default_factory=list)
    # Per-axis voxel count to zero out at each border of the per-window
    # importance map after Gaussian/constant construction. Discards window
    # border predictions from the weighted average. Honored by the lazy
    # sliding-window path; ignored by the eager MONAI path.
    border_mask: List[int] = field(default_factory=list)
    # In-memory dtype for the value accumulator that holds weighted prediction
    # sums during lazy sliding-window inference. Defaults to float32. Set to
    # "float16" or "bfloat16" to halve accumulator RAM at some precision cost.
    # Weight accumulator (the divider) stays float32 regardless.
    accumulator_dtype: Optional[str] = None


@dataclass
class InferenceExecutionConfig:
    """Inference execution mode independent of output-array/storage details."""

    strategy: str = "whole_volume"  # "whole_volume" or "chunked"
    do_eval: bool = True


@dataclass
class ChunkStitchingConfig:
    """Boundary stitching configuration for chunked decoded outputs."""

    method: str = "affinity_cc_boundary_union"
    threshold: Optional[float] = None
    edge_offset: Optional[int] = None
    min_contact: int = 1


@dataclass
class ChunkingConfig:
    """Chunked inference for volumes too large to materialize at once.

    Chunking is an inference execution strategy: data sections still describe
    whole volumes, while this section controls how each volume is partitioned
    and written/decoded.
    """

    enabled: bool = False
    output_mode: str = "decoded"  # "decoded" or "raw_prediction"
    chunk_size: Optional[List[int]] = None  # ZYX after test-time val_transpose.
    halo: List[int] = field(default_factory=lambda: [0, 0, 0])
    axes: str = "all"  # "all" or "z"; "z" keeps full YX in each chunk.
    temp_dir: str = ""
    save_intermediate: bool = False
    stitching: ChunkStitchingConfig = field(default_factory=ChunkStitchingConfig)


@dataclass
class OutputArrayPartitionConfig:
    """Partitioning of the final prediction array for chunked inference."""

    enabled: bool = False
    output_mode: str = "decoded"  # "decoded" or "raw_prediction"
    size: Optional[List[int]] = None  # ZYX after test-time val_transpose.
    halo: List[int] = field(default_factory=lambda: [0, 0, 0])
    axes: str = "all"  # "all" or "z"; "z" keeps full YX in each chunk.
    temp_dir: str = ""
    save_intermediate: bool = False
    stitching: ChunkStitchingConfig = field(default_factory=ChunkStitchingConfig)


@dataclass
class OutputArrayStoreConfig:
    """Physical storage for prediction-array artifacts."""

    enabled: bool = False
    backend: str = "h5"  # "h5", "zarr", or "tensorstore" when implemented by runtime.
    output_path: Optional[str] = None
    cache_suffix: str = "_x1_prediction.h5"
    save_all_heads: bool = False
    dtype: Optional[str] = None
    compression: Optional[str] = "gzip"
    chunks: Optional[List[int]] = None  # CZYX storage chunks when backend supports it.
    write_mode: str = "single_writer"  # "single_writer" or "distributed_regions".


@dataclass
class OutputArrayConfig:
    """Final prediction-array shape, dtype, partitioning, and storage."""

    # Optional channel selector applied after channel activations and before
    # decoding/saving. Accepts None, int, slice string, or explicit index list.
    select_channel: Optional[Any] = None
    # Crop context padding before decoding/saving predictions:
    # [D,H,W]/[H,W] symmetric or [Db,Da,Hb,Ha,Wb,Wa] asymmetric.
    crop_pad: Optional[List[int]] = None
    # In-memory dtype for lazy sliding-window value accumulators.
    accumulator_dtype: Optional[str] = None
    partition: OutputArrayPartitionConfig = field(default_factory=OutputArrayPartitionConfig)
    store: OutputArrayStoreConfig = field(default_factory=OutputArrayStoreConfig)


@dataclass
class TestTimeAugmentationConfig:
    """Test-time augmentation (TTA) configuration."""

    enabled: bool = False
    distributed_sharding: bool = True  # Split TTA variants across DDP ranks for one-volume tests
    distributed_reduce_chunk_mb: int = 128  # Chunk size for rank reduction of large volumes
    # Optional channel-wise activation overrides applied before aggregation.
    # Each entry is a mapping:
    #   {channels: "start:end" | int | [int, ...], activation: "sigmoid"|"softmax"|"tanh"|"none"}
    # Channel selectors follow Python/NumPy indexing rules:
    #   -1 means the last channel, ":" means all channels, ":-1" excludes the last.
    channel_activations: List[Dict[str, Any]] = field(default_factory=list)
    flip_axes: Any = "all"  # "all" | "none" | list[int] (0-indexed spatial: 0=z, 1=y, 2=x)
    flip_combinations: Optional[List[List[int]]] = None  # explicit list of axis subsets
    rotation90_axes: Any = None  # "all" | None | [[int, int], ...] spatial plane pairs
    rotate90_k: Optional[List[int]] = None  # subset of quarter-turns, defaults to [0,1,2,3]
    patch_first_local: bool = True  # slide once, apply local TTA inside each ROI batch
    apply_mask: bool = True
    transforms: Optional[List[Dict[str, Any]]] = None  # advanced explicit transforms

    # Aggregation — single mode or per-channel list.
    # Single string: "mean", "min", "max" applied to all channels.
    # Per-channel list: [["0:3", "min"], ["3:", "mean"]] where each entry is
    # [channel_selector, mode].  Channel selectors use the same syntax as
    # loss pred_slice / target_slice (e.g. "0:3", "-1:", "3").
    ensemble_mode: Any = "mean"
    # CUDA cache cleanup cadence during TTA loop (0 disables cleanup).
    empty_cache_interval: int = 4


@dataclass
class SavePredictionConfig:
    """Prediction saving configuration."""

    enabled: bool = False
    output_formats: List[str] = field(default_factory=lambda: ["h5"])  # Any of: h5, tiff, png
    output_path: Optional[str] = None
    cache_suffix: str = "_x1_prediction.h5"
    save_all_heads: bool = False

    # Optional file/cache dtype conversion. This does not affect decoding or
    # evaluation. Use inference.prediction_transform for semantic value changes.
    storage_dtype: Optional[str] = None

    # File writing behavior
    compression: Optional[str] = "gzip"


@dataclass
class PredictionTransformConfig:
    """Semantic prediction transforms applied before decoding/evaluation.

    Unlike ``save_prediction``, these transforms affect the in-memory prediction
    array used by downstream decoding even when intermediate predictions are not
    saved. Keep disabled unless decoder thresholds are configured for the
    transformed value range.
    """

    enabled: bool = False
    # -1 keeps native prediction values. >0 scales prediction values.
    intensity_scale: float = -1.0
    # Optional in-memory dtype conversion. Prefer float16/float32 here; integer
    # dtypes change threshold semantics and are mainly for save_prediction.
    intensity_dtype: Optional[str] = None


@dataclass
class InferenceMemoryCleanupConfig:
    """Memory cleanup policy for large-volume inference."""

    enabled: bool = True
    gc_collect: bool = True
    empty_cuda_cache: bool = True
    # Opt-in only: safe for one-volume test jobs, but it prevents additional
    # forward passes in the same test epoch unless Lightning moves the module
    # back to the accelerator.
    release_model_after_inference: bool = False


@dataclass
class InferenceConfig:
    """Inference configuration.

    Shared inference settings for sliding window and test-time augmentation.
    Data paths are resolved in merged runtime `cfg.data`.

    Key Features:
    - Sliding window inference for large volumes
    - Test-time augmentation (TTA) support
    - Saving intermediate predictions
    - Optional postprocessing for prediction outputs
    - Runtime resource overrides for inference

    Note: stage-specific overrides are merged before runtime; consumers should read `cfg.inference`.
    """

    # Named output head selection for multi-head models. When unset, falls back
    # to model.primary_head or the sole configured head.
    head: Optional[str] = None
    # Optional channel selector applied after channel activations and before
    # decoding/saving. Accepts None, int, slice string, or explicit index list.
    select_channel: Optional[Any] = None
    strategy: str = "whole_volume"  # "whole_volume" or "chunked"
    # Crop context padding before decoding/saving predictions:
    # [D,H,W]/[H,W] symmetric or [Db,Da,Hb,Ha,Wb,Wa] asymmetric.
    crop_pad: Optional[List[int]] = None
    execution: InferenceExecutionConfig = field(default_factory=InferenceExecutionConfig)
    window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    output_array: OutputArrayConfig = field(default_factory=OutputArrayConfig)
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    test_time_augmentation: TestTimeAugmentationConfig = field(
        default_factory=TestTimeAugmentationConfig
    )
    # Optional explicit intermediate prediction file (.h5). If set in test
    # mode, pipeline loads this file directly and proceeds to top-level decoding.
    tta_result_path: str = ""
    prediction_transform: PredictionTransformConfig = field(
        default_factory=PredictionTransformConfig
    )
    save_prediction: SavePredictionConfig = field(default_factory=SavePredictionConfig)
    memory_cleanup: InferenceMemoryCleanupConfig = field(
        default_factory=InferenceMemoryCleanupConfig
    )
    # If True, switch model to eval() during test/predict. Set False to keep
    # train() mode (useful for BatchNorm recalibration or MC-Dropout workflows).
    do_eval: bool = True

    # Inference-specific runtime overrides (applied in test/tune modes)
    # `system` overrides selected keys on top-level cfg.system during stage resolution.
    system: SystemConfig = field(default_factory=lambda: SystemConfig(num_gpus=-1, num_workers=-1))


def _copy_attrs(src: object, dst: object, names: list[str]) -> None:
    for name in names:
        if hasattr(src, name) and hasattr(dst, name):
            setattr(dst, name, getattr(src, name))


def sync_inference_runtime_aliases(cfg: object) -> None:
    """Materialize canonical inference sections into runtime-owned objects.

    Runtime code still consumes the narrow owner objects (`sliding_window`,
    `chunking`, and `save_prediction`). Canonical YAML should configure
    `execution`, `window`, and `output_array`; this sync keeps one effective
    runtime representation after config load and stage resolution.
    """
    inference = getattr(cfg, "inference", None)
    if inference is None:
        return

    execution = getattr(inference, "execution", None)
    if execution is not None and execution != InferenceExecutionConfig():
        strategy = getattr(execution, "strategy", None)
        if strategy is not None:
            inference.strategy = strategy
        inference.do_eval = bool(getattr(execution, "do_eval", getattr(inference, "do_eval", True)))

    window = getattr(inference, "window", None)
    sliding_window = getattr(inference, "sliding_window", None)
    if window is not None and sliding_window is not None:
        if window != SlidingWindowConfig():
            _copy_attrs(
                window,
                sliding_window,
                [
                    "enabled",
                    "window_size",
                    "sw_batch_size",
                    "overlap",
                    "blending",
                    "sigma_scale",
                    "padding_mode",
                    "cval",
                    "keep_input_on_cpu",
                    "lazy_load",
                    "distributed_sharding",
                    "distributed_reduce_chunk_mb",
                    "sw_device",
                    "output_device",
                    "snap_to_edge",
                    "target_context",
                    "border_mask",
                    "accumulator_dtype",
                ],
            )

    output_array = getattr(inference, "output_array", None)
    if output_array is None:
        return

    if getattr(output_array, "select_channel", None) is not None:
        inference.select_channel = output_array.select_channel
    if getattr(output_array, "crop_pad", None) is not None:
        inference.crop_pad = output_array.crop_pad

    accumulator_dtype = getattr(output_array, "accumulator_dtype", None)
    if accumulator_dtype is not None and sliding_window is not None:
        sliding_window.accumulator_dtype = accumulator_dtype

    partition = getattr(output_array, "partition", None)
    chunking = getattr(inference, "chunking", None)
    if partition is not None and chunking is not None:
        if partition != OutputArrayPartitionConfig():
            chunking.enabled = bool(getattr(partition, "enabled", False))
            chunking.output_mode = getattr(partition, "output_mode", chunking.output_mode)
            chunking.chunk_size = getattr(partition, "size", chunking.chunk_size)
            chunking.halo = getattr(partition, "halo", chunking.halo)
            chunking.axes = getattr(partition, "axes", chunking.axes)
            chunking.temp_dir = getattr(partition, "temp_dir", chunking.temp_dir)
            chunking.save_intermediate = bool(
                getattr(partition, "save_intermediate", chunking.save_intermediate)
            )
            if getattr(partition, "stitching", None) is not None:
                chunking.stitching = partition.stitching

    store = getattr(output_array, "store", None)
    save_prediction = getattr(inference, "save_prediction", None)
    if store is not None and save_prediction is not None:
        if store != OutputArrayStoreConfig():
            save_prediction.enabled = bool(getattr(store, "enabled", False))
            backend = str(getattr(store, "backend", "h5")).lower()
            save_prediction.output_formats = [backend]
            save_prediction.output_path = getattr(store, "output_path", None)
            save_prediction.cache_suffix = getattr(store, "cache_suffix", "_x1_prediction.h5")
            save_prediction.save_all_heads = bool(getattr(store, "save_all_heads", False))
            save_prediction.storage_dtype = getattr(store, "dtype", None)
            save_prediction.compression = getattr(store, "compression", None)
