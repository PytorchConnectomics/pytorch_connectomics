"""
Pure functions for connectomics-specific augmentation operations.

All functions operate on numpy arrays only (no MONAI, no config, no dict keys).
MONAI transform wrappers in transforms.py delegate to these functions.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Misalignment operations
# ---------------------------------------------------------------------------


def _move_depth_axis_to_front(img: np.ndarray, depth_axis: int) -> np.ndarray:
    """Return a view with depth axis moved to the front."""
    if depth_axis == 0:
        return img
    return np.moveaxis(img, depth_axis, 0)


def _restore_depth_axis(img: np.ndarray, depth_axis: int) -> np.ndarray:
    """Restore a depth-first array back to the original depth axis."""
    if depth_axis == 0:
        return img
    return np.moveaxis(img, 0, depth_axis)


def shift_2d(section: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2D section by (dy, dx) pixels, filling with zeros."""
    h, w = section.shape[-2:]
    out = np.zeros_like(section)
    sy = slice(max(0, -dy), min(h, h - dy))
    sx = slice(max(0, -dx), min(w, w - dx))
    ty = slice(max(0, dy), min(h, h + dy))
    tx = slice(max(0, dx), min(w, w + dx))
    out[..., ty, tx] = section[..., sy, sx]
    return out


def apply_misalignment_translation(
    img: np.ndarray,
    displacement: int,
    dy0: int,
    dx0: int,
    dy1: int,
    dx1: int,
    split_idx: int,
    mode: str,
    depth_axis: int = 0,
) -> np.ndarray:
    """Apply translation-based misalignment to a 3D numpy volume.

    Args:
        img: 3D array (D, H, W)
        displacement: max displacement in pixels (used for bounds checking)
        dy0, dx0: shift for sections before split_idx (translation mode)
        dy1, dx1: shift for section at/after split_idx
        split_idx: section index at which to split
        mode: 'slip' (single section) or 'translation' (block shift)
    """
    if img.ndim < 3 or img.shape[depth_axis] <= 2:
        return img
    if img.shape[-2] <= displacement or img.shape[-1] <= displacement:
        return img

    depth_first = _move_depth_axis_to_front(img, depth_axis).copy()
    if mode == "slip":
        depth_first[split_idx] = shift_2d(depth_first[split_idx], dy1, dx1)
    else:
        for i in range(split_idx, depth_first.shape[0]):
            depth_first[i] = shift_2d(depth_first[i], dy1, dx1)
        for i in range(0, split_idx):
            depth_first[i] = shift_2d(depth_first[i], dy0, dx0)
    return _restore_depth_axis(depth_first, depth_axis)


def apply_misalignment_rotation(
    img: np.ndarray,
    displacement: int,
    angle: float,
    split_idx: int,
    mode: str,
    depth_axis: int = 0,
) -> np.ndarray:
    """Apply rotation-based misalignment to a 3D numpy volume.

    Args:
        img: 3D array (D, H, W) — must be square in H, W
        displacement: max displacement (used to compute angle range)
        angle: rotation angle in degrees
        split_idx: section index at which to split
        mode: 'slip' (single section) or 'translation' (block rotation)
    """
    if img.ndim < 3 or img.shape[depth_axis] <= 2:
        return img

    depth_first = _move_depth_axis_to_front(img, depth_axis).copy()
    height, width = depth_first.shape[-2:]
    if height != width:
        return img

    M = cv2.getRotationMatrix2D((height / 2, height / 2), angle, 1)
    interpolation = cv2.INTER_LINEAR if depth_first.dtype == np.float32 else cv2.INTER_NEAREST

    if mode == "slip":
        depth_first[split_idx] = _warp_section(
            depth_first[split_idx], M, (height, width), interpolation
        )
    else:
        for i in range(split_idx, depth_first.shape[0]):
            depth_first[i] = _warp_section(
                depth_first[i], M, (height, width), interpolation
            )
    return _restore_depth_axis(depth_first, depth_axis)


def _warp_section(
    section: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    """Apply the same affine transform to a 2D slice or channel-first stack."""
    if section.ndim == 3:
        warped = np.empty_like(section)
        for c in range(section.shape[0]):
            warped[c] = cv2.warpAffine(
                section[c],
                matrix,
                output_shape,
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return warped
    return cv2.warpAffine(
        section,
        matrix,
        output_shape,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )


def compute_misalignment_angle_range(displacement: int, height: int) -> float:
    """Compute max rotation angle from displacement and image height."""
    x = displacement / 2.0
    y = ((height - displacement) / 2.0) * 1.42
    return math.asin(x / y) * 2.0 * 57.2958


# ---------------------------------------------------------------------------
# Missing section / parts operations
# ---------------------------------------------------------------------------


def zero_out_sections(
    img: np.ndarray, indices: np.ndarray, depth_axis: int = 0
) -> np.ndarray:
    """Zero out sections at given indices along depth_axis."""
    img = img.copy()
    if depth_axis == 0:
        img[indices, ...] = 0
    else:
        img[:, indices, ...] = 0
    return img


def fill_sections(
    img: np.ndarray,
    indices: np.ndarray,
    fill_value: float,
    depth_axis: int = 0,
) -> np.ndarray:
    """Fill entire sections with a constant value."""
    img = img.copy()
    if depth_axis == 0:
        img[indices, ...] = fill_value
    else:
        img[:, indices, ...] = fill_value
    return img


def create_missing_hole(
    img: np.ndarray,
    y_start: int,
    x_start: int,
    hole_h: int,
    hole_w: int,
    section_axis: Optional[int],
    section_idx: Optional[int],
) -> np.ndarray:
    """Create a rectangular hole (zero region) in an image/volume."""
    img = img.copy()
    index = [slice(None)] * img.ndim
    if section_axis is not None and section_idx is not None:
        index[section_axis] = section_idx
    index[-2] = slice(y_start, y_start + hole_h)
    index[-1] = slice(x_start, x_start + hole_w)
    img[tuple(index)] = 0
    return img


def fill_region(
    img: np.ndarray,
    y_start: int,
    x_start: int,
    hole_h: int,
    hole_w: int,
    section_axis: Optional[int],
    section_idx: Optional[int],
    fill_value: float,
) -> np.ndarray:
    """Fill a rectangular region with a constant value."""
    img = img.copy()
    index = [slice(None)] * img.ndim
    if section_axis is not None and section_idx is not None:
        index[section_axis] = section_idx
    index[-2] = slice(y_start, y_start + hole_h)
    index[-1] = slice(x_start, x_start + hole_w)
    img[tuple(index)] = fill_value
    return img


# ---------------------------------------------------------------------------
# Motion blur
# ---------------------------------------------------------------------------


def create_motion_blur_kernel(kernel_size: int, horizontal: bool) -> np.ndarray:
    """Create a directional motion blur kernel."""
    kernel = np.zeros((kernel_size, kernel_size))
    center = int((kernel_size - 1) / 2)
    if horizontal:
        kernel[center, :] = np.ones(kernel_size)
    else:
        kernel[:, center] = np.ones(kernel_size)
    return kernel / kernel_size


def ensure_odd_kernel_size(kernel_size: int) -> int:
    """Return a positive odd kernel size for OpenCV filters."""
    kernel_size = max(1, int(kernel_size))
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def apply_gaussian_blur(
    img: np.ndarray,
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    """Apply 2D Gaussian blur to a slice or channel-first stack."""
    kernel = (ensure_odd_kernel_size(kernel_size), ensure_odd_kernel_size(kernel_size))
    if img.ndim == 3:
        result = np.empty_like(img)
        for c in range(img.shape[0]):
            result[c] = cv2.GaussianBlur(
                img[c], kernel, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
            )
        return result
    return cv2.GaussianBlur(
        img, kernel, sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )


def blur_sections(
    img: np.ndarray,
    indices: np.ndarray,
    kernel_size: int,
    sigma: float,
    depth_axis: int = 0,
) -> np.ndarray:
    """Apply full-section Gaussian blur along the depth axis."""
    depth_first = _move_depth_axis_to_front(img, depth_axis).copy()
    for idx in indices:
        depth_first[idx] = apply_gaussian_blur(depth_first[idx], kernel_size, sigma)
    return _restore_depth_axis(depth_first, depth_axis)


def blur_region(
    img: np.ndarray,
    section_idx: int,
    y_start: int,
    x_start: int,
    hole_h: int,
    hole_w: int,
    kernel_size: int,
    sigma: float,
    depth_axis: int = 0,
) -> np.ndarray:
    """Apply Gaussian blur to a rectangular region within one section."""
    depth_first = _move_depth_axis_to_front(img, depth_axis).copy()
    blurred = apply_gaussian_blur(depth_first[section_idx], kernel_size, sigma)
    depth_first[section_idx][..., y_start:y_start + hole_h, x_start:x_start + hole_w] = (
        blurred[..., y_start:y_start + hole_h, x_start:x_start + hole_w]
    )
    return _restore_depth_axis(depth_first, depth_axis)


def apply_motion_blur(
    img: np.ndarray,
    kernel: np.ndarray,
    section_indices: np.ndarray,
) -> np.ndarray:
    """Apply motion blur kernel to specified sections of a 3D volume."""
    img = img.copy()
    for idx in section_indices:
        section = img[idx]
        if section.ndim == 3:  # [C, H, W]
            for c in range(section.shape[0]):
                section[c] = cv2.filter2D(section[c], -1, kernel)
        else:  # [H, W]
            img[idx] = cv2.filter2D(section, -1, kernel)
    return img


# ---------------------------------------------------------------------------
# Cut noise
# ---------------------------------------------------------------------------


def apply_cut_noise(
    img: np.ndarray,
    slices: List[slice],
    noise: np.ndarray,
) -> np.ndarray:
    """Add noise to a cuboid region of an image."""
    img = img.copy()
    idx = tuple(slices)
    img[idx] = np.clip(img[idx] + noise, 0, 1)
    return img


# ---------------------------------------------------------------------------
# CutBlur
# ---------------------------------------------------------------------------


def apply_cutblur(
    img: np.ndarray,
    zl: Optional[int],
    zh: Optional[int],
    yl: int,
    yh: int,
    xl: int,
    xh: int,
    down_ratio: float,
    downsample_z: bool,
) -> np.ndarray:
    """Apply CutBlur: downsample then upsample a cuboid region."""
    from scipy.ndimage import zoom

    img = img.copy()

    if img.ndim == 4:
        temp = img[:, zl:zh, yl:yh, xl:xh].copy()
        if downsample_z:
            out_shape = np.array(temp.shape) / np.array([1, down_ratio, down_ratio, down_ratio])
        else:
            out_shape = np.array(temp.shape) / np.array([1, 1, down_ratio, down_ratio])
    elif img.ndim == 3:
        temp = img[zl:zh, yl:yh, xl:xh].copy()
        if downsample_z:
            out_shape = np.array(temp.shape) / down_ratio
        else:
            out_shape = np.array(temp.shape) / np.array([1, down_ratio, down_ratio])
    else:
        temp = img[yl:yh, xl:xh].copy()
        out_shape = np.array(temp.shape) / np.array([down_ratio, down_ratio])

    out_shape = np.maximum(out_shape.astype(int), 1)

    zoom_down = [o / i for o, i in zip(out_shape, temp.shape)]
    downsampled = zoom(temp, zoom_down, order=1, mode="reflect", prefilter=True)

    zoom_up = [o / i for o, i in zip(temp.shape, downsampled.shape)]
    upsampled = zoom(downsampled, zoom_up, order=0, mode="reflect", prefilter=False)

    if img.ndim == 4:
        img[:, zl:zh, yl:yh, xl:xh] = upsampled
    elif img.ndim == 3:
        img[zl:zh, yl:yh, xl:xh] = upsampled
    else:
        img[yl:yh, xl:xh] = upsampled

    return img


# ---------------------------------------------------------------------------
# Stripe artifacts
# ---------------------------------------------------------------------------


def add_stripes_to_slice(
    slice_2d: np.ndarray,
    stripe_params: List[Tuple[float, int, float]],
    angle: float,
    mode: str,
) -> np.ndarray:
    """Add stripe artifacts at an arbitrary angle to a 2D slice.

    Args:
        slice_2d: 2D array (H, W)
        stripe_params: list of (center, thickness, intensity) tuples
        angle: stripe angle in degrees
        mode: 'add' or 'replace'
    """
    if slice_2d.ndim != 2:
        return slice_2d

    height, width = slice_2d.shape
    angle_rad = np.deg2rad(angle)

    y_coords, x_coords = np.ogrid[:height, :width]
    rotated_coords = x_coords * np.sin(angle_rad) - y_coords * np.cos(angle_rad)

    coord_min = rotated_coords.min()
    coord_max = rotated_coords.max()
    if coord_max - coord_min == 0:
        return slice_2d

    result = slice_2d.copy()
    for stripe_center, thickness, intensity in stripe_params:
        half_thickness = thickness / 2.0
        stripe_mask = np.abs(rotated_coords - stripe_center) <= half_thickness

        if mode == "add":
            result[stripe_mask] = np.clip(result[stripe_mask] + intensity, 0, 1)
        else:
            result[stripe_mask] = np.clip(intensity, 0, 1)

    return result


def apply_stripes(
    img: np.ndarray,
    stripe_params: List[Tuple[float, int, float]],
    angle: float,
    mode: str,
) -> np.ndarray:
    """Apply stripe artifacts to all slices of a volume."""
    img = img.copy()
    if img.ndim == 3:
        for z in range(img.shape[0]):
            img[z] = add_stripes_to_slice(img[z], stripe_params, angle, mode)
    elif img.ndim == 4:
        for c in range(img.shape[0]):
            for z in range(img.shape[1]):
                img[c, z] = add_stripes_to_slice(img[c, z], stripe_params, angle, mode)
    elif img.ndim == 2:
        img = add_stripes_to_slice(img, stripe_params, angle, mode)
    return img


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def smart_normalize(
    volume: np.ndarray,
    mode: str,
    divide_value: Optional[float] = None,
    clip_percentile_low: float = 0.0,
    clip_percentile_high: float = 1.0,
) -> np.ndarray:
    """Apply smart normalization with optional percentile clipping.

    Args:
        volume: numpy array to normalize
        mode: 'none', 'normal' (z-score), '0-1' (min-max), or 'divide'
        divide_value: divisor when mode='divide'
        clip_percentile_low: lower percentile for clipping (0.0 = no clip)
        clip_percentile_high: upper percentile for clipping (1.0 = no clip)
    """
    volume = volume.copy()

    if clip_percentile_low > 0.0 or clip_percentile_high < 1.0:
        low_val = np.percentile(volume, clip_percentile_low * 100)
        high_val = np.percentile(volume, clip_percentile_high * 100)
        volume = np.clip(volume, low_val, high_val)

    if mode == "none":
        pass
    elif mode == "normal":
        data_mean = volume.mean()
        data_std = volume.std()
        if data_std > 1e-8:
            volume = (volume - data_mean) / data_std
    elif mode == "0-1":
        min_val = volume.min()
        max_val = volume.max()
        if max_val > min_val:
            volume = (volume - min_val) / (max_val - min_val)
    elif mode == "divide":
        volume = volume / divide_value

    return volume
