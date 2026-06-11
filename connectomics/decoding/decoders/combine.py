"""Multi-input decoder-graph combine operations."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _validate_label_input(arr: np.ndarray, *, name: str) -> None:
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"combine_split {name} must be an integer label array.")
    if np.issubdtype(arr.dtype, np.signedinteger) and arr.size and int(arr.min()) < 0:
        raise ValueError(f"combine_split {name} must not contain negative labels.")


def _check_key_space(max_a: int, max_b: int) -> int:
    max_uint64 = int(np.iinfo(np.uint64).max)
    if max_b >= max_uint64:
        raise OverflowError("combine_split pair-key base exceeds uint64 range.")
    base = max_b + 1
    if max_a > (max_uint64 - max_b) // base:
        raise OverflowError("combine_split pair keys would overflow uint64.")
    return base


def combine_split(
    inputs: Sequence[np.ndarray],
    *,
    output_dtype: str | np.dtype = "uint32",
) -> np.ndarray:
    """Return the background-preserving coarsest common refinement of two labels."""
    if len(inputs) != 2:
        raise ValueError(f"combine_split expects exactly two inputs, got {len(inputs)}.")

    a = np.asarray(inputs[0])
    b = np.asarray(inputs[1])
    if a.shape != b.shape:
        raise ValueError(
            f"combine_split inputs must have matching shapes, got {a.shape} and {b.shape}."
        )
    _validate_label_input(a, name="input 0")
    _validate_label_input(b, name="input 1")

    dtype = np.dtype(output_dtype)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"combine_split output_dtype must be an integer dtype, got {dtype}.")

    out = np.zeros(a.shape, dtype=dtype)
    fg = (a != 0) & (b != 0)
    if not bool(fg.any()):
        return out

    a_fg = a[fg]
    b_fg = b[fg]
    max_a = int(a_fg.max())
    max_b = int(b_fg.max())
    base = _check_key_space(max_a, max_b)

    key = a_fg.astype(np.uint64, copy=False)
    if key is a_fg:
        key = key.copy()
    key *= np.uint64(base)
    np.add(key, b_fg.astype(np.uint64, copy=False), out=key)

    _, inv = np.unique(key, return_inverse=True)
    n_labels = int(inv.max()) + 1 if inv.size else 0
    if n_labels >= 2**32:
        raise OverflowError("combine_split produced too many labels for uint32 output.")
    dtype_info = np.iinfo(dtype)
    if n_labels > int(dtype_info.max):
        raise OverflowError(
            f"combine_split produced {n_labels} labels, exceeding output dtype {dtype}."
        )

    labels = np.arange(1, n_labels + 1, dtype=dtype)
    out[fg] = labels[inv]
    return out


__all__ = ["combine_split"]
