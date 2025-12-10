from __future__ import print_function, division
from typing import Tuple, List, Union
import numpy as np

# Optional matplotlib imports
try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from skimage.color import label2rgb

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def get_seg_type(max_id: int) -> type:
    """Get optimal numpy dtype for segmentation with given max ID."""
    if max_id < 2**8:
        return np.uint8
    elif max_id < 2**16:
        return np.uint16
    elif max_id < 2**32:
        return np.uint32
    else:
        return np.uint64


def get_padsize(pad_size: Union[int, List[int]], ndim: int = 3) -> Tuple[int]:
    """Convert the padding size for 3D input volumes into numpy.pad compatible format.

    Args:
        pad_size (int, List[int]): number of values padded to the edges of each axis.
        ndim (int): the dimension of the array to be padded. Default: 3
    """
    if isinstance(pad_size, int):
        pad_size = [tuple([pad_size, pad_size]) for _ in range(ndim)]
        return tuple(pad_size)

    assert len(pad_size) in [1, ndim, 2 * ndim]
    if len(pad_size) == 1:
        pad_size = pad_size[0]
        pad_size = [tuple([pad_size, pad_size]) for _ in range(ndim)]
        return tuple(pad_size)

    if len(pad_size) == ndim:
        return tuple([tuple([x, x]) for x in pad_size])

    return tuple([tuple([pad_size[2 * i], pad_size[2 * i + 1]]) for i in range(len(pad_size) // 2)])


def array_unpad(data: np.ndarray, pad_size: Tuple[int]) -> np.ndarray:
    """Unpad a given numpy.ndarray based on the given padding size.

    Args:
        data (numpy.ndarray): the input volume to unpad.
        pad_size (tuple): number of values removed from the edges of each axis.
            Should be in the format of ((before_1, after_1), ... (before_N, after_N))
            representing the unique pad widths for each axis.
    """
    diff = data.ndim - len(pad_size)
    if diff > 0:
        extra = [(0, 0) for _ in range(diff)]
        pad_size = tuple(extra + list(pad_size))

    assert len(pad_size) == data.ndim
    index = tuple([slice(pad_size[i][0], data.shape[i] - pad_size[i][1]) for i in range(data.ndim)])
    return data[index]


def normalize_image(
    image: np.ndarray, mean: float = 0.5, std: float = 0.5, match_act: str = "none"
) -> np.ndarray:
    # the image should be float32 within [0.0, 1.0]
    if match_act == "sigmoid":
        return image
    elif match_act == "tanh":  # (0,1)->(-1,1)
        return image * 2.0 - 1.0

    assert image.dtype == np.float32
    image = (image - mean) / std
    return image


def show_image(
    image, image_type="im", num_row=1, cmap="gray", title="Test Title", interpolation=None
):
    num_imgs = image.shape[0]
    num_col = (num_imgs + num_row - 1) // num_row
    fig = plt.figure(figsize=(20.0, 3.0))
    fig.suptitle(title, fontsize=15)
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(num_row, num_col),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )
    image_list = np.split(image, num_imgs, 0)
    for ax, im in zip(grid, [np.squeeze(x) for x in image_list]):
        # Iterating over the grid returns the Axes.
        if image_type == "seg":
            im = label2rgb(im)
        ax.imshow(im, cmap=cmap, interpolation=interpolation)
        ax.axis("off")

    plt.show()
