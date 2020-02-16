
def microstructure(l=256):
    """
    Synthetic binary data: binary microstructure with blobs.

    Parameters
    ----------

    l: int, optional
        linear size of the returned image
    """
    n = 5
    x, y = np.ogrid[0:l, 0:l]
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)
    points = l * generator.rand(2, n ** 2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l / (4. * n))
    return (mask > mask.mean()).astype(np.float)

label = microstructure()
label_vol = np.stack([label for _ in range(8)], 0)
print('volume shape:', label_vol.shape)
vol_distance, vol_skeleton = skeleton_transform_volume(label_vol)
print(vol_distance.shape, vol_distance.dtype)
print(vol_skeleton.shape, vol_skeleton.dtype)
print('test done')


