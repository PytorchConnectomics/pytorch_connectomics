from __future__ import print_function, division
import os, sys
import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.morphology import skeletonize, dilation, erosion

def skeleton_transform(label, relabel=True):
    resolution = (1.0, 1.0)
    alpha = 1.0
    beta = 0.8

    if relabel == True: # run connected component
        label = morphology.label(label, background=0)
    
    label_id = np.unique(label)
    # print(np.unique(label_id))
    skeleton = np.zeros(label.shape, dtype=np.uint8)
    distance = np.zeros(label.shape, dtype=np.float32)
    Temp = np.zeros(label.shape, dtype=np.uint8)
    if len(label_id) == 1: # only one object within current volume
        if label_id[0] == 0:
            return distance, skeleton
        else:
            temp_id = label_id
    else:
        temp_id = label_id[1:]

    for idx in temp_id:
        temp1 = (label == idx)
        temp2 = morphology.remove_small_holes(temp1, 16, connectivity=1)

        #temp3 = erosion(temp2)
        temp3 = temp2.copy()
        Temp += temp3
        skeleton_mask = skeletonize(temp3).astype(np.uint8)
        skeleton += skeleton_mask

        skeleton_edt = ndimage.distance_transform_edt(
            1-skeleton_mask, resolution)
        dist_max = np.max(skeleton_edt*temp3)
        dist_max = np.clip(dist_max, a_min=2.0, a_max=None)
        skeleton_edt = skeleton_edt / (dist_max*alpha)
        skeleton_edt = skeleton_edt**(beta)

        reverse = 1.0-(skeleton_edt*temp3)
        distance += reverse*temp3

    # generate boundary
    distance[np.where(Temp == 0)] = -1.0

    return distance, skeleton

def skeleton_transform_volume(label):
    vol_distance = np.zeros_like(label, dtype=np.float32)
    vol_skeleton = np.zeros_like(label, dtype=np.uint8)
    for i in range(label.shape[0]):
        label_img = label[i].copy()
        distance, skeleton = skeleton_transform(label_img)
        vol_distance[i] = distance
        vol_skeleton[i] = skeleton

    return vol_distance, vol_skeleton

def test():
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

if __name__ == '__main__':
    test()