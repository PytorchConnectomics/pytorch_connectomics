# Useful utils for post-processing
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects, dilation

def binarize_and_median(pred, size=(7,7,7), thres=0.8):
    """First binarize the prediction with a given threshold, and
    then conduct median filtering to reduce noise.
    """
    pred = (pred > thres).astype(np.uint8)
    pred = ndimage.median_filter(pred, size=size)
    return pred

def remove_small_instances(segm, thres_small=128, mode='background'):
    """Remove small spurious instances. 
    """
    assert mode in ['background', 'neighbor']
    if mode == 'background':
        return remove_small_objects(segm, thres_small)

    seg_idx = np.unique(segm)[1:]
    for idx in seg_idx:
        temp = (segm == idx).astype(np.uint8)
        if temp.sum() < thres_small:
            temp_dilated = dilation(temp, np.ones((1,3,3)))
            diff = temp_dilated - temp
            diff_mask = segm.copy()
            diff_mask[np.where(diff==0)]=0
            touch_idx, counts = np.unique(diff_mask, return_counts=True)

            if len(touch_idx) > 1 and touch_idx[0] == 0:
                touch_idx = touch_idx[1:]
                counts = counts[1:]

            segm[np.where(segm==idx)] = touch_idx[np.argmax(counts)]

    return segm
