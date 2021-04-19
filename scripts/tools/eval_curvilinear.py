# Evaluation scripts for curvilinear structures based on
# correctness, completness, quality (Mosinska et al. https://arxiv.org/abs/1712.02190)
# and foreground IoU. Support multi-cpu paralellism with Python multiprocessing.

import os
import imageio
import argparse
import numpy as np
import multiprocessing

from skimage.morphology import skeletonize, dilation, square

def get_args():
    parser = argparse.ArgumentParser(description='Curvilinear structure evaluation.')
    parser.add_argument('--gt-path',  type=str, help='path to groundtruth mask')
    parser.add_argument('--pd-path',  type=str, help='path to predicted structures')
    parser.add_argument('--thres', type=int, default=128, help='threshold for prediction [0, 255]')
    parser.add_argument('--max-index', type=int, default=200, help='maximum image index')
    args = parser.parse_args()
    return args

args = get_args()
print(args)
# ---------------------------------------------------------------------------------------

def compute_metrics(skeleton_output, skeleton_gt, skeleton_output_dil, skeleton_gt_dil):

    """
    inputs:
    skeleton_output - list containing skeletonized network probability maps after binarization at 0.5
    skeleton_gt - list containing skeletonized groun-truth images
    skeleton_output_dil - list containing skeletonized outputs dilated by the factor N
    skeleton_gt_dil - list containing skeletonized ground truth images dilated by the factor N
    """
    
    tpcor = 0
    tpcom = 0
    fn = 0
    fp = 0

    for i in range(0, len(skeleton_output)):
        tpcor += ((skeleton_output[i] == skeleton_gt_dil[i]) & (skeleton_output[i] == 1)).sum()
        tpcom += ((skeleton_gt[i]==skeleton_output_dil[i]) & (skeleton_gt[i]==1)).sum()
        fn += (skeleton_gt[i]==1).sum() - ((skeleton_gt[i]==skeleton_output_dil[i]) & (skeleton_gt[i]==1)).sum()
        fp += (skeleton_output[i]==1).sum() - ((skeleton_output[i] == skeleton_gt_dil[i]) & (skeleton_output[i] == 1)).sum()

    correctness = tpcor/(tpcor+fp)
    completness = tpcom/(tpcom+fn)
    if (completness - completness*correctness + correctness) == 0.0:
        quality = 0.0
    else:    
        quality = completness*correctness/(completness - completness*correctness + correctness)

    return correctness, completness, quality

def compute_precision_recall(pred, gt):
    pred_skel = skeletonize(pred)
    pred_dil = dilation(pred_skel, square(5))
    gt_skel = skeletonize(gt)
    gt_dil = dilation(gt_skel, square(5))
    return compute_metrics([pred_skel], [gt_skel], [pred_dil], [gt_dil])

def calc_iou(pred, gt):
    # Both the pred and gt are binarized.
    # Calculate foreground iou:
    inter = np.logical_and(pred, gt).astype(np.float32)
    union = np.logical_or(pred, gt).astype(np.float32)
    if union.sum() == 0:
        foreground_iou = 0.0
    else:
        foreground_iou = inter.sum() / union.sum()
    return foreground_iou

def binarize(pred, gt):
    pred = (pred > args.thres).astype(np.uint8)
    gt = (gt!=0).astype(np.uint8) * (gt!=255).astype(np.uint8)
    return pred, gt

def evaluate(i):
    
    pd_file = args.pd_path + '%03d_pred.png' % i
    gt_file = args.gt_path + '%03d.png' % i
    if os.path.exists(pd_file):
        pred = imageio.imread(pd_file)
        gt = imageio.imread(gt_file)
        pred, gt = binarize(pred, gt)
        num_gt = gt.sum()
        if num_gt == 0:
            return 1.0, 1.0, 1.0, 1.0
        else:
            foreground_iou = calc_iou(pred, gt)
            correctness, completness, quality = compute_precision_recall(pred, gt)
            print(i, foreground_iou, correctness, completness, quality)
            return foreground_iou, correctness, completness, quality
    else:
        return []

def main():
    num_cores = multiprocessing.cpu_count()
    print('num_cores: ', num_cores)

    p = multiprocessing.Pool(num_cores)
    results = p.map(evaluate, list(range(0, args.max_index)))
    results = [x for x in results if x != []]
    results = np.array(results)
    print(results.shape[0], results.mean(0))

if __name__ == "__main__":
    main()