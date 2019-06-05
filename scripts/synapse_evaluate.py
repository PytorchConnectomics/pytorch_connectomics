# Multi-CPU Implementation of Synapse Evaluation 
# based on Intersection over Union (IoU)
# Year 2019
# Zudi Lin

from __future__ import division, print_function
import os
import sys
import glob
import numpy as np
import random
import pickle
import h5py
import time
import argparse
import itertools
import datetime

import torch
import torch.nn as nn
import torch.utils.data

from skimage import measure
from skimage.morphology import dilation, remove_small_objects

class SynEvaluate(torch.utils.data.Dataset):
    def __init__(self, pred_path, gt_path, iou_thres=0.2, min_size=300, margin=(14, 0, 0)):
        # Evaluation of synapse prediction results 
        self.iou_thres = iou_thres
        print('IoU Thres: ', self.iou_thres)
        self.margin = margin # margin of the prediction

        self.gt_vol = np.array(h5py.File(gt_path, 'r')['main'])
        self.gt_vol = self.gt_vol.astype(np.uint16) # convert to uint16
        valid_mask = np.zeros(self.gt_vol.shape, dtype=self.gt_vol.dtype)
        valid_mask[self.margin[0]:valid_mask.shape[0]-self.margin[0], 
                   self.margin[1]:valid_mask.shape[1]-self.margin[1],
                   self.margin[2]:valid_mask.shape[2]-self.margin[2]] = 1
        self.gt_vol = self.gt_vol * valid_mask
        self.pred_vol = np.array(h5py.File(pred_path, 'r')['main'])
        self.pred_vol = self.pred_vol.astype(self.gt_vol.dtype)
        self.pred_vol = self.pred_vol * valid_mask
        #print('gt_vol: ', self.gt_vol.shape, self.gt_vol.dtype)
        #print('pred_vol: ', self.pred_vol.shape, self.pred_vol.dtype)
        assert(self.gt_vol.shape == self.pred_vol.shape)

        # prepare ground-truth volume
        synapse_labels = np.unique(self.gt_vol)[1:]
        #self.num_syn_gt = len(synapse_labels) // 2
        #print('Number of instances in gt: ', self.num_syn_gt, len(synapse_labels))
        #print(synapse_labels)

        self.gt_pos = self.gt_vol.copy()
        self.gt_neg = self.gt_vol.copy()
        self.gt_pos[np.where(self.gt_pos%2==0)] = 0 # keep odd numbers
        self.gt_neg[np.where(self.gt_neg%2==1)] = 0 # keep even numbers
        self.gt_pos_id = np.unique(self.gt_pos)[1:]
        self.gt_neg_id = np.unique(self.gt_neg)[1:]

        self.gt_cleft = self.gt_vol.copy()
        self.gt_cleft = (self.gt_cleft - 1) // 2 * 2 + 1
        # self.gt_vol.dtype = np.uint16
        max_value = np.iinfo(self.gt_cleft.dtype).max
        self.gt_cleft[np.where(self.gt_cleft==max_value)] = 0
        self.gt_id = np.unique(self.gt_cleft)[1:]
        print('Number of instances in gt: ', len(self.gt_id))
        #print(self.gt_id)

        #print(self.gt_id)
        #print(len(self.gt_id))

        #self.search_margin = (4, -4, 64, -64, 64, -64)
        self.search_margin = (0,0,0,0,0,0)

        # prepare the prediction volume
        self.pos_mask = (self.pred_vol == 1).astype(np.uint8)
        self.neg_mask = (self.pred_vol == 2).astype(np.uint8)
        self.syn_mask = np.logical_or(self.pos_mask, self.neg_mask)
        self.syn_mask = (self.syn_mask).astype(np.uint8)

        syn_mask_dilated = dilation(self.syn_mask)
        pred_instances = measure.label(syn_mask_dilated)
        print('Remove instances with size less than %d pixels.' % (min_size))
        self.pred_instances = remove_small_objects(
            pred_instances, min_size=min_size, connectivity=1, in_place=False)
        self.pred_id = np.unique(self.pred_instances)[1:]
        self.num_syn_pred = len(self.pred_id)
        print('Number of instances in prediction: ', self.num_syn_pred)
        print('Maximum label id: ', np.max(self.pred_id))

        # save labelled prediction
        print('Save labelled prediction.')
        fl = h5py.File('pd_instances.h5', 'w')
        fl.create_dataset('main', data=self.pred_instances, compression='gzip')
        fl.close()

        print('Save gt clefts.')
        hk = h5py.File('gt_instances.h5', 'w')
        hk.create_dataset('main', data=self.gt_cleft, compression='gzip')
        hk.close()

    def evaluate_fp(self, label_id):
        TP, FP = 0, 0
        temp = (self.pred_instances == label_id).astype(np.uint8)
        z, y, x = np.where(temp != 0)
        bbox = np.array([np.min(z), np.max(z)+1, 
                         np.min(y), np.max(y)+1, 
                         np.min(x), np.max(x)+1])
        roi = bbox - np.array(self.search_margin)

        gt_temp = self.gt_cleft[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        pr_temp = temp[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

        if len(np.unique(gt_temp)) == 1 and np.unique(gt_temp)[0]==0:
            # no synapse found in gt
            FP += 1
            target_id = 0
        else:
            if len(np.unique(gt_temp)) == 1:
                gt_labels = np.unique(gt_temp)
            else:    
                gt_labels = np.unique(gt_temp)[1:]
            max_iou = 0.0
            target_id = 0
            for gt_idx in gt_labels:
                # syn_roi = (gt_temp == gt_idx).astype(np.uint8)
                # inter = np.logical_and(syn_roi, pr_temp)
                # union = np.logical_or(syn_roi, pr_temp)
                syn_roi = (self.gt_cleft == gt_idx).astype(np.uint8)
                inter = np.logical_and(syn_roi, temp)
                union = np.logical_or(syn_roi, temp)
                iou = float(inter.sum()) / float(union.sum())
                if iou > max_iou:
                    max_iou = iou
                    target_id = gt_idx

            if max_iou > self.iou_thres:
                TP += 1
            else:
                FP += 1

        return TP, FP, target_id

    def evaluate_fn(self, label_id):
        TP, FN = 0, 0
        temp = (self.gt_cleft == label_id).astype(np.uint8)
        z, y, x = np.where(temp != 0)
        bbox = np.array([np.min(z), np.max(z)+1, 
                         np.min(y), np.max(y)+1, 
                         np.min(x), np.max(x)+1])
        roi = bbox - np.array(self.search_margin)

        gt_temp = temp[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        pr_temp = self.pred_instances[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

        if len(np.unique(pr_temp)) == 1 and np.unique(pr_temp)[0]==0:
            # no synapse in prediction
            FN += 1
            target_id = 0
        else:
            if len(np.unique(pr_temp)) == 1:
                pr_labels = np.unique(pr_temp)
            else:    
                pr_labels = np.unique(pr_temp)[1:]
            max_iou = 0.0
            target_id = 0
            for pr_idx in pr_labels:
                # pred_roi = (pr_temp == pr_idx).astype(np.uint8)
                # inter = np.logical_and(pred_roi, gt_temp)
                # union = np.logical_or(pred_roi, gt_temp)
                pred_roi = (self.pred_instances == pr_idx).astype(np.uint8)
                inter = np.logical_and(pred_roi, temp)
                union = np.logical_or(pred_roi, temp)
                iou = float(inter.sum()) / float(union.sum())
                if iou > max_iou:
                    max_iou = iou
                    target_id = pr_idx

            if max_iou > self.iou_thres:
                TP += 1
            else:
                FN += 1

        return TP, FN, target_id

    def __getitem__(self, index):
        if index < self.num_syn_pred:
            label_id = self.pred_id[index]
            TP, FP, target_id = self.evaluate_fp(label_id)
            return TP, FP, 0, True, label_id, target_id
        else:
            label_id = self.gt_id[index-self.num_syn_pred]
            TP, FN, target_id = self.evaluate_fn(label_id)
            return TP, 0, FN, False, target_id, label_id

        # if index < self.num_syn_pred:
        #     label_id = self.pred_id[index]
        #     print(index, label_id)
        #     return 0, 0, 0, True, 0, 0
        # else:
        #     label_id = self.gt_id[index-self.num_syn_pred]
        #     print(index, label_id)
        #     return 0, 0, 0, False, 0, 0

    def __len__(self):  # number of possible position
        return self.num_syn_pred + len(self.gt_id)


def collate_fn_list(batch):
    tp, fp, fn, index, target_id, label_id = zip(*batch)
    return tp, fp, fn, index, target_id, label_id


if __name__ == "__main__":
    print('Start evaluation!')
    #gt_path = '/n/coxfs01/vcg_connectomics/human/roi215/w08_tr5_tc4_label_v3.1_polarity.h5'
    #pd_path = '/n/coxfs01/vcg_connectomics/human/roi215/tim/964355253395_w08_roi215_subset_predictions_20190207_8x8.h5'
    # pred_16nm_path = '/n/coxfs01/vcg_connectomics/human/roi215/tim/964355253395_w08_roi215_subset_predictions_20190207_16x16.h5'
    gt_path = 'jwr_vol2/vol2_gt.h5'
    pd_path = 'jwr_vol2/vol2_unet.h5'
    #pd_path = 'jwr_vol2/vol2_pred.h5'
    print(gt_path)
    print(pd_path)

    TP1 = 0
    TP2 = 0
    FP, FN = 0, 0

    Eva_08nm = SynEvaluate(pd_path, gt_path, iou_thres=float(sys.argv[1]))
    dataloader = torch.utils.data.DataLoader(
        Eva_08nm, batch_size=64, shuffle=False, collate_fn=collate_fn_list,
        num_workers=8, pin_memory=True)

    f1 = open('tpfp.txt', 'w')
    f2 = open('tpfn.txt', 'w')
    for i, (tp, fp, fn, index, target_id, label_id) in enumerate(dataloader):
        print('iteration: ', i)
        for k in range(len(index)):
            FP += fp[k]
            FN += fn[k]
            if index[k] == True:
                TP1 += tp[k]
                f1.write('%d\t%d\t%d\t%d\n' % (tp[k], fp[k], target_id[k], label_id[k]))
            else:
                TP2 += tp[k]
                f2.write('%d\t%d\t%d\t%d\n' % (tp[k], fn[k], target_id[k], label_id[k]))

    print('Report result: ')
    print('pred2gt: %d %d' % (TP1, FP))
    print('gt2pred: %d %d' % (TP2, FN))
    f1.close()
    f2.close()
    print('Finish evaluation.')
