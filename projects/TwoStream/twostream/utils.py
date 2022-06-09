from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
from scipy import stats
from skimage.morphology import binary_dilation


class VAELoss(nn.Module):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    """
    def __init__(self, kld_weight=0.01):
        super().__init__()
        self.kld_weight = kld_weight

    def forward(self, recons, input, mu, log_var) -> dict:
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.kld_weight * kld_loss
        loss_vis = {
            'recon_loss' : recons_loss.detach(),
            'KLD_loss' : self.kld_weight * kld_loss.detach()
        }
        return loss, loss_vis


def collate_fn_patch(batch):
    return BatchOfPatches(batch)


class BatchOfPatches:
    def __init__(self, batch):
        self._handle_batch(*zip(*batch))

    def _handle_batch(self, pos, img, seg):
        self.pos = pos
        self.img = torch.from_numpy(np.stack(img, 0))
        self.seg = torch.from_numpy(np.stack(seg, 0))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.seg = self.seg.pin_memory()
        return self


def calculate_rot(syn, struct_sz=3, return_overlap=False, mode='linear'):
    """Calculate the rotation angle to align masks with different orientations
    """
    assert mode in ['linear', 'siegel', 'theil']
    struct = np.ones((struct_sz, struct_sz), np.uint8)
    pos = binary_dilation(np.logical_and((syn % 2) == 1, syn > 0), struct)
    neg = binary_dilation(np.logical_and((syn % 2) == 0, syn > 0), struct)
    overlap = pos.astype(np.uint8) * neg.astype(np.uint8)
    if overlap.sum() <= 20: # (almost) no overlap
        overlap = (syn!=0).astype(np.uint8)

    pt = np.where(overlap != 0)
    if len(np.unique(pt[0]))==1: # only contains one value
        angle, slope = 90, 1e6 # approximation of infty
        if return_overlap:
            return angle, slope, overlap

        return angle, slope

    if mode == 'linear':
        slope, _, _, _, _ = stats.linregress(pt[0], pt[1])
    elif mode == 'siegel':
        slope, _ = stats.siegelslopes(pt[1], pt[0])
    elif mode == 'theil':
        slope, _, _, _ = stats.theilslopes(pt[1], pt[0])
    else:
        raise ValueError("Unknown mode %s in regression." % mode)
        
    angle = np.arctan(slope) / np.pi * 180
    if return_overlap:
        return angle, slope, overlap
    
    return angle, slope


def rotateIm(image, angle, center=None, scale=1.0, 
             target_type: str = 'img'):
    if angle == 0:
        return image

    interpolation = {'img': cv2.INTER_LINEAR,
                     'mask': cv2.INTER_NEAREST}
    assert image.dtype == np.float32
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image, M, (h, w), 1.0, borderMode=cv2.BORDER_CONSTANT,
        flags=interpolation[target_type]
    )
    return rotated


def rotateIm_polarity(image, label, angle, center=None, scale=1.0):
    rotated = rotateIm(label, angle, center, scale, target_type='mask')
    pos = np.logical_and((rotated % 2) == 1, rotated > 0)
    neg = np.logical_and((rotated % 2) == 0, rotated > 0)
    pos_coord = np.where(pos!=0)
    neg_coord = np.where(neg!=0)
    pos_center = pos_coord[1].mean() if len(pos_coord[1]) >= 1 else None
    neg_center = neg_coord[1].mean() if len(neg_coord[1]) >= 1 else None

    if (pos_center is not None) and (neg_center is not None) and (pos_center > neg_center):
        rotated = np.rot90(rotated, 2, axes=(0, 1))
        angle = angle-180 if angle >= 0 else angle+180

    rotated_image = rotateIm(image, angle, center, scale, target_type='img')
    return rotated_image, rotated, angle


def center_crop(image, out_size):
    if isinstance(out_size, int):
        out_size = (out_size, out_size)

    # channel-last image in (h, w, c) format
    assert image.ndim in [2, 3]
    h, w = image.shape[:2]
    assert h >= out_size[0] and w >= out_size[1]
    margin_h = int((h - out_size[0]) // 2)
    margin_w = int((w - out_size[1]) // 2)

    h0, h1 = margin_h, margin_h + out_size[0]
    w0, w1 = margin_w, margin_w + out_size[1]
    return image[h0:h1, w0:w1]
