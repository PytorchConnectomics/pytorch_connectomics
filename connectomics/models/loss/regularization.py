from __future__ import print_function, division
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.

    Args:
        pred (torch.Tensor): foreground logits.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None
    """    
    def forward(self, 
                pred: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        pred = torch.sigmoid(pred)
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=1e-2)
        loss = 1.0 / diff

        if mask is not None:
            loss *= mask
        return loss.mean()


class ForegroundDTConsistency(nn.Module):    
    """Consistency regularization between the binary foreground mask and
    signed distance transform.

    Args:
        pred1 (torch.Tensor): foreground logits.
        pred2 (torch.Tensor): signed distance transform.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None
    """
    def forward(self, 
                pred1: torch.Tensor, 
                pred2: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):

        log_prob_pos = F.logsigmoid(pred1)
        log_prob_neg = F.logsigmoid(-pred1)
        distance = torch.tanh(pred2)
        dist_pos = torch.clamp(distance, min=0.0)
        dist_neg = - torch.clamp(distance, max=0.0)

        loss_pos = - log_prob_pos * dist_pos
        loss_neg = - log_prob_neg * dist_neg
        loss = loss_pos + loss_neg

        if mask is not None:
            loss *= mask
        return loss.mean()


class ContourDTConsistency(nn.Module):
    """Consistency regularization between the instance contour map and
    signed distance transform.

    Args:
        pred1 (torch.Tensor): contour logits.
        pred2 (torch.Tensor): signed distance transform.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None.
    """
    def forward(self, 
                pred1: torch.Tensor, 
                pred2: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):

        contour_prob = torch.sigmoid(pred1)
        distance_abs = torch.abs(torch.tanh(pred2))
        assert contour_prob.shape == distance_abs.shape
        loss = contour_prob * distance_abs
        loss = loss**2

        if mask is not None:
            loss *= mask
        return loss.mean()


class FgContourConsistency(nn.Module):
    """Consistency regularization between the binary foreground map and 
    instance contour map.

    Args:
        pred1 (torch.Tensor): foreground logits.
        pred2 (torch.Tensor): contour logits.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None.
    """
    sobel = torch.tensor([1, 0, -1], dtype=torch.float32)
    eps = 1e-7

    def __init__(self, tsz_h=1) -> None:
        super().__init__()

        self.sz = 2*tsz_h + 1
        self.sobel_x = self.sobel.view(1,1,1,1,3)
        self.sobel_y = self.sobel.view(1,1,1,3,1)

    def forward(self, 
                pred1: torch.Tensor, 
                pred2: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):

        fg_prob = torch.sigmoid(pred1)
        contour_prob = torch.sigmoid(pred2)

        self.sobel_x = self.sobel_x.to(fg_prob.device)
        self.sobel_y = self.sobel_y.to(fg_prob.device)

        # F.conv3d - padding: implicit paddings on both sides of the input. 
        # Can be a single number or a tuple (padT, padH, padW).
        edge_x = F.conv3d(fg_prob, self.sobel_x, padding=(0,0,1))
        edge_y = F.conv3d(fg_prob, self.sobel_y, padding=(0,1,0))

        edge = torch.sqrt(edge_x**2 + edge_y**2 + self.eps)
        edge = torch.clamp(edge, min=self.eps, max=1.0-self.eps)

        # F.pad: the padding size by which to pad some dimensions of input are 
        # described starting from the last dimension and moving forward.
        edge = F.pad(edge, (1,1,1,1,0,0))
        edge = F.max_pool3d(edge, kernel_size=(1, self.sz, self.sz), stride=1)

        assert edge.shape == contour_prob.shape
        loss = F.mse_loss(edge, contour_prob, reduction='none')

        if mask is not None:
            loss *= mask
        return loss.mean()


class NonoverlapReg(nn.Module):
    """Regularization to prevent overlapping prediction of pre- and post-synaptic
    masks in synaptic polarity prediction ("1" in MODEL.TARGET_OPT).

    Args:
        fg_masked (bool): mask the regularization region with predicted cleft. Defaults: True
    """
    def __init__(self, fg_masked: bool = True) -> None:
        super().__init__()
        self.fg_masked = fg_masked

    def forward(self, pred: torch.Tensor):
        # pred in (B, C, Z, Y, X)
        pos = torch.sigmoid(pred[:, 0]) # pre-synaptic
        neg = torch.sigmoid(pred[:, 1]) # post-synaptic
        loss = pos * neg

        if self.fg_masked:
            # masked by the cleft (union of pre and post)
            # detached to avoid decreasing the cleft probability
            loss = loss * torch.sigmoid(pred[:, 2].detach())

        return loss.mean()
