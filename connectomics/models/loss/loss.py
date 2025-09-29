from __future__ import print_function, division
from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """DICE loss.
    """
    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    def __init__(self, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.0

        for index in range(pred.size()[0]):
            iflat = pred[index].contiguous().view(-1)
            tflat = target[index].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            if self.power == 1:
                loss += 1 - ((2. * intersection + self.smooth) /
                             (iflat.sum() + tflat.sum() + self.smooth))
            else:
                loss += 1 - ((2. * intersection + self.smooth) /
                             ((iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        if self.power == 1:
            loss = 1 - ((2. * intersection + self.smooth) /
                        (iflat.sum() + tflat.sum() + self.smooth))
        else:
            loss = 1 - ((2. * intersection + self.smooth) /
                        ((iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))
        return loss

    def forward(self, pred, target, weight_mask=None):
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(
                target.size(), pred.size()))

        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:
            loss = self.dice_loss_batch(pred, target)
        return loss


class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, pred, target, weight=None):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).to(pred.device)
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight_mask=None):
        return self.weighted_mse_loss(pred, target, weight_mask)


class WeightedMAE(nn.Module):
    """Mask weighted mean absolute error (MAE) energy function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        loss = F.l1_loss(pred, target, reduction='none')
        loss = loss * weight_mask
        return loss.mean()


class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """

    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight_mask=None):
        return F.binary_cross_entropy(pred, target, weight_mask)


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted binary cross-entropy with logits.
    """

    def __init__(self, size_average=True, reduce=True, eps=0.):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.eps = eps

    def forward(self, pred, target, weight_mask=None):
        return F.binary_cross_entropy_with_logits(pred, target.clamp(self.eps,1-self.eps), weight_mask)


class WeightedCE(nn.Module):
    """Mask weighted multi-class cross-entropy (CE) loss.
    """

    def __init__(self, class_weight: Optional[List[float]] = None):
        super().__init__()
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight)

    def forward(self, pred, target, weight_mask=None):
        # Different from, F.binary_cross_entropy, the "weight" parameter
        # in F.cross_entropy is a manual rescaling weight given to each
        # class. Therefore we need to multiply the weight mask after the
        # loss calculation.
        if self.class_weight is not None:
            self.class_weight = self.class_weight.to(pred.device)

        loss = F.cross_entropy(
            pred, target, weight=self.class_weight, reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


class WeightedLS(nn.Module):
    """Weighted CE loss with label smoothing (LS). The code is based on:
    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    """
    dim = 1

    def __init__(self, classes=10, cls_weights=None, smoothing=0.2):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

        self.weights = 1.0
        if cls_weights is not None:
            self.weights = torch.tensor(cls_weights)

    def forward(self, pred, target, weight_mask=None):
        shape = (1, -1, 1, 1, 1) if pred.ndim == 5 else (1, -1, 1, 1)
        if isinstance(self.weights, torch.Tensor) and self.weights.ndim == 1:
            self.weights = self.weights.view(shape).to(pred.device)

        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist*pred*self.weights, dim=self.dim)
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


class WeightedBCEFocalLoss(nn.Module):
    """Weighted binary focal loss with logits.
    """
    def __init__(self, gamma=2., alpha=0.25, eps=0.):
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, weight_mask=None):
        pred_sig = pred.sigmoid()
        pt = (1-target)*(1-pred_sig) + target * pred_sig
        at = (1-self.alpha) * target + self.alpha * (1-target)
        wt = at * (1 - pt)**self.gamma
        if weight_mask is not None:
            wt *= weight_mask
        # return -(wt * pt.log()).mean() # log causes overflow
        bce = F.binary_cross_entropy_with_logits(pred, target.clamp(self.eps,1-self.eps), reduction='none')
        return (wt *  bce).mean()


class WSDiceLoss(nn.Module):
    def __init__(self, smooth=100.0, power=2.0, v2=0.85, v1=0.15):
        super().__init__()
        self.smooth = smooth
        self.power = power
        self.v2 = v2
        self.v1 = v1

    def dice_loss(self, pred, target):
        iflat = pred.reshape(pred.shape[0], -1)
        tflat = target.reshape(pred.shape[0], -1)
        wt = tflat * (self.v2 - self.v1) + self.v1
        g_pred = wt*(2*iflat - 1)
        g = wt*(2*tflat - 1)
        intersection = (g_pred * g).sum(-1)
        loss = 1 - ((2. * intersection + self.smooth) /
                    ((g_pred**self.power).sum(-1) + (g**self.power).sum(-1) + self.smooth))

        return loss.mean()

    def forward(self, pred, target, weight_mask=None):
        loss = self.dice_loss(pred, target)
        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives (vanilla, lsgan, and wgangp).
    Based on Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self,
                 gan_mode: str = 'lsgan',
                 target_real_label: float = 1.0,
                 target_fake_label: float = 0.0):
        """ Initialize the GANLoss class.

        Args:
            gan_mode (str): the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool): label for a real image
            target_fake_label (bool): label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool):
        """Create label tensors with the same size as the input.

        Args:
            prediction (torch.Tensor): tpyically the prediction from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool):
        """Calculate loss given Discriminator's output and grount truth labels.

        Args:
            prediction (torch.Tensor): tpyically the prediction output from a discriminator
            target_is_real (bool): if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
