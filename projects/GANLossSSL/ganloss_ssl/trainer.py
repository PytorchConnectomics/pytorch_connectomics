from __future__ import print_function, division
from collections import OrderedDict
from typing import Optional
import warnings

import os
import time
import copy

import torch
from yacs.config import CfgNode

from connectomics.engine import Trainer
from connectomics.engine.solver import *
from connectomics.model import *
from connectomics.model.loss import GANLoss
from connectomics.model.arch import Discriminator3D
from connectomics.model.utils import SplitActivation, ImagePool
from connectomics.model.build import make_parallel
from connectomics.data.dataset import collate_fn_test, build_dataloader
from connectomics.data.augmentation import build_train_augmentor


class TrainerGANLoss(Trainer):
    """Trainer for semi-supervised segmentation with GAN losses.
    """
    def __init__(self, cfg: CfgNode, **kwargs):
        super().__init__(cfg, **kwargs)
        if self.mode != 'train':
            return # initialization is done if not training

        # 1. build the discriminator
        self.seg_handler = SplitActivation.build_from_cfg(cfg, do_cat=True)
        self.Ds = self.build_discriminator(sum(self.seg_handler.split_channels))
        pool_size = 50 * self.cfg.SOLVER.SAMPLES_PER_BATCH
        self.image_pool = ImagePool(pool_size, self.device, on_cpu=True)

        # 2. build optimizers and loss functions
        self.optimizer_Ds = build_optimizer(self.cfg, self.Ds)
        self.lr_scheduler_Ds = build_lr_scheduler(self.cfg, self.optimizer_Ds)
        self.criterionGAN = GANLoss(gan_mode='lsgan').to(self.device)

        # 3. update configs for the unabeled data and load it
        cfg_unlabel = copy.deepcopy(cfg)
        cfg_unlabel.DATASET.IMAGE_NAME = cfg.UNLABELED.IMAGE_NAME
        cfg_unlabel.DATASET.LABEL_NAME = None
        cfg_unlabel.AUGMENTOR.ADDITIONAL_TARGETS_NAME = None
        cfg_unlabel.AUGMENTOR.ADDITIONAL_TARGETS_TYPE = None
        bs_unlabel = cfg.UNLABELED.SAMPLES_PER_BATCH
        cfg_unlabel.SOLVER.SAMPLES_PER_BATCH = bs_unlabel if bs_unlabel is not None \
            else max(cfg.SOLVER.SAMPLES_PER_BATCH // 2, 1)

        augmentor_unlabeled = build_train_augmentor(cfg_unlabel)
        self.loader_unlabled = iter(build_dataloader(
            cfg_unlabel, augmentor_unlabeled, self.mode, rank=self.rank, cf=collate_fn_test))

    def train(self):
        self.model.train()
        self.Ds.train()

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            volume, target, weight, volume_unlabeled = self.load_batch()
            self.data_time = time.perf_counter() - self.start_time

            # forward labeled and unlabeled images simultaneously
            split_rule = [volume.shape[0], volume_unlabeled.shape[0]]
            pred = self.model(torch.cat([volume, volume_unlabeled], 0))
            pred_labeled, pred_unlabeled = torch.split(pred, split_rule)

            loss, losses_vis = self.criterion(pred_labeled, target, weight)
            if self.cfg.UNLABELED.GAN_UNLABELED_ONLY:
                fake_seg = self.seg_handler(pred_unlabeled)
            else: # apply GAN loss to both labeled and unlabeled samples
                fake_seg = self.seg_handler(pred)               

            loss_gan_weight = float(self.cfg.UNLABELED.GAN_WEIGHT)
            loss_gan = self.criterionGAN(self.Ds(fake_seg), True) * loss_gan_weight
            loss = loss + loss_gan
            losses_vis["GAN_Loss_SSL"] = loss_gan # for visualization only

            self._train_misc(loss, pred, volume, target, weight,
                             iter_total, losses_vis)

            # update discriminator for segmentation
            real_seg = torch.cat(target, 1).to(self.device) # concatenate over channel dim
            fake_seg = self.image_pool.query(fake_seg)
            loss_D = self.update_Ds(real_seg, fake_seg)

            if (iter_total + 1) % self.cfg.MONITOR.ITERATION_NUM[0] == 0 and self.is_main_process:
                print("[Volume %d] discriminator_loss=%0.4f lr=%.5f\n" % (
                    iter_total, loss_D.item(), self.optimizer_Ds.param_groups[0]['lr']))

    def update_Ds(self, real: torch.Tensor, fake: torch.Tensor):
        self.optimizer_Ds.zero_grad()

        pred_real, pred_fake = self.Ds(real), self.Ds(fake.detach())
        loss_real = self.criterionGAN(pred_real, True)
        loss_fake = self.criterionGAN(pred_fake, False)
        loss_D = loss_real + loss_fake

        loss_D.backward()
        self.optimizer_Ds.step()
        self.lr_scheduler_Ds.step()

        return loss_D

    def load_batch(self):
        sample = next(self.dataloader) # labeled data
        volume = sample.out_input
        target, weight = sample.out_target_l, sample.out_weight_l

        sample_unlabeled = next(self.loader_unlabled)
        volume_unlabeled = sample_unlabeled.out_input
        volume, volume_unlabeled = self.to_device(volume, volume_unlabeled)
        return volume, target, weight, volume_unlabeled

    def build_discriminator(self, in_channel: int = 1, **kwargs):
        _D = Discriminator3D(in_channel=in_channel, act_mode=self.cfg.MODEL.ACT_MODE,
                             dilation=self.cfg.UNLABELED.D_DILATION, **kwargs)
        return make_parallel(_D, self.cfg, self.device, self.rank)
        
    def save_checkpoint(self, iteration: int, is_best: bool = False):
        r"""Save the model checkpoint (including the discriminator).
        """
        if self.is_main_process:
            print("Save model checkpoint at iteration ", iteration)
            state = {'iteration': iteration + 1,
                     # Saving DataParallel or DistributedDataParallel models
                     'state_dict': self.model.module.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict()}

            Ds_state = {
                'state_dict': self.Ds.module.state_dict(),
                'optimizer': self.optimizer_Ds.state_dict(),
                'lr_scheduler': self.lr_scheduler_Ds.state_dict()}
            state['Ds'] = Ds_state

            # Saves checkpoint to experiment directory
            filename = 'checkpoint_%05d.pth.tar' % (iteration + 1)
            if is_best:
                filename = 'checkpoint_best.pth.tar'
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def update_checkpoint(self, checkpoint: Optional[str] = None):
        super().update_checkpoint(checkpoint) # checkpoint for self.model

        if checkpoint is None:
            if self.mode == 'test':
                warnings.warn("Test mode without specified checkpoint!")
            return # nothing to load

        if self.mode == 'train':
            if 'Ds' in checkpoint.keys():
                Ds_state = checkpoint['Ds']
                self.Ds.module.load_state_dict(Ds_state['state_dict'])
                self.optimizer_Ds.load_state_dict(Ds_state['optimizer'])
                self.lr_scheduler_Ds.load_state_dict(Ds_state['lr_scheduler'])
            else:
                warnings.warn("Checkpoint is given, but Ds states are not given.")


class TrainerGANLossFeatAlign(TrainerGANLoss):
    """Trainer for semi-supervised segmentation and domain adaptation with 
    with GAN losses (for both segmentation and feature alignment).
    """
    def __init__(self, cfg: CfgNode, **kwargs):
        super().__init__(cfg, **kwargs)
        if self.mode != 'train':
            return # initialization is done if not training

        # build the discriminator(s) for feature alignment
        self.feat_indices = cfg.MODEL.RETURN_FEATS
        filters = list(reversed(cfg.MODEL.FILTERS.copy()))
        self.aligner = OrderedDict()
        for i, k in enumerate(self.feat_indices):
            self.aligner[k] = OrderedDict()
            kwargs_D = self.configure_aligner_D(i)
            self.aligner[k]['D'] = self.build_discriminator(in_channel=filters[k], **kwargs_D)
            self.aligner[k]['optimizer'] = build_optimizer(self.cfg, self.aligner[k]['D'])
            self.aligner[k]['lr_scheduler'] = build_lr_scheduler(self.cfg, self.aligner[k]['optimizer'])

    def configure_aligner_D(self, i):
        kwargs_D = {}
        aligner_cfg = self.cfg.UNLABELED.FEATURE_ALIGNMENT
        def _kwarg_from_cfg(name: str, cfg_name: str):
            if aligner_cfg[cfg_name] is not None:
                if aligner_cfg[cfg_name][i] != "default":
                    kwargs_D[name] = aligner_cfg[cfg_name][i]

        _kwarg_from_cfg('filters', 'D_FILTERS')
        _kwarg_from_cfg('is_isotropic', 'D_IS_ISOTROPIC')
        _kwarg_from_cfg('stride_list', 'STRIDE_LIST')
        return kwargs_D

    def train(self):
        self.model.train()
        self.Ds.train()

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            volume, target, weight, volume_unlabeled = self.load_batch()
            self.data_time = time.perf_counter() - self.start_time

            # forward labeled and unlabeled images simultaneously
            split_rule = [volume.shape[0], volume_unlabeled.shape[0]]
            pred, feats = self.model(torch.cat([volume, volume_unlabeled], 0))
            pred_labeled, pred_unlabeled = torch.split(pred, split_rule)
            if self.is_main_process and i == 0: # show feature dimensions
                print("Feature keys and dimension: ")
                for key in feats.keys():
                    print(key, feats[key].shape)

            loss, losses_vis = self.criterion(pred_labeled, target, weight)
            if self.cfg.UNLABELED.GAN_UNLABELED_ONLY:
                fake_seg = self.seg_handler(pred_unlabeled)
            else: # apply GAN loss to both labeled and unlabeled samples
                fake_seg = self.seg_handler(pred)               

            loss_gan_weight = float(self.cfg.UNLABELED.GAN_WEIGHT)
            loss_gan = self.criterionGAN(self.Ds(fake_seg), True) * loss_gan_weight

            # compute feature alignment losses
            loss_align = 0.0
            feats_labeled, feats_unlabeled = OrderedDict(), OrderedDict()
            for k in self.feat_indices:
                feats_labeled[k], feats_unlabeled[k] = torch.split(feats[k], split_rule)
                loss_align = loss_align + self.criterionGAN(
                    self.aligner[k]['D'](feats_unlabeled[k]), True) * loss_gan_weight

            loss = loss + loss_gan + loss_align
            losses_vis["GAN_Loss_SSL"] = loss_gan
            losses_vis["GAN_Loss_Feat_Align"] = loss_align

            self._train_misc(loss, pred, volume, target, weight,
                             iter_total, losses_vis)

            # update discriminator for segmentation
            real_seg = torch.cat(target, 1).to(self.device) # concatenate over channel dim
            fake_seg = self.image_pool.query(fake_seg)
            loss_D = self.update_Ds(real_seg, fake_seg)

            # update discriminator for feature alignment
            loss_D_aligner = self.update_Ds_aligner(feats_labeled, feats_unlabeled)

            if (iter_total + 1) % self.cfg.MONITOR.ITERATION_NUM[0] == 0 and self.is_main_process:
                print("[Volume %d] D_loss=%0.4f lr=%.5f D_loss_align=%0.4f\n" % (
                    iter_total, loss_D.item(), self.optimizer_Ds.param_groups[0]['lr'], loss_D_aligner))

    def update_Ds_aligner(self, feats_labeled: dict, feats_unlabeled: dict):
        loss_D_aligner = 0.0
        for k in self.feat_indices:
            self.aligner[k]['optimizer'].zero_grad()
            self.optimizer_Ds.zero_grad()

            pred_real = self.aligner[k]['D'](feats_labeled[k].detach())
            pred_fake = self.aligner[k]['D'](feats_unlabeled[k].detach())
            loss_real = self.criterionGAN(pred_real, True)
            loss_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_real + loss_fake) / 2.0

            loss_D.backward()
            self.aligner[k]['optimizer'].step()
            self.aligner[k]['lr_scheduler'].step()
            loss_D_aligner += loss_D.item()

        return loss_D_aligner
