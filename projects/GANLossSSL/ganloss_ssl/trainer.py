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
    def __init__(self, cfg: CfgNode, **kwargs):
        super().__init__(cfg, **kwargs)
        if self.mode != 'train':
            return # initialization is done if not training

        # 1. build the discriminator
        self.seg_handler = SplitActivation.build_from_cfg(cfg, do_cat=True)
        self.Ds = self.build_discriminator(sum(self.seg_handler.split_channels))
        pool_size = 20 * self.cfg.SOLVER.SAMPLES_PER_BATCH
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
        cfg_unlabel.SOLVER.SAMPLES_PER_BATCH = max(cfg.SOLVER.SAMPLES_PER_BATCH // 2, 1)

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

            loss_gan = self.criterionGAN(self.Ds(fake_seg), True)
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

    def build_discriminator(self, in_channel: int = 1):
        _D = Discriminator3D(in_channel=in_channel, act_mode=self.cfg.MODEL.ACT_MODE)
        return make_parallel(_D, self.cfg, self.device, self.rank)
        