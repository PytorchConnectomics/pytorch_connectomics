from typing import Optional

import time
import copy
import GPUtil
import random
from yacs.config import CfgNode

import os
import torch
import torch.nn as nn

from connectomics.engine import Trainer
from connectomics.engine.solver import *
from connectomics.model import *
from connectomics.model.loss import GANLoss
from connectomics.model.arch import Discriminator3D
from connectomics.model.utils import SplitActivation, ImagePool
from connectomics.model.build import make_parallel
from connectomics.data.dataset import VolumeDatasetRecon, build_dataloader
from connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from connectomics.utils.monitor import build_monitor

from .utils import *

class TrainerCySGAN(Trainer):
    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None,
                 checkpoint: Optional[str] = None):
        self.init_basics(cfg, device, mode, rank)
        # backward generator is needed for both train and test
        self.Gb = build_model(self.cfg, self.device, rank)

        assert self.mode in ['train', 'test']
        if self.mode == 'train':
            self.start_iter = cfg.MODEL.PRE_MODEL_ITER
            self.seg_handler = SplitActivation.build_from_cfg(cfg, do_cat=True)

            # 1. build models
            self.Gf = copy.deepcopy(self.Gb) # Gf and Gb are identical
            self.Dx = self.build_discriminator(cfg.MODEL.IN_PLANES)
            self.Dy = self.build_discriminator(cfg.MODEL.IN_PLANES)
            self.Ds = self.build_discriminator(sum(self.seg_handler.split_channels))

            pool_size = 20 * self.cfg.SOLVER.SAMPLES_PER_BATCH
            self.image_pool = {}
            for name in ["Dx", "Dy", "Ds"]:
                self.image_pool[name] = ImagePool(pool_size, self.device, on_cpu=True)

            # 2. build optimizers and loss functions
            self.optimizer, self.lr_scheduler = {}, {}
            for name in ["Gf", "Gb", "Dx", "Dy", "Ds"]:
                self.optimizer[name] = build_optimizer(self.cfg, getattr(self, name))
                self.lr_scheduler[name] = build_lr_scheduler(self.cfg, self.optimizer[name])

            # finetune if checkpoint is not None
            # also load the checkpoints for optimizers and lr_scheduler
            self.update_checkpoint(checkpoint)

            self.criterionSeg = Criterion.build_from_cfg(self.cfg, self.device) # supervised
            self.criterionGAN = GANLoss(gan_mode='lsgan').to(self.device)
            self.criterionConst = nn.L1Loss().to(self.device)

            if self.is_main_process:
                self.monitor = build_monitor(self.cfg)
                self.monitor.load_info(self.cfg, self.Gb)

            # 3. build dataloaders (for both labeled and unlabeled domains)
            augmentorX = build_train_augmentor(self.cfg)
            self.loaderX = iter(build_dataloader(
                self.cfg, augmentorX, self.mode, rank=rank,
                dataset_class=VolumeDatasetRecon, cf=collate_fn_trainX))

            # update configs for the unabeled domain
            cfg.DATASET.IMAGE_NAME = cfg.NEW_DOMAIN.IMAGE_NAME
            cfg.DATASET.LABEL_NAME = None
            cfg.AUGMENTOR.ADDITIONAL_TARGETS_NAME = ['recon_image']
            cfg.AUGMENTOR.ADDITIONAL_TARGETS_TYPE = ['img']
            augmentorY = build_train_augmentor(self.cfg)
            self.loaderY = iter(build_dataloader(
                self.cfg, augmentorY, self.mode, rank=rank,
                dataset_class=VolumeDatasetRecon, cf=collate_fn_trainY))

            # 4. misc
            self.total_iter_nums = cfg.SOLVER.ITERATION_TOTAL - self.start_iter
            self.total_time = 0.0

        else:
            assert checkpoint is not None, "checkpoint required in test mode!"
            self.update_checkpoint(checkpoint)
            # build test-time augmentor and update output filename
            self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
            if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME
                self.test_filename = self.augmentor.update_name(self.test_filename)

            self.dataloader = build_dataloader(
                self.cfg, self.augmentor, self.mode, rank=rank)
            self.dataloader = iter(self.dataloader)

    def build_discriminator(self, in_channel: int = 1):
        _D = Discriminator3D(in_channel=in_channel, act_mode=self.cfg.MODEL.ACT_MODE)
        return make_parallel(_D, self.cfg, self.device, self.rank)

    def train(self):
        for name in ["Gf", "Gb", "Dx", "Dy", "Ds"]:
            getattr(self, name).train()

        for i in range(self.total_iter_nums):
            self.start_time = time.perf_counter()
            iter_total = self.start_iter + i
            self.optimizer['Gf'].zero_grad()
            self.optimizer['Gb'].zero_grad()

            imageX, reconX, targetX, weightX, imageY, reconY = self.load_batch()
            self.data_time = time.perf_counter() - self.start_time

            # X->Y loop (X is annotated)
            fakeY = self.Gf(imageX)
            fakeYimg, fakeYseg = torch.tanh(fakeY[:, :1]), fakeY[:, 1:]
            recX = self.Gb(fakeYimg)
            recXimg = torch.tanh(recX[:, :1])
            recXseg = self.Gb(fakeYimg.clone().detach())[:, 1:]

            gan_loss_Gf = self.criterionGAN(self.Dy(fakeYimg), True)
            cyc_loss_X = self.criterionConst(recXimg, reconX)
            seg_loss1, _ = self.criterionSeg(fakeYseg, targetX, weightX)
            seg_loss2, seg_loss2_vis = self.criterionSeg(recXseg, targetX, weightX)
            loss_X2Y = gan_loss_Gf + cyc_loss_X + seg_loss1 + seg_loss2

            # Y->X loop (Y is unlabeled)
            fakeX = self.Gb(imageY)
            fakeXimg, fakeXseg = torch.tanh(fakeX[:, :1]), fakeX[:, 1:]
            recY = self.Gf(fakeXimg)
            recYimg = torch.tanh(recY[:, :1])

            gan_loss_Gb = self.criterionGAN(self.Dx(fakeXimg), True)
            cyc_loss_Y = self.criterionConst(recYimg, reconY)
            loss_Y2X = gan_loss_Gb + cyc_loss_Y

            if self.cfg.NEW_DOMAIN.SEMI_SUP: # self-supervised objectives
                recYseg = self.Gf(fakeXimg.clone().detach())[:, 1:]
                gan_loss_Gb_seg = self.criterionGAN(
                    self.Ds(self.seg_handler(fakeXseg)), True)
                gan_loss_Gf_seg = self.criterionGAN(
                    self.Ds(self.seg_handler(recYseg)), True)
                struct_const = self.criterionConst(
                    self.seg_handler(fakeXseg), self.seg_handler(recYseg))
                loss_Y2X = loss_Y2X + gan_loss_Gb_seg + gan_loss_Gf_seg + struct_const

            # update generators
            loss_G = loss_X2Y + loss_Y2X
            loss_G.backward()
            self.optimizer['Gf'].step()
            self.optimizer['Gb'].step()
            self.lr_scheduler['Gf'].step()
            self.lr_scheduler['Gb'].step()

            # update discriminators
            loss_Dx = self.update_netD(self.Dx, reconX, self.image_pool['Dx'].query(fakeXimg),
                                       self.optimizer['Dx'], self.lr_scheduler['Dx'])
            loss_Dy = self.update_netD(self.Dy, reconY, self.image_pool['Dy'].query(fakeYimg),
                                       self.optimizer['Dy'], self.lr_scheduler['Dy'])

            loss_Ds = 0.0 # no segmentation-based adversarial loss
            if self.cfg.NEW_DOMAIN.SEMI_SUP:
                real_seg = torch.cat(targetX, 1).to(self.device) # concatenate over channel dim
                fake_seg = self.seg_handler(fakeXseg if random.random() > 0.5 else recYseg)
                loss_Ds = self.update_netD(self.Ds, real_seg, self.image_pool['Ds'].query(fake_seg),
                                           self.optimizer['Dy'], self.lr_scheduler['Dy'])
            loss_D = loss_Dx + loss_Dy + loss_Ds # discriminator losses

            self.iter_time = time.perf_counter() - self.start_time
            self.total_time += self.iter_time
            if hasattr(self, 'monitor'): # logging and update record
                do_vis = self.monitor.update(iter_total, seg_loss2, seg_loss2_vis,
                                             self.optimizer['Gb'].param_groups[0]['lr'])
                if do_vis:
                    additional_image_groups = {
                        "XYX": [imageX, fakeYimg, recXimg, reconX],
                        "YXY": [imageY, fakeXimg, recYimg, reconY],
                    }
                    self.monitor.visualize(
                        fakeYimg, targetX, fakeYseg, weightX, iter_total,
                        additional_image_groups=additional_image_groups)
                    if torch.cuda.is_available():
                        GPUtil.showUtilization(all=True)

            if self.is_main_process and iter_total % 10 == 0:
                total_hour = self.total_time / 3600.0
                avg_iter = self.total_time / float(iter_total+1-self.start_iter)
                print("[%06d]" % iter_total,
                      "lossG %.4f" % loss_G.item(),
                      "lossD %.4f" % loss_D.item(),
                      "data_time %.4f" % self.data_time,
                      "iter_time %.4f" % self.iter_time,
                      "avg_iter %.4f" % avg_iter,
                      "total %.2f h" % total_hour)

            if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                self.save_checkpoint(iter_total)

            # Release some GPU memory
            del imageX, reconX, targetX, weightX, imageY, reconY, \
                loss_G, loss_D, fakeY, recX, fakeX, recY

    def load_batch(self):
        # load a batch from the labeled domain
        sampleX = next(self.loaderX)
        imageX, reconX = sampleX.out_input, sampleX.out_recon
        targetX, weightX = sampleX.out_target_l, sampleX.out_weight_l

        # load a batch from the unlabeled domain
        sampleY = next(self.loaderY)
        imageY, reconY = sampleY.out_input, sampleY.out_recon

        imageX, reconX, imageY, reconY = self.to_device(
            imageX, reconX, imageY, reconY)
        return imageX, reconX, targetX, weightX, imageY, reconY

    def update_netD(self, netD, real, fake, optim, lr_sched):
        optim.zero_grad()

        pred_real, pred_fake = netD(real), netD(fake.detach())
        loss_real = self.criterionGAN(pred_real, True)
        loss_fake = self.criterionGAN(pred_fake, False)
        loss_D = loss_real + loss_fake

        loss_D.backward()
        optim.step()
        lr_sched.step()

        return loss_D

    def save_checkpoint(self, iteration: int):
        r"""Save the checkpoints of model, optimizer and lr_scheduler.
        """
        if self.is_main_process:
            state = {}
            print("Save model checkpoint at iteration ", iteration)
            state['iteration'] = iteration + 1
            for name in ["Gf", "Gb", "Dx", "Dy", "Ds"]:
                 state[name] = {
                    'state_dict': getattr(self, name).module.state_dict(),
                    'optimizer': self.optimizer[name].state_dict(),
                    'lr_scheduler': self.lr_scheduler[name].state_dict()}

            # Saves checkpoint to experiment directory
            filename = 'checkpoint_%06d.pth.tar' % (iteration + 1)
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def update_checkpoint(self, checkpoint: Optional[str] = None):
        r"""Update the model with the specified checkpoint file path.
        """
        if checkpoint is None:
            return

        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        print('checkpoints: ', checkpoint.keys())

        if self.mode == 'test':
            # only the backward generator is needed for segmenting the new domain
            assert "Gb" in checkpoint.keys()
            self.Gb.module.load_state_dict(checkpoint["Gb"]["state_dict"])
            return

        # train mode, state dicts should have exact match with modules
        self.start_iter = checkpoint["iteration"]
        for name in ["Gf", "Gb", "Dx", "Dy", "Ds"]:
            getattr(self, name).module.load_state_dict(checkpoint[name]["state_dict"])
            self.optimizer[name].load_state_dict(checkpoint[name]["optimizer"])
            self.lr_scheduler[name].load_state_dict(checkpoint[name]["lr_scheduler"])

    # -----------------------------------------------------------------------------
    # Inference function
    # -----------------------------------------------------------------------------

    def test(self):
        self.model = copy.deepcopy(self.Gb)
        super().test() # reuse the inference function in Trainer
