from __future__ import print_function, division
from typing import Optional

import time
import torch
import GPUtil
from yacs.config import CfgNode

from connectomics.engine import Trainer
from connectomics.engine.solver import *
from connectomics.model import *
from connectomics.model.build import make_parallel
from connectomics.data.dataset import build_dataloader, collate_fn_test
from connectomics.utils.monitor import build_monitor

from .vae import VAE
from .dataset import VolumeDatasetCenterPatch
from .utils import VAELoss, collate_fn_patch

class TrainerVAE(Trainer):
    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None,
                 checkpoint: Optional[str] = None):
        self.init_basics(cfg, device, mode, rank)
        self.model = self.build_vae(self.cfg)

        if self.mode == 'train':
            self._init_train(checkpoint)
        else:
            self.update_checkpoint(checkpoint)

        # Since only 2d vae models are currently supported, load 2d patches 
        # from 3d volumes or directly load 2d images.
        ts_cfg = self.cfg.TWOSTREAM
        if ts_cfg.IMAGE_VAE:
            self.dataloader = build_dataloader(
                self.cfg, None, self.mode, rank=rank, cf=collate_fn_test)
        else: # mask vae dataloader (depends on bounding boxes)
            dataset_options = {
                "sample_size": ts_cfg.WIDTH,
                "label_type": ts_cfg.LABEL_TYPE}
            self.dataloader = build_dataloader(
                self.cfg, None, self.mode, rank=rank,
                dataset_class=VolumeDatasetCenterPatch,
                dataset_options=dataset_options,
                cf=collate_fn_patch)
        self.dataloader = iter(self.dataloader)

    def _init_train(self, checkpoint):        
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        self.criterion = VAELoss(self.cfg.TWOSTREAM.KLD_WEIGHT).to(self.device)
        self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
        self.update_checkpoint(checkpoint)

        if self.is_main_process:
            self.monitor = build_monitor(self.cfg)
            self.monitor.load_info(self.cfg, self.model)

        self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
        self.total_time = 0

    def train(self):
        self.model.train()

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            # load data
            sample = next(self.dataloader)
            inputs = sample.out_input if self.cfg.TWOSTREAM.IMAGE_VAE else sample.seg
            self.data_time = time.perf_counter() - self.start_time

            # prediction
            inputs = inputs.to(self.device, non_blocking=True)
            recons, _, mu, log_var = self.model(inputs)
            loss, losses_vis = self.criterion(recons, inputs, mu, log_var)

            self._train_misc(loss, recons, inputs, iter_total, losses_vis)

    def _train_misc(self, loss, recons, inputs, iter_total, losses_vis):
        self.backward_pass(loss)  # backward pass
        self.scheduler_step(iter_total, loss)

        # logging and update record
        if hasattr(self, 'monitor'):
            do_vis = self.monitor.update(iter_total, loss, losses_vis,
                                         self.optimizer.param_groups[0]['lr'])
            # visualize inputs and reconstructions
            if do_vis:
                image_groups = {"inputs_recons": [inputs, recons]}
                self.monitor.visualize_image_groups(iter_total, image_groups, is_3d=False)
                if torch.cuda.is_available():
                    GPUtil.showUtilization(all=True)

        if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
            self.save_checkpoint(iter_total)

        if self.is_main_process:
            self.iter_time = time.perf_counter() - self.start_time
            self.total_time += self.iter_time
            avg_iter_time = self.total_time / (iter_total+1-self.start_iter)
            est_time_left = avg_iter_time * \
                (self.total_iter_nums+self.start_iter-iter_total-1) / 3600.0
            info = [
                '[Iteration %05d]' % iter_total, 'Data time: %.4fs,' % self.data_time,
                'Iter time: %.4fs,' % self.iter_time, 'Avg iter time: %.4fs,' % avg_iter_time,
                'Time Left %.2fh.' % est_time_left]
            print(' '.join(info))

        del recons, inputs, loss, losses_vis

    def test(self):
        self.model.eval() if self.cfg.INFERENCE.DO_EVAL else self.model.train()
        raise NotImplementedError

    def build_vae(self, cfg):
        kwargs = {
            # defaults in pytorch-connectomics
            "img_channels": cfg.MODEL.IN_PLANES,
            "act_mode": cfg.MODEL.ACT_MODE,
            "norm_mode": cfg.MODEL.NORM_MODE,
            # unique ones in two-stream project
            "latent_dim": cfg.TWOSTREAM.LATENT_DIM,
            "hidden_dims": cfg.TWOSTREAM.HIDDEN_DIMS,
            "width": cfg.TWOSTREAM.WIDTH,
        }
        model = VAE(**kwargs)
        return make_parallel(model, self.cfg, self.device, self.rank)
