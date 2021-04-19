from __future__ import print_function, division
from typing import Optional

import os
import time
import math
import GPUtil
import numpy as np
from yacs.config import CfgNode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .solver import *
from ..model import *
from ..utils.monitor import build_monitor
from ..data.augmentation import build_train_augmentor, TestAugmentor
from ..data.dataset import build_dataloader, get_dataset
from ..data.dataset.build import _get_file_list, _make_path_list
from ..data.utils import build_blending_matrix, writeh5
from ..data.utils import get_padsize, array_unpad


class Trainer(object):
    r"""Trainer class for supervised learning.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``). Default: ``'train'``
        rank (int, optional): node rank for distributed training. Default: `None`
        checkpoint (str, optional): the checkpoint file to be loaded. Default: `None`
    """

    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None,
                 checkpoint: Optional[str] = None):

        assert mode in ['train', 'test']
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        self.mode = mode
        self.rank = rank
        self.is_main_process = rank is None or rank == 0
        self.inference_singly = (mode == 'test') and cfg.INFERENCE.DO_SINGLY

        self.model = build_model(self.cfg, self.device, rank)
        if self.mode == 'train':
            self.optimizer = build_optimizer(self.cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
            self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
            self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
            self.update_checkpoint(checkpoint)

            # stochastic weight averaging
            if self.cfg.SOLVER.SWA.ENABLED:
                self.swa_model, self.swa_scheduler = build_swa_model(
                    self.cfg, self.model, self.optimizer)

            self.augmentor = build_train_augmentor(self.cfg)
            self.criterion = Criterion.build_from_cfg(self.cfg, self.device)
            if self.is_main_process:
                self.monitor = build_monitor(self.cfg)
                self.monitor.load_info(self.cfg, self.model)

            self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
            self.total_time = 0
        else:
            self.update_checkpoint(checkpoint)
            # build test-time augmentor and update output filename
            self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
            if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME
                self.test_filename = self.augmentor.update_name(
                    self.test_filename)

        self.dataset, self.dataloader = None, None
        if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
            self.dataloader = build_dataloader(
                self.cfg, self.augmentor, self.mode, rank=rank)
            self.dataloader = iter(self.dataloader)
            if self.mode == 'train' and cfg.DATASET.VAL_IMAGE_NAME is not None:
                self.val_loader = build_dataloader(
                    self.cfg, None, mode='val', rank=rank)

    def train(self):
        r"""Training function of the trainer class.
        """
        self.model.train()

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            # load data
            sample = next(self.dataloader)
            volume = sample.out_input
            target, weight = sample.out_target_l, sample.out_weight_l
            self.data_time = time.perf_counter() - self.start_time

            # prediction
            volume = volume.to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                pred = self.model(volume)
                loss, losses_vis = self.criterion(pred, target, weight)

            self._train_misc(loss, pred, volume, target, weight,
                             iter_total, losses_vis)

        self.maybe_save_swa_model()

    def _train_misc(self, loss, pred, volume, target, weight,
                    iter_total, losses_vis):
        self.backward_pass(loss)  # backward pass

        # logging and update record
        if hasattr(self, 'monitor'):
            do_vis = self.monitor.update(iter_total, loss, losses_vis,
                                         self.optimizer.param_groups[0]['lr'])
            if do_vis:
                self.monitor.visualize(
                    volume, target, pred, weight, iter_total)
                if torch.cuda.is_available():
                    GPUtil.showUtilization(all=True)

        # Save model
        if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
            self.save_checkpoint(iter_total)

        if (iter_total+1) % self.cfg.SOLVER.ITERATION_VAL == 0:
            self.validate(iter_total)

        # update learning rate
        self.maybe_update_swa_model(iter_total)
        self.scheduler_step(iter_total, loss)

        if self.is_main_process:
            self.iter_time = time.perf_counter() - self.start_time
            self.total_time += self.iter_time
            avg_iter_time = self.total_time / (iter_total+1-self.start_iter)
            est_time_left = avg_iter_time * \
                (self.total_iter_nums+self.start_iter-iter_total-1) / 3600.0
            print('[Iteration %05d] Data time: %.4fs, Iter time: %.4fs, Avg iter time: %.4fs, Time Left %.2fh.' % (
                iter_total, self.data_time, self.iter_time, avg_iter_time, est_time_left))

        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del volume, target, pred, weight, loss, losses_vis

    def validate(self, iter_total):
        r"""Validation function of the trainer class.
        """
        if not hasattr(self, 'val_loader'):
            return

        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i, sample in enumerate(self.val_loader):
                volume = sample.out_input
                target, weight = sample.out_target_l, sample.out_weight_l

                # prediction
                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    pred = self.model(volume)
                    loss, _ = self.criterion(pred, target, weight)
                    val_loss += loss.data

        if hasattr(self, 'monitor'):
            self.monitor.logger.log_tb.add_scalar(
                'Validation_Loss', val_loss, iter_total)
            self.monitor.visualize(volume, target, pred,
                                   weight, iter_total, suffix='Val')

        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = val_loss

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(iter_total, is_best=True)

        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del pred, loss, val_loss

        # model.train() only called at the beginning of Trainer.train().
        self.model.train()

    def test(self):
        r"""Inference function of the trainer class.
        """
        self.model.eval() if self.cfg.INFERENCE.DO_EVAL else self.model.train()
        output_scale = self.cfg.INFERENCE.OUTPUT_SCALE
        spatial_size = list(np.ceil(
            np.array(self.cfg.MODEL.OUTPUT_SIZE) *
            np.array(output_scale)).astype(int))
        channel_size = self.cfg.MODEL.OUT_PLANES

        sz = tuple([channel_size] + spatial_size)
        ww = build_blending_matrix(spatial_size, self.cfg.INFERENCE.BLENDING)

        output_size = [tuple(np.ceil(np.array(x) * np.array(output_scale)).astype(int))
                       for x in self.dataloader._dataset.volume_size]
        result = [np.stack([np.zeros(x, dtype=np.float32)
                            for _ in range(channel_size)]) for x in output_size]
        weight = [np.zeros(x, dtype=np.float32) for x in output_size]
        print("Total number of batches: ", len(self.dataloader))

        start = time.perf_counter()
        with torch.no_grad():
            for i, sample in enumerate(self.dataloader):
                print('progress: %d/%d batches, total time %.2fs' %
                      (i+1, len(self.dataloader), time.perf_counter()-start))

                pos, volume = sample.pos, sample.out_input
                volume = volume.to(self.device, non_blocking=True)
                output = self.augmentor(self.model, volume)

                if torch.cuda.is_available() and i % 50 == 0:
                    GPUtil.showUtilization(all=True)

                for idx in range(output.shape[0]):
                    st = pos[idx]
                    st = (np.array(st) *
                          np.array([1]+output_scale)).astype(int).tolist()
                    out_block = output[idx]
                    if result[st[0]].ndim - out_block.ndim == 1:  # 2d model
                        out_block = out_block[:, np.newaxis, :]

                    result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2],
                                  st[3]:st[3]+sz[3]] += out_block * ww[np.newaxis, :]
                    weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2],
                                  st[3]:st[3]+sz[3]] += ww

        end = time.perf_counter()
        print("Prediction time: %.2fs" % (end-start))

        for vol_id in range(len(result)):
            if result[vol_id].ndim > weight[vol_id].ndim:
                weight[vol_id] = np.expand_dims(weight[vol_id], axis=0)
            result[vol_id] /= weight[vol_id]  # in-place to save memory
            result[vol_id] *= 255
            result[vol_id] = result[vol_id].astype(np.uint8)

            if self.cfg.INFERENCE.UNPAD:
                pad_size = (np.array(self.cfg.DATASET.PAD_SIZE) *
                            np.array(output_scale)).astype(int).tolist()
                pad_size = get_padsize(pad_size)
                result[vol_id] = array_unpad(result[vol_id], pad_size)

        if self.output_dir is None:
            return result
        else:
            print('Final prediction shapes are:')
            for k in range(len(result)):
                print(result[k].shape)
            writeh5(os.path.join(self.output_dir, self.test_filename), result,
                    ['vol%d' % (x) for x in range(len(result))])
            print('Prediction saved as: ', self.test_filename)

    def test_singly(self):
        dir_name = _get_file_list(self.cfg.DATASET.INPUT_PATH)
        img_name = _get_file_list(self.cfg.DATASET.IMAGE_NAME)
        assert len(dir_name) == 1

        num_file = len(img_name)
        start_idx = self.cfg.INFERENCE.DO_SINGLY_START_INDEX
        for i in range(start_idx, num_file):
            dataset = get_dataset(
                self.cfg, self.augmentor, self.mode, self.rank,
                dir_name_init=dir_name, img_name_init=[img_name[i]])
            self.dataloader = build_dataloader(
                self.cfg, self.augmentor, self.mode, dataset, self.rank)
            self.dataloader = iter(self.dataloader)

            digits = int(math.log10(num_file))+1
            self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME + \
                '_' + str(i).zfill(digits) + '.h5'
            self.test_filename = self.augmentor.update_name(
                self.test_filename)

            self.test()

    # -----------------------------------------------------------------------------
    # Misc functions
    # -----------------------------------------------------------------------------

    def backward_pass(self, loss):
        if self.cfg.MODEL.MIXED_PRECESION:
            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            self.scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

        else:  # standard backward pass
            loss.backward()
            self.optimizer.step()

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        r"""Save the model checkpoint.
        """
        if self.is_main_process:
            print("Save model checkpoint at iteration ", iteration)
            state = {'iteration': iteration + 1,
                     # Saving DataParallel or DistributedDataParallel models
                     'state_dict': self.model.module.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'lr_scheduler': self.lr_scheduler.state_dict()}

            # Saves checkpoint to experiment directory
            filename = 'checkpoint_%05d.pth.tar' % (iteration + 1)
            if is_best:
                filename = 'checkpoint_best.pth.tar'
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def update_checkpoint(self, checkpoint: Optional[str] = None):
        r"""Update the model with the specified checkpoint file path.
        """
        if checkpoint is None:
            return

        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint)
        print('checkpoints: ', checkpoint.keys())

        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = update_state_dict(
                self.cfg, pretrained_dict, mode=self.mode)
            model_dict = self.model.module.state_dict()  # nn.DataParallel
            # 1. filter out unnecessary keys by name
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict (if size match)
            for param_tensor in pretrained_dict:
                if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                    model_dict[param_tensor] = pretrained_dict[param_tensor]
            # 3. load the new state dict
            self.model.module.load_state_dict(model_dict)  # nn.DataParallel

        if self.mode == 'train' and not self.cfg.SOLVER.ITERATION_RESTART:
            if hasattr(self, 'optimizer') and 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if hasattr(self, 'lr_scheduler') and 'lr_scheduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            if hasattr(self, 'start_iter') and 'iteration' in checkpoint.keys():
                self.start_iter = checkpoint['iteration']

    def maybe_save_swa_model(self):
        if not hasattr(self, 'swa_model'):
            return

        if self.cfg.MODEL.NORM_MODE in ['bn', 'sync_bn']:  # update bn statistics
            for _ in range(self.cfg.SOLVER.SWA.BN_UPDATE_ITER):
                sample = next(self.dataloader)
                volume = sample.out_input
                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    pred = self.swa_model(volume)

        # save swa model
        if self.is_main_process:
            print("Save SWA model checkpoint.")
            state = {'state_dict': self.swa_model.module.state_dict()}
            filename = 'checkpoint_swa.pth.tar'
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def maybe_update_swa_model(self, iter_total):
        if not hasattr(self, 'swa_model'):
            return

        swa_start = self.cfg.SOLVER.SWA.START_ITER
        swa_merge = self.cfg.SOLVER.SWA.MERGE_ITER
        if iter_total >= swa_start and iter_total % swa_merge == 0:
            self.swa_model.update_parameters(self.model)

    def scheduler_step(self, iter_total, loss):
        if hasattr(self, 'swa_scheduler') and iter_total >= self.cfg.SOLVER.SWA.START_ITER:
            self.swa_scheduler.step()
            return

        if self.cfg.SOLVER.LR_SCHEDULER_NAME == 'ReduceLROnPlateau':
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

    # -----------------------------------------------------------------------------
    # Chunk processing for TileDataset
    # -----------------------------------------------------------------------------
    def run_chunk(self, mode: str):
        r"""Run chunk-based training and inference for large-scale datasets.
        """
        self.dataset = get_dataset(self.cfg, self.augmentor, mode)
        if mode == 'train':
            num_chunk = self.total_iter_nums // self.cfg.DATASET.DATA_CHUNK_ITER
            self.total_iter_nums = self.cfg.DATASET.DATA_CHUNK_ITER
            for chunk in range(num_chunk):
                self.dataset.updatechunk()
                self.dataloader = build_dataloader(self.cfg, self.augmentor, mode,
                                                   dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)
                print('start train for chunk %d' % chunk)
                self.train()
                print('finished train for chunk %d' % chunk)
                self.start_iter += self.cfg.DATASET.DATA_CHUNK_ITER
                del self.dataloader
            return

        # inference mode
        num_chunk = len(self.dataset.chunk_ind)
        print("Total number of chunks: ", num_chunk)
        for chunk in range(num_chunk):
            self.dataset.updatechunk(do_load=False)
            self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME + \
                '_' + self.dataset.get_coord_name() + '.h5'
            self.test_filename = self.augmentor.update_name(
                self.test_filename)
            if not os.path.exists(os.path.join(self.output_dir, self.test_filename)):
                self.dataset.loadchunk()
                self.dataloader = build_dataloader(self.cfg, self.augmentor, mode,
                                                   dataset=self.dataset.dataset)
                self.dataloader = iter(self.dataloader)
                self.test()
