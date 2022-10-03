from __future__ import print_function, division
from typing import Optional
import warnings

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

from connectomics.engine import Trainer
from connectomics.engine.solver import *
from connectomics.model import *

from connectomics.utils.monitor import build_monitor
from connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from connectomics.data.dataset import VolumeDatasetCond
from connectomics.data.utils import build_blending_matrix, get_padsize, array_unpad, writeh5

from .build import build_dataloader, get_dataset


class TrainerCondSeg(Trainer):
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
        self.init_basics(cfg, device, mode, rank)

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
                self.test_filename = self.augmentor.update_name(self.test_filename)

        self.dataset, self.dataloader = None, None
        if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
            self.dataloader = build_dataloader(
                self.cfg, self.augmentor, self.mode, rank=rank, dataset_class=VolumeDatasetCond)
            self.dataloader = iter(self.dataloader)
            if self.mode == 'train' and cfg.DATASET.VAL_IMAGE_NAME is not None:
                self.val_loader = build_dataloader(
                    self.cfg, None, mode='val', rank=rank, dataset_class=VolumeDatasetCond)
        else:
            raise NotImplementedError("Chunk and single mode are not implemented yet")

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

            # center slice of the target volume
            depth = self.cfg.MODEL.INPUT_SIZE[0]
            # we add -1 as indexing starts at zero
            center = int(max(0, (depth//2 if depth % 2 == 0 else depth//2 + 1) - 1))

            # create blanc volume like target and overwrite central slice
            anno_label = torch.zeros_like(target[0])
            anno_label[:,:,center,:,:] = target[0][:,:,center,:,:] 

            # conncatenate source volume with anno label volume along the channel dimension
            volume = torch.cat((volume, anno_label), axis=1)

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

            # splitting the 4 chanel volume in to image, pre-, post-, and joint synapsis
            additional_image_groups = {
                "SYN": [volume[:,1:2,:,:], volume[:,2:3,:,:], volume[:,3:,:,:]]
            }

            if do_vis:
                self.monitor.visualize(
                    volume[:,:1,:,:], target, pred, weight, iter_total,
                    additional_image_groups=additional_image_groups)
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
            info = [
                '[Iteration %05d]' % iter_total, 'Data time: %.4fs,' % self.data_time,
                'Iter time: %.4fs,' % self.iter_time, 'Avg iter time: %.4fs,' % avg_iter_time,
                'Time Left %.2fh.' % est_time_left]
            print(' '.join(info))

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

        channel_size = self.cfg.MODEL.OUT_PLANES

        output_size = [tuple(np.ceil(np.array(x) * np.array(output_scale)).astype(int))
                       for x in self.dataloader._dataset.volume_size]

        result = [np.stack([np.zeros(x, dtype=np.float32)
                            for _ in range(channel_size)]) for x in output_size]

        # empty canvases to which we draw the predictions with shape INPUT_SIZE \
        # before cropping out original BB shape
        canvas = [np.zeros_like(x) for x in result]
        
        print("Total number of batches: ", len(self.dataloader))

        start = time.perf_counter()
        with torch.no_grad():
            for i, sample in enumerate(self.dataloader):
                print('progress: %d/%d batches, total time %.2fs' %
                      (i+1, len(self.dataloader), time.perf_counter()-start))

                poses, volume = sample.pos, sample.out_input
                conditional = torch.reshape(sample.out_target_l, (volume.shape[0],3,volume.shape[2],volume.shape[3],volume.shape[4]))

                print("conditional.shape: ", conditional.shape)
                print("volume.shape: ", volume.shape)
                volume = torch.cat((volume, conditional), axis=1)

                volume = volume.to(self.device, non_blocking=True)
                output = self.augmentor(self.model, volume)

                if torch.cuda.is_available() and i % 50 == 0:
                    GPUtil.showUtilization(all=True)

                # max boundaries
                # ToDo: Currently the inferencing can only handle a single volume 
                zh, yh, xh = self.dataloader._dataset.volume_size[0] 
                # min boundaries
                zl, yl, xl = [0, 0, 0]

                for idx in range(output.shape[0]):
                    
                    # retrive the pos and cropped pose (the pose that was padded to match the INPUT_SIZE)
                    st, stc = poses[idx]

                    # assert that the original bounding box is smaller then the padded one (in some cases the original bb got truncated)
                    assert (stc[1] <= st[1] and stc[2] >= st[2]), f"Stc Z0,Z1 is {stc[1],stc[2]} while st Z0,Z1 is {st[1],st[2]}"
                    assert (stc[3] <= st[3] and stc[4] >= st[4]), f"Stc Z0,Z1 is {stc[3],stc[4]} while st Z0,Z1 is {st[3],st[4]}"
                    assert (stc[5] <= st[5] and stc[6] >= st[6]), f"Stc Z0,Z1 is {stc[5],stc[6]} while st Z0,Z1 is {st[5],st[6]}"

                    # retrive the prediction
                    out_block = output[idx]

                    # unpack the padded bounding box
                    z0, z1, y0, y1, x0, x1 = stc[1:]

                    # binary array that logs which boundaries where violated
                    fix = np.array([z0<zl, z1>zh, y0<yl, y1>yh, x0<xl, x1>xh]).astype(int)
                    # calculate by how much the boundaries where violated
                    dif = np.array([z0-zl, z1-zh, y0-yl, y1-yh, x0-xl, x1-xh])
                    dif = fix*np.abs(dif)
                    
                    z0, y0, x0 = max(z0, zl), max(y0, yl), max(x0, xl)
                    z1, y1, x1 = min(z1, zh), min(y1, yh), min(x1, xh)

                    # draw the prediction to the canvas, cutting away the padding that was added based on boundary violations
                    canvas[st[0]][:,z0:z1, y0:y1, x0:x1] = out_block[:,dif[0]:out_block.shape[1]-dif[1],dif[2]:out_block.shape[2]-dif[3],dif[4]:out_block.shape[3]-dif[5]]

                    # crop out the original bb from the canvas and add it to the results
                    result[st[0]][:,st[1]:st[2], st[3]:st[4], st[5]:st[6]] += canvas[st[0]][:,st[1]:st[2], st[3]:st[4], st[5]:st[6]] 

        end = time.perf_counter()
        print("Prediction time: %.2fs" % (end-start))

        if self.output_dir is None:
            return result
        else:
            print('Final prediction shapes are:')
            for k in range(len(result)):
                print(result[k].shape)
            writeh5(os.path.join(self.output_dir, self.test_filename), result,
                    ['vol%d' % (x) for x in range(len(result))])
            print('Prediction saved as: ', self.test_filename)

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
        checkpoint = torch.load(checkpoint, map_location=self.device)
        print('checkpoints: ', checkpoint.keys())

        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = update_state_dict(
                self.cfg, pretrained_dict, mode=self.mode)
            model_dict = self.model.module.state_dict()  # nn.DataParallel

            # show model keys that do not match pretrained_dict
            if not model_dict.keys() == pretrained_dict.keys():
                warnings.warn("Module keys in model.state_dict() do not exactly "
                              "match the keys in pretrained_dict!")
                for key in model_dict.keys():
                    if not key in pretrained_dict:
                        print(key)

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

