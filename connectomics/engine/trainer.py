from __future__ import print_function, division
from typing import Optional

import os
import time
import GPUtil
import numpy as np
from yacs.config import CfgNode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from .solver import *
from ..utils.monitor import build_monitor
from ..model import build_model, Criterion
from ..data.augmentation import build_train_augmentor, TestAugmentor
from ..data.dataset import build_dataloader, get_dataset
from ..data.utils import build_blending_matrix, writeh5
from ..data.utils import get_padsize, array_unpad

class Trainer(object):
    r"""Trainer class for supervised learning.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device type. GPUs are recommended for model training and inference.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``). Default: ``'train'``
        rank (int, optional): node rank for distributed training. Default: `None`
        checkpoint (str, optional): the checkpoint file to be loaded. Default: `None`
    """
    def __init__(self, cfg: CfgNode, device: torch.device, mode: str = 'train', 
                 rank: Optional[int] = None, checkpoint: Optional[str] = None):
        assert mode in ['train', 'test']
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        self.mode = mode
        self.rank = rank

        self.model = build_model(self.cfg, self.device, self.rank)
        if checkpoint is not None:
            self.update_checkpoint(checkpoint)

        if self.mode == 'train':
            self.optimizer = build_optimizer(self.cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
            self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
            self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None

            self.augmentor = build_train_augmentor(self.cfg)
            self.criterion = Criterion.build_from_cfg(self.cfg, self.device)
            self.monitor = None
            if self.rank is None or self.rank == 0:
                self.monitor = build_monitor(self.cfg)
                self.monitor.load_config(self.cfg) # show config in tensorboard
                self.monitor.reset()

            self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
            self.total_time = 0                
        else:
            # build test-time augmentor and update output filename
            self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
            if not self.cfg.DATASET.DO_CHUNK_TITLE:
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME
                self.test_filename = self.augmentor.update_name(self.test_filename)

        if cfg.DATASET.DO_CHUNK_TITLE == 0:
            self.dataloader = build_dataloader(self.cfg, self.augmentor, self.mode, rank=rank)
            self.dataloader = iter(self.dataloader)
        else:
            self.dataset = None
            self.dataloader = None

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
                loss = self.criterion.eval(pred, target, weight)
            self._train_misc(loss, pred, volume, target, weight, iter_total)

    def _train_misc(self, loss, pred, volume, target, weight, iter_total):
        self.backward_pass(loss) # backward pass

        # logging and update record
        if self.monitor is not None:
            do_vis = self.monitor.update(self.lr_scheduler, iter_total, loss, 
                                         self.optimizer.param_groups[0]['lr']) 
            if do_vis:
                self.monitor.visualize(volume, target, pred, iter_total)
                # Display GPU stats using the GPUtil package.
                if torch.cuda.is_available(): GPUtil.showUtilization(all=True)

        # Save model
        if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
            self.save_checkpoint(iter_total)

        # update learning rate
        self.lr_scheduler.step(loss) if self.cfg.SOLVER.LR_SCHEDULER_NAME == 'ReduceLROnPlateau' else self.lr_scheduler.step()

        if self.rank is None or self.rank == 0:
            self.iter_time = time.perf_counter() - self.start_time
            self.total_time += self.iter_time
            avg_iter_time = self.total_time / (iter_total+1)
            est_time_left = avg_iter_time * (self.total_iter_nums - iter_total -1) / 3600.0
            print('[Iteration %05d] Data time: %.4fs, Iter time: %.4fs, Avg iter time: %.4fs, Time Left %.2fh.' % (
                iter_total, self.data_time, self.iter_time, avg_iter_time, est_time_left))

        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to 
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del loss, pred

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
            for i, (pos, volume) in enumerate(self.dataloader):
                print('progress: %d/%d batches, total time %.2fs' % 
                      (i+1, len(self.dataloader), time.perf_counter()-start))

                # for gpu computing
                volume = torch.from_numpy(volume).to(self.device)
                if self.cfg.DATASET.DO_2D: volume = volume.squeeze(1)

                output = self.augmentor(self.model, volume)

                for idx in range(output.shape[0]):
                    st = pos[idx]
                    st = (np.array(st)*np.array([1]+output_scale)).astype(int).tolist()
                    out_block = output[idx]
                    if result[st[0]].ndim - output[idx].ndim == 1: # 2d model
                        out_block = out_block[:,None,:]

                    result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += output[idx] * ww[np.newaxis,:]
                    weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += ww

        end = time.perf_counter()
        print("Prediction time: %.2fs" % (end-start))

        for vol_id in range(len(result)):
            if result[vol_id].ndim > weight[vol_id].ndim:
                weight[vol_id] = np.expand_dims(weight[vol_id], axis=0)
            result[vol_id] /= weight[vol_id] # in-place to save memory
            result[vol_id] *= 255
            result[vol_id] = result[vol_id].astype(np.uint8)
            pad_size = (np.array(self.cfg.DATASET.PAD_SIZE) * \
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
                    ['vol%d'%(x) for x in range(len(result))])
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

        else: # standard backward pass
            loss.backward()
            self.optimizer.step()    

    def save_checkpoint(self, iteration: int):
        r"""Save the model checkpoint.
        """
        if self.rank is None or self.rank == 0:
            print("Save model checkpoint at iteration ", iteration)
            state = {'iteration': iteration + 1,
                    # Saving DataParallel or DistributedDataParallel models
                    'state_dict': self.model.module.state_dict(), 
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict()}
                    
            # Saves checkpoint to experiment directory
            filename = 'checkpoint_%05d.pth.tar' % (iteration + 1)
            filename = os.path.join(self.output_dir, filename)
            torch.save(state, filename)

    def update_checkpoint(self, checkpoint: str):
        r"""Update the model with the specified checkpoint file path.
        """
        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint)
        print('checkpoints: ', checkpoint.keys())
        
        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            model_dict = self.model.module.state_dict() # nn.DataParallel
            # 1. filter out unnecessary keys by name
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict (if size match)
            for param_tensor in pretrained_dict:
                if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                    model_dict[param_tensor] = pretrained_dict[param_tensor]  
            # 3. load the new state dict
            self.model.module.load_state_dict(model_dict) # nn.DataParallel

        if not self.cfg.SOLVER.ITERATION_RESTART:
            # update optimizer
            if hasattr(self, 'optimizer') and 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            # update lr scheduler
            if hasattr(self, 'lr_scheduler') and 'lr_scheduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            # load iteration
            if hasattr(self, 'start_iter') and 'iteration' in checkpoint.keys():
                self.start_iter = checkpoint['iteration']

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

        else: # inference mode
            num_chunk = len(self.dataset.chunk_num_ind)
            for chunk in range(num_chunk):
                self.dataset.updatechunk(do_load=False)
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME + '_' + self.dataset.get_coord_name() + '.h5'
                self.test_filename = self.augmentor.update_name(self.test_filename)
                if not os.path.exists(os.path.join(self.output_dir, self.test_filename)):
                    self.dataset.loadchunk()
                    self.dataloader = build_dataloader(self.cfg, self.augmentor, mode, 
                                                       dataset=self.dataset.dataset)
                    self.dataloader = iter(self.dataloader)
                    self.test()
