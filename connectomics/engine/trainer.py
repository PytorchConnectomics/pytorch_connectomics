import os, sys, glob, time, itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .solver import *
from connectomics.model import *
from connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from connectomics.data.dataset import build_dataloader
from connectomics.data.utils import blend_gaussian, writeh5

class Trainer(object):
    def __init__(self, cfg, device, mode, output_dir='outputs/', checkpoint=None):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.mode = mode
        
        self.model = build_model(self.cfg, self.device, checkpoint)
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        if self.mode == 'train':
            self.augmentor = build_train_augmentor(self.cfg)
        else:
            self.augmentor = None
        self.dataloader = build_dataloader(self.cfg, self.augmentor, self.mode)
        self.monitor = build_monitor(self.cfg)
        self.criterion = build_criterion(self.cfg, self.device)

        self.dataloader = iter(self.dataloader)

    def train(self):
        # setup
        self.model.train()
        self.monitor.reset()
        self.optimizer.zero_grad()

        for iteration in range(self.cfg.SOLVER.ITERATION_TOTAL):
            iter_total = self.cfg.MODEL.PRE_MODEL_ITER+iteration
            start = time.perf_counter()

            # load data
            batch = next(self.dataloader)
            _, volume, target, weight = batch
            time1 = time.perf_counter()

            # prediction
            volume = torch.from_numpy(volume).to(self.device, dtype=torch.float)
            pred = self.model(volume)
           
            loss = self.criterion.eval(pred, target, weight)

            # compute gradient
            loss.backward()
            if (iteration+1) % self.cfg.SOLVER.ITERATION_STEP == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # logging and update record
            do_vis = self.monitor.update(self.lr_scheduler, iter_total, loss, self.optimizer.param_groups[0]['lr']) 
            if do_vis:
                self.monitor.visualize(volume, torch.from_numpy(target[0]), pred, iter_total)
            #Save model
            if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
                torch.save(self.model.state_dict(), self.output_dir+('/volume_%d%s.pth' % (iter_total, self.cfg.MODEL.FINETUNE)))

            end = time.perf_counter()
            print('[Iteration %05d] Data time: %.5f, Iter time:  %.5f' % (iteration, time1 - start, end - start))

    def test(self):
        if self.cfg.INFERENCE.DO_EVAL:
            self.model.eval()
        else:
            self.model.train()
        volume_id = 0

        ww = blend_gaussian(self.cfg.MODEL.OUTPUT_SIZE)
        NUM_OUT = self.cfg.MODEL.OUT_PLANES
        pad_size = self.cfg.DATASET.PAD_SIZE
        if len(self.cfg.DATASET.PAD_SIZE)==3:
            pad_size = [self.cfg.DATASET.PAD_SIZE[0],self.cfg.DATASET.PAD_SIZE[0],
                        self.cfg.DATASET.PAD_SIZE[1],self.cfg.DATASET.PAD_SIZE[1],
                        self.cfg.DATASET.PAD_SIZE[2],self.cfg.DATASET.PAD_SIZE[2]]
        
        if ("super" in self.cfg.MODEL.ARCHITECTURE):
            output_size = np.array(self.dataloader.dataset.input_size)*np.array(self.cfg.DATASET.SCALE_FACTOR).tolist()
            result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in output_size]
            weight = [np.zeros(x, dtype=np.float32) for x in output_size]
        else:
            result = [np.stack([np.zeros(x, dtype=np.float32) for _ in range(NUM_OUT)]) for x in self.dataloader.dataset.input_size]
            weight = [np.zeros(x, dtype=np.float32) for x in self.dataloader.dataset.input_size]

        # print(result[0].shape, weight[0].shape)

        start = time.time()

        sz = tuple([NUM_OUT] + list(self.cfg.MODEL.OUTPUT_SIZE))
        with torch.no_grad():
            for _, (pos, volume) in enumerate(self.dataloader):
                volume_id += self.cfg.MODEL.BATCH_SIZE
                print('volume_id:', volume_id)

                # for gpu computing
                volume = torch.from_numpy(volume).to(self.device)
                if not self.cfg.INFERENCE.DO_3D:
                    volume = volume.squeeze(1)

                if self.cfg.INFERENCE.AUG_NUM!=0:
                    output = TestAugmentor(self.cfg.INFERENCE.AUG_MODE, self.cfg.INFERENCE.AUG_NUM)
                    output = output(self.model, volume)
                else:
                    output = self.model(volume).cpu().detach().numpy()

                if self.cfg.INFERENCE.MODEL_OUTPUT_ID[0] is not None: # select channel, self.cfg.INFERENCE.MODEL_OUTPUT_ID is a list [None]
                    output = output[self.cfg.INFERENCE.MODEL_OUTPUT_ID[0]]
                if not "super" in self.cfg.MODEL.ARCHITECTURE:
                    for idx in range(output.shape[0]):
                        st = pos[idx]
                        result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += output[idx] * np.expand_dims(ww, axis=0)
                        weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += ww
                else:
                    for idx in range(output.shape[0]):
                        st = pos[idx]
                        st = (np.array(st)*np.array([1]+self.cfg.DATASET.SCALE_FACTOR)).tolist()
                        result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += output[idx] * np.expand_dims(ww, axis=0)
                        weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2], \
                        st[3]:st[3]+sz[3]] += ww

        end = time.time()
        print("Prediction time:", (end-start))

        
        for vol_id in range(len(result)):
            if result[vol_id].ndim > weight[vol_id].ndim:
                weight[vol_id] = np.expand_dims(weight[vol_id], axis=0)
            # For segmentation masks, use uint16
            result[vol_id] = (result[vol_id]/weight[vol_id]*255).astype(np.uint8)
            sz = result[vol_id].shape
            result[vol_id] = result[vol_id][:,
                        pad_size[0]:sz[1]-pad_size[1],
                        pad_size[2]:sz[2]-pad_size[3],
                        pad_size[4]:sz[3]-pad_size[5]]

        if self.output_dir is None:
            return result
        else:
            print('save h5')
            writeh5(os.path.join(self.output_dir, self.cfg.INFERENCE.OUTPUT_NAME), result,['vol%d'%(x) for x in range(len(result))])

