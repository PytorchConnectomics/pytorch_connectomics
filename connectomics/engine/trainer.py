import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import connectomics.model



# class Trainer(object):
#     def __init__(self, cfg, device, output_dir='outputs/'):
#         self.cfg = cfg
#         self.device = device
#         self.output_dir = output_dir
        
#         # make experiment directory and save config.ymal
#         self.experiment_dir = create_exp_dir(cfg, output_dir)

#         self.summary = TensorboardSummary(self.experiment_dir)
#         self.writer = self.summary.create_summary()

#         # build model and solver
#         if self.cfg.MODEL.SEPARATE:
#             self.encoder, self.decoder = build_model(self.cfg, self.device)
#             self.optimizer_encoder = build_optimizer(self.cfg, self.encoder)
#             self.optimizer_decoder = build_optimizer(self.cfg, self.decoder)
#             self.lr_scheduler_encoder = build_lr_scheduler(self.cfg, self.optimizer_encoder)
#             self.lr_scheduler_decoder = build_lr_scheduler(self.cfg, self.optimizer_decoder)
#         else:
#             self.model = build_model(self.cfg, self.device)
#             self.optimizer = build_optimizer(self.cfg, self.model)
#             self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)

#         # build dataloader
#         self.dataloader = build_dataloader(self.cfg)
