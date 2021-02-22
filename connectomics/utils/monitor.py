import os
import copy
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import matplotlib
from matplotlib import pyplot as plt

from .visualizer import Visualizer
from connectomics.config.utils import convert_cfg_markdown

def build_monitor(cfg):
    time_now = str(datetime.datetime.now()).split(' ')
    date = time_now[0]
    time = time_now[1].split('.')[0].replace(':','-')
    log_path = os.path.join(cfg.DATASET.OUTPUT_PATH, 'log'+date+'_'+time)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    return Monitor(cfg, log_path, cfg.MONITOR.LOG_OPT+[cfg.SOLVER.SAMPLES_PER_BATCH], \
                   cfg.MONITOR.VIS_OPT, cfg.MONITOR.ITERATION_NUM)

class Logger(object):
    def __init__(self, log_path='', log_opt=[1,1,0], batch_size=1):
        self.n = batch_size
        self.reset()
        # tensorboard visualization
        self.log_tb = None
        self.do_print = log_opt[0]==1
        if log_opt[1] > 0:
            self.log_tb = SummaryWriter(log_path)
        # txt
        self.log_txt = None 
        if log_opt[2] > 0:
            self.log_txt = open(log_path+'/log.txt','w') # unbuffered, write instantly

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.val_dict = {}

    def update(self, val, val_dict):
        self.val = val
        self.sum += val * self.n
        self.count += self.n

        for key in val_dict.keys():
            if key in self.val_dict:
                self.val_dict[key] += val_dict[key] * self.n
            else:
                self.val_dict[key] = val_dict[key] * self.n
    
    def output(self, iter_total, lr):
        avg = self.sum / self.count
        if self.do_print:
            print('[Iteration %05d] train_loss=%.5f lr=%.5f' % (iter_total, avg, lr))
        if self.log_tb is not None:
            self.log_tb.add_scalar('Loss', avg, iter_total)
            self.log_tb.add_scalar('Learning Rate', lr, iter_total)
            for key in self.val_dict:
                self.log_tb.add_scalar(key, self.val_dict[key]/self.count, iter_total)

            losses_pie = plot_loss_ratio(self.val_dict)
            self.log_tb.add_figure('Loss Ratio', losses_pie, iter_total)
            plt.close('all')
            
        if self.log_txt is not None:
            self.log_txt.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iter_total, avg, lr))
            self.log_txt.flush() 
        return avg

class Monitor(object):
    """Computes and stores the average and current value"""
    def __init__(self, cfg, log_path='', log_opt=[1,1,0,1], vis_opt=[0,16], iter_num=[10,100]):
        self.logger = Logger(log_path, log_opt[:3], log_opt[3])
        self.vis = Visualizer(cfg, vis_opt[0], vis_opt[1])
        self.log_iter, self.vis_iter = iter_num
        self.do_vis = False if self.logger.log_tb is None else True

        self.reset()

    def update(self, iter_total, loss, losses_vis, lr=0.1):
        do_vis = False
        self.logger.update(loss, losses_vis)
        if (iter_total+1) % self.log_iter == 0:
            avg = self.logger.output(iter_total, lr)
            self.logger.reset()
            if (iter_total+1) % self.vis_iter == 0:
                do_vis = self.do_vis
        return do_vis

    def visualize(self, volume, label, output, iter_total, val=False):
        self.vis.visualize(volume, label, output, iter_total, 
                           self.logger.log_tb, val)

    def load_config(self, cfg):
        self.logger.log_tb.add_text('Config', convert_cfg_markdown(cfg), 0)

    def load_model(self, model: nn.Module, image: torch.Tensor):
        self.logger.log_tb.add_graph(model, image)

    def reset(self):
        self.logger.reset()


def plot_loss_ratio(loss_dict: dict) -> matplotlib.figure.Figure:
    labels = []
    sizes = []
    for key in loss_dict.keys():
        labels.append(key)
        sizes.append(loss_dict[key])

    fig, ax = plt.subplots()
    colors = ['#ff6666', '#ffcc99', '#99ff99', '#66b3ff', '#c2c2f0','#ffb3e6']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig
