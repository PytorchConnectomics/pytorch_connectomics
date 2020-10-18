import os,sys
import numpy as np
# tensorboardX
from tensorboardX import SummaryWriter
from .visualizer import Visualizer

class Logger(object):
    def __init__(self, log_path='', log_opt=[1,1,0],  batch_size=1):
        self.n = batch_size
        self.reset()
        # tensorboardX
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

    def update(self, val):
        self.val = val
        self.sum += val * self.n
        self.count += self.n
    
    def output(self, iter_total, lr):
        avg = self.sum / self.count
        if self.do_print:
            print('[Iteration %05d] train_loss=%.5f lr=%.5f' % (iter_total, avg, lr))
        if self.log_tb is not None:
            self.log_tb.add_scalar('Loss', avg, iter_total)
            self.log_tb.add_scalar('Learning Rate', lr, iter_total)
        if self.log_txt is not None:
            self.log_txt.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iter_total, avg, lr))
            self.log_txt.flush() 
        return avg

class Monitor(object):
    """Computes and stores the average and current value"""
    def __init__(self, cfg, log_path='', log_opt=[1,1,0,1], vis_opt=[0,8], iter_num=[10,100], 
                 do_2d=False):
        # log_opt: do_tb, do_txt, batch_size, log_iteration
        # vis_opt: vis_type, vis_number_section, do_2d
        self.logger = Logger(log_path, log_opt[:3], log_opt[3])
        self.vis = Visualizer(cfg, vis_opt[0], vis_opt[1], do_2d)
        self.log_iter, self.vis_iter = iter_num
        self.do_vis = False if self.logger.log_tb is None else True

    def update(self, scheduler, iter_total, loss, lr=0.1):
        do_vis = False
        self.logger.update(loss)
        if (iter_total+1) % self.log_iter == 0:
            avg = self.logger.output(iter_total, lr)
            self.logger.reset()
            if (iter_total+1) % self.vis_iter == 0:
                # scheduler.step(avg)
                do_vis = self.do_vis
        return do_vis

    def visualize(self, volume, label, output, iter_total):
        self.vis.visualize(volume, label, output, iter_total, self.logger.log_tb)

    def load_config(self, cfg):
        self.logger.log_tb.add_text('Config', str(cfg), 0)

    def reset(self):
        self.logger.reset()
