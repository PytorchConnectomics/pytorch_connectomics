import os,sys
import numpy as np
import h5py, time, argparse, itertools
from scipy import ndimage

import torchvision.utils as vutils

# tensorboardX
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_logger(args):
    # txt logger
    logger = None
    if args.visualize_txt == 1:
        logger = open(args.output+'log.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = None
    if args.visualize_tensorboard == 1:
        writer = SummaryWriter(args.output)
    return logger, writer
