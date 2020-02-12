import os,sys
import numpy as np
import h5py, time, argparse, itertools
from scipy import ndimage
import imageio

import torchvision.utils as vutils

# tensorboardX
from tensorboardX import SummaryWriter

# data io
def writeh5(filename, dtarray, datasetname='main'):
    fid=h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

def readvol(filename, datasetname=''):
    img_suf = filename[filename.rfind('.')+1:]
    if img_suf == 'h5':
        fid = h5py.File(filename, 'r')
        if datasetname=='':
            datasetname = list(fid)[0]
        data = np.array(fid[datasetname])
        fid.close()
    elif 'tif' in img_suf:
        data = imageio.volread(filename).squeeze()
    else:
        raise ValueError('unrecognizable file format for %s'%(filename))

    return data


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
