import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from torch_connectomics.utils.net import *
from torch_connectomics.run.test import test

def main():
    args = get_args(mode='test')

    print('0. initial setup')
    model_io_size, device = init(args)
    print('model I/O size:', model_io_size) 

    print('1. setup data')
    test_loader = get_input(args, model_io_size, 'test')

    print('2. setup model')
    model = setup_model(args, device, exact=True)

    print('3. start testing')
    pad_size = model_io_size//2 # 50% overlap 
    test(args, test_loader, model, device, model_io_size, pad_size)
  
    print('4. finish testing')

if __name__ == "__main__":
    main()
