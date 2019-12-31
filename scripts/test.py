import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from torch_connectomics.io import *
from torch_connectomics.run.test import test

def main():
    args = get_args(mode='test')

    print('1. setup dataloader')
    test_loader = get_dataloader(args, 'test')

    print('2. setup model')
    model = get_model(args, exact=True)

    print('3. start testing')
    test(args, test_loader, model)
  
    print('4. finish testing')

if __name__ == "__main__":
    main()
