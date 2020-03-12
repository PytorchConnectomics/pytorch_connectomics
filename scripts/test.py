import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from connectomics.io import *
from connectomics.run.test import test

def main():
    args = get_args(mode='test')

    print('1. setup model')
    model = get_model(args, exact=True)

    print('2. start testing')
    if args.do_tile == 0:
        test_loader = get_dataloader(args, 'test')
        test(args, test_loader, model)
    else:
        test_dataset = get_dataloader(args, 'test')
        num_chunk = len(test_dataset.chunk_num_ind)
        for chunk in range(num_chunk):
            test_dataset.updatechunk()
            test_loader = get_dataloader(args, 'test', dataset=test_dataset.dataset)
            test(args, test_loader, model)

    print('4. finish testing')

if __name__ == "__main__":
    main()
