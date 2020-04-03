import os,sys
import numpy as np
import torch
import h5py, time, itertools, datetime
from connectomics.io import *
from connectomics.run import test

def main():
    args = get_args(mode='test')

    print('1. setup model')
    model = get_model(args, exact=True)

    print('2. start testing')
    if args.do_chunk_tile == 0:
        test_loader = get_dataloader(args, 'test')
        test(args, test_loader, model)
    else:
        tile_dataset = get_dataset(args, 'test')
        num_chunk = len(tile_dataset.chunk_num_ind)
        for chunk in range(num_chunk):
            tile_dataset.updatechunk(do_load=False)
            sn = 'result-'+tile_dataset.get_coord_name()+'.h5'
            if not os.path.exists(sn):
                tile_dataset.loadchunk()
                test_loader = get_dataloader(args, 'test', dataset=tile_dataset.dataset)
                test(args, test_loader, model, output_name=sn)

    print('4. finish testing')

if __name__ == "__main__":
    main()
