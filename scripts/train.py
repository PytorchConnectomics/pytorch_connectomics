import os,sys
import numpy as np
import h5py, time, itertools, datetime

import torch

from torch_connectomics.io import *
from torch_connectomics.run.train import train

def main():
    args = get_args(mode='train')

    print('0. initial setup')
    logger, writer = get_logger(args)

    print('1. setup data')

    print('2.0 setup model')
    model = get_model(args)
            
    print('2.1 setup loss function')
    criterion = get_criterion(args)

    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=1e-5, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                patience=10000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                min_lr=1e-7, eps=1e-08)
    
    print('4. start training')
    if args.do_tile == 0:
        train_loader = get_dataloader(args, 'train')
        train(args, train_loader, model, criterion, optimizer, scheduler, logger, writer)
    else:
        train_dataset = get_dataloader(args, 'train')
        num_chunk = (args.iteration_total+args.data_chunk_iter-1) // args.data_chunk_iter
        for chunk in range(num_chunk):
            train_dataset.updatechunk()
            train_loader = get_dataloader(args, 'train', dataset=train_dataset.dataset)
            train(args, train_loader, model, criterion, optimizer, scheduler, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()

