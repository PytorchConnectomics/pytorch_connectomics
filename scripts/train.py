import os,sys
import numpy as np
import h5py, time, itertools, datetime

import torch

from connectomics.io import *
from connectomics.run import train

def main():
    args = get_args(mode='train')

    print('0. initial setup')


    print('2.0 setup model')
    model = get_model(args)
    criterion = get_criterion(args)
    monitor = get_monitor(args)

    print('3. setup optimizer')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(30000, 35000), gamma=0.1)
    
    print('4. start training')
    if args.do_chunk_tile == 0:
        train_loader = get_dataloader(args, 'train')
        train(args, train_loader, model, criterion, optimizer, scheduler, monitor, pre_iter=args.pre_model_iter)
    else:
        pre_iter = args.pre_model_iter
        tile_dataset = get_dataset(args, 'train')
        num_chunk = (args.iteration_total-pre_iter) // args.data_chunk_iter
        args.iteration_total = args.data_chunk_iter
        for chunk in range(num_chunk):
            tile_dataset.updatechunk()
            train_loader = get_dataloader(args, 'train', dataset=tile_dataset.dataset)
            train(args, train_loader, model, criterion, optimizer, scheduler, monitor, pre_iter=pre_iter)
            pre_iter += args.data_chunk_iter
            del train_loader
  
    print('5. finish training')
    if logger is not None:
        logger.close()
    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()

