import os,sys
import numpy as np
import h5py, time, itertools, datetime

import torch

from torch_connectomics.model.loss import *
from torch_connectomics.utils.net import *
from torch_connectomics.utils.vis import visualize, visualize_aff

def train(args, train_loader, model, device, criterion, optimizer, scheduler, logger, writer):
    for iteration, (_, volume, label, class_weight, _) in enumerate(train_loader):

        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output = label

        if iteration % 10 == 0 and iteration >= 1:
            visualize_aff(volume, label, output, iteration, writer)

        # Terminate
        if iteration >= args.iteration_total:
            break    #     

def main():
    args = get_args(mode='train')

    print('0. initial setup')
    model_io_size, device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train')

    print('2.0 setup model')
    model = setup_model(args, device)
            
    print('2.1 setup loss function')
    criterion = WeightedBCE()   
 
    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                patience=1000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                min_lr=1e-7, eps=1e-08)
    

    print('4. start training')
    train(args, train_loader, model, device, criterion, optimizer, scheduler, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
