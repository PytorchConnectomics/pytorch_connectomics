import os,sys
import numpy as np
import h5py, time, itertools, datetime

import torch

from vcg_connectomics.model.loss import *
from vcg_connectomics.utils.net import *
from vcg_connectomics.utils.vis import visualize, visualize_aff

def train(args, train_loader, model, device, criterion, optimizer, scheduler, logger, writer):
    record = AverageMeter()
    model.train()

    for iteration, (_, volume, label, class_weight, _) in enumerate(train_loader):

        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output = model(volume)

        loss = criterion(output, label, class_weight)
        record.update(loss, args.batch_size) 

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (iteration, \
                loss.item(), optimizer.param_groups[0]['lr']))

        if iteration % 10 == 0 and iteration >= 1:
            writer.add_scalar('Loss', record.avg, iteration)
            print('[Iteration %d] train_loss=%0.4f lr=%.6f' % (iteration, \
                  record.avg, optimizer.param_groups[0]['lr']))
            scheduler.step(record.avg)
            record.reset()
            visualize_aff(volume, label, output, iteration, writer)
            #print('weight factor: ', weight_factor) # debug
            # debug
            # if iteration < 50:
            #     fl = h5py.File('debug_%d_h5' % (iteration), 'w')
            #     output = label[0].cpu().detach().numpy().astype(np.uint8)
            #     print(output.shape)
            #     fl.create_dataset('main', data=output)
            #     fl.close()

        #Save model
        if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (iteration)))

        # Terminate
        if iteration >= args.iteration_total:
            break    #     

def main():
    args = args = get_args(mode='train')

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
