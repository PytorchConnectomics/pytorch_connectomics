import os,sys
import numpy as np
import h5py, time, argparse, itertools, datetime
from scipy import ndimage

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from vcg_connectomics.model.model_zoo import *
from vcg_connectomics.model.loss import *
from vcg_connectomics.data.dataset import AffinityDataset, collate_fn
from vcg_connectomics.utils.net import AverageMeter, init, get_logger
from vcg_connectomics.utils.vis import visualize, visualize_aff

from vcg_connectomics.libs.sync import DataParallelWithCallback

def get_args():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('-t','--train',  default='/n/coxfs01/',
                        help='Input folder (train)')
    parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                        help='Image data path')
    parser.add_argument('-ln','--seg-name',  default='seg-groundtruth2-malis.h5',
                        help='Ground-truth label path')
    parser.add_argument('-o','--output', default='result/train/',
                        help='Output path')
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='I/O size of deep network')

    # model option
    parser.add_argument('-ac','--architecture', help='model architecture')                    
    parser.add_argument('-ft','--finetune', type=bool, default=False,
                        help='Fine-tune on previous model [Default: False]')
    parser.add_argument('-pm','--pre-model', type=str, default='',
                        help='Pre-trained model path')                  

    # optimization option
    parser.add_argument('-lt', '--loss', type=int, default=1,
                        help='Loss function')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--iteration-total', type=int, default=1000,
                        help='Total number of iteration')
    parser.add_argument('--iteration-save', type=int, default=100,
                        help='Number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='Batch size')

    args = parser.parse_args()
    return args

def get_input(args, model_io_size, opt='train'):
    if opt=='train':
        dir_name = args.train.split('@')
        num_worker = args.num_cpu
        img_name = args.img_name.split('@')
        seg_name = args.seg_name.split('@')
        #dis_name = args.distance.split('@')
    else:
        dir_name = args.val.split('@')
        num_worker = 1
        img_name = args.img_name_val.split('@')
        seg_name = args.seg_name_val.split('@')

    print(img_name)
    print(seg_name)

    # may use datasets from multiple folders
    # should be either one or the same as dir_name
    seg_name = [dir_name[0] + x for x in seg_name]
    img_name = [dir_name[0] + x for x in img_name]
    
    # 1. load data
    assert len(img_name)==len(seg_name)
    train_input = [None]*len(img_name)
    train_label = [None]*len(seg_name)
    #train_distance = [None]*len(dis_name)

    # original image is in [0, 255], normalize to [0, 1]
    for i in range(len(img_name)):
        train_input[i] = np.array(h5py.File(img_name[i], 'r')['main'])/255.0
        train_label[i] = np.array(h5py.File(seg_name[i], 'r')['main'])
        train_label[i] = (train_label[i] != 0).astype(np.float32)
        train_input[i] = train_input[i].astype(np.float32)
        #train_distance[i] = train_distance[i].astype(np.float32)
        print("volume shape: ", train_input[i].shape)
        print("label shape: ", train_label[i].shape)
   
    data_aug = False
    print('data augmentation: ', data_aug)
    dataset = AffinityDataset(volume=train_input, label=train_label, sample_input_size=model_io_size,
                              sample_label_size=model_io_size, augmentor = None, mode = 'train')                            
    # to have evaluation during training (two dataloader), has to set num_worker=0
    SHUFFLE = (opt=='train')
    print('batch size: ', args.batch_size)
    img_loader =  torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn,
            num_workers=num_worker, pin_memory=True)
    return img_loader

def train(args, train_loader, model, device, criterion, optimizer, logger, writer):
    record = AverageMeter()
    model.train()

    for iteration, (pos, volume, label, class_weight, _) in enumerate(train_loader):

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

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])

        if iteration % 10 == 0 and iteration >= 1:
            writer.add_scalar('Loss', record.avg, iteration)
            record.reset()
            visualize_aff(volume, label, output, iteration, writer)

        #Save model
        if iteration % args.iteration_save == 0 or iteration >= args.iteration_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (iteration)))

        # Terminate
        if iteration >= args.iteration_total:
            break    #     

def main():
    args = get_args()

    print('0. initial setup')
    model_io_size, device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train')

    print('2.0 setup model')
    model = unetv3(in_channel=1, out_channel=3)
    print('model: ', model.__class__.__name__)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    model = model.to(device)

    print('Fine-tune? ', bool(args.finetune))
    if bool(args.finetune):
        model.load_state_dict(torch.load(args.pre_model))
        print('fine-tune on previous model:')
        print(args.pre_model)
            
    print('2.1 setup loss function')
    criterion = WeightedBCE()   
 
    print('3. setup optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=1e-6, amsgrad=True)

    print('4. start training')
    train(args, train_loader, model, device, criterion, optimizer, logger, writer)
  
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
