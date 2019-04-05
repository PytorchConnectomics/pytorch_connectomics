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
from vcg_connectomics.utils.net import AverageMeter
from vcg_connectomics.utils.vis import visualize, visualize_aff

from libs.sync import DataParallelWithCallback

# tensorboardX
from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='Training Synapse Detection Model')
    # I/O
    parser.add_argument('-t','--train',  default='/n/coxfs01/',
                        help='Input folder (train)')
    # parser.add_argument('-v','--val',  default='',
    #                     help='input folder (test)')
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
    # parser.add_argument('-lr_decay', default='inv,0.0001,0.75',
    #                     help='learning rate decay')
    # parser.add_argument('-betas', default='0.99,0.999',
    #                     help='beta for adam')
    # parser.add_argument('-wd', type=float, default=5e-6,
    #                     help='weight decay')
    parser.add_argument('--volume-total', type=int, default=1000,
                        help='Total number of iteration')
    parser.add_argument('--volume-save', type=int, default=100,
                        help='Number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='Batch size')

    # training settings:
    parser.add_argument('-dw','--distance', default='im_uint8.h5',
                        help='Loss weight based on distance transform.')
    args = parser.parse_args()
    return args

def init(args):
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    # I/O size in (z,y,x), no specified channel number
    model_io_size = np.array([int(x) for x in args.model_input.split(',')])

    # select training machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("output path: ", sn)
    print("device: ", device)

    return model_io_size, device

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
    #dis_name = [x for x in dis_name]
    #print(dis_name)
    #print(img_name)
    #print(seg_name)
    
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
    # dataset = SynapseDataset(volume=train_input, label=train_label, vol_input_size=model_io_size, \
    #                              vol_label_size=model_io_size, data_aug = data_aug, mode = 'train')
    dataset = AffinityDataset(volume=train_input, label=train_label, sample_input_size=model_io_size,
                              sample_label_size=model_io_size, augmentor = None, mode = 'train')                            
    # to have evaluation during training (two dataloader), has to set num_worker=0
    SHUFFLE = (opt=='train')
    print('batch size: ', args.batch_size)
    img_loader =  torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=SHUFFLE, collate_fn = collate_fn,
            num_workers=num_worker, pin_memory=True)
    return img_loader

def get_logger(args):
    log_name = args.output+'/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_'+date+'_'+time
    logger = open(log_name+'.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter('runs/'+log_name)
    return logger, writer

def train(args, train_loader, model, device, criterion, optimizer, logger, writer):
    # switch to train mode
    model.train()
    volume_id = 0

    for _, (pos, volume, label, class_weight, _) in enumerate(train_loader):
        volume_id += args.batch_size

        # if i == 0: print(volume.size())
        # restrict the weight
        # class_weight.clamp(max=1000)

        # for gpu computing
        # print(weight_factor)
        volume, label = volume.to(device), label.to(device)
        class_weight = class_weight.to(device)
        output = model(volume)

        loss = criterion(output, label, class_weight)
        writer.add_scalar('Loss', loss.item(), volume_id) 

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.write("[Volume %d] train_loss=%0.4f lr=%.5f\n" % (volume_id, \
                loss.item(), optimizer.param_groups[0]['lr']))

        # LR update
        #if args.lr > 0:
            #decay_lr(optimizer, args.lr, volume_id, lr_decay[0], lr_decay[1], lr_decay[2])
        
        if volume_id % 1000 < args.batch_size:
            print('visualize: ', volume_id)
            volume_show = volume[0].permute(1,0,2,3)
            output_show = output[0].permute(1,0,2,3)
            label_show = label[0].permute(1,0,2,3)
            visualize_aff(volume_show, label_show, output_show, volume_id, writer)

        if volume_id % args.volume_save < args.batch_size or volume_id >= args.volume_total:
            torch.save(model.state_dict(), args.output+('/volume_%d.pth' % (volume_id)))

        # Terminate
        if volume_id >= args.volume_total:
            break    #     

def main():
    args = get_args()

    print('0. initial setup')
    model_io_size, device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    train_loader = get_input(args, model_io_size, 'train')

    print('2.0 setup model')
    #model = fpn(in_channel=1, out_channel=3)
    model = unetv3(in_channel=1, out_channel=3)
    print('model: ', model.__class__.__name__)
    model = DataParallelWithCallback(model, device_ids=range(args.num_gpu))
    model = model.to(device)

    #from torchsummary import summary
    #summary(model, input_size=(1, 8, 160, 160))

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
