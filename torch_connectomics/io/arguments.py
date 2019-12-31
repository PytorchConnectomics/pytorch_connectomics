import argparse
import os
import torch
import numpy as np

def get_args(mode='train'):
    assert mode in ['train', 'test']
    if mode == 'train':
        parser = argparse.ArgumentParser(description='Specify model training arguments.')
    else:
        parser = argparse.ArgumentParser(description='Specify model inference arguments.')
    
    """
    define tasks
			{0: 'neuron segmentation',
            1: 'synapse detection',
            11: 'synapse polarity detection',
            2: 'mitochondria segmentation',
            22:'mitochondira segmentation with skeleton transform'}    
	"""

    parser.add_argument('--task', type=int, default=0,
                        help='specify the task')

    # I/O
    parser.add_argument('-t','--train',  default='/n/coxfs01/',
                        help='Input folder (train)')
    parser.add_argument('-dn','--img-name',  default='im_uint8.h5',
                        help='Image data path')
    parser.add_argument('-o','--output', default='result/train/',
                        help='Output path')
    parser.add_argument('-dp','--data-pad', type=str,  default='',
                        help='Pad size of the input data for maximum usage of gt data')
    parser.add_argument('-ds','--data-scale', type=str,  default='1,1,1',
                        help='Scale size of the input data for different resolutions')

    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='Input size of deep network')
    parser.add_argument('-mo','--model-output', type=str,  default='3,116,116',
                        help='Output size of deep network')
    parser.add_argument('-ma','--architecture', help='model architecture')  
    parser.add_argument('-mf','--model-filters', type=str,  default='28,36,48,64,80',
                        help='number of filters per unet block')

    # model option
    parser.add_argument('-pm','--pre-model', type=str, default='',
                        help='Pre-trained model path')      
    parser.add_argument('--out-channel', type=int, default=3,
                        help='Number of output channel(s).')

    # machine option
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='Batch size')

    if mode == 'train':
        parser.add_argument('-ln','--label-name',  default='seg-groundtruth2-malis.h5',
                            help='Ground-truth label path')
        parser.add_argument('-le','--label-erosion', type=int,  default=0,
                            help='Half Patch size for 2D label erosion')

        parser.add_argument('-vm','--valid-mask', default=None,
                            help='Mask for the train images')

        parser.add_argument('-ft','--finetune', type=str, default='',
                            help='Fine-tune suffix for model saving')

        # optimization option
        parser.add_argument('-lt', '--loss', type=int, default=1,
                            help='Loss function')
        parser.add_argument('-lr', type=float, default=0.0001,
                            help='Learning rate')
        parser.add_argument('--iteration-total', type=int, default=1000,
                            help='Total number of iteration')
        parser.add_argument('--iteration-save', type=int, default=100,
                            help='Number of iteration to save')
        parser.add_argument('--iteration-step', type=int, default=1,
                            help='Number of steps to update')
    elif mode == 'test':
        parser.add_argument('-tsz', '--test-size', type=str, default='18,160,160',
                            help='input size during inference')
        parser.add_argument('-tsd', '--test-stride', type=str, default='',
                            help='stride during inference')
        parser.add_argument('-tam', '--test-aug-mode', type=str, default='min',
                            help='use data augmentation at test time: "mean", "min"')
        parser.add_argument('-tan', '--test-aug-num', type=int, default=0,
                            help='use data augmentation 4-fold, 16-fold')

    args = parser.parse_args()

    ## pre-process
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    # I/O size in (z,y,x), no specified channel number
    args.model_io_size = np.array([int(x) for x in args.model_input.split(',')])

    # select training machine
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_pad=='':
        args.pad_size = args.model_io_size//2
    else:
        args.pad_size = [int(x) for x in args.data_pad.split(',')]

    args.filters = [int(x) for x in args.model_filters.split(',')]

    args.data_scale = np.array([int(x) for x in args.data_scale.split(',')])

    if mode == 'test':
        # test stride
        if args.test_size=='': # not defined, do default model input size
            args.test_size = args.model_io_size
        else:
            args.test_size = [int(x) for x in args.test_size.split(',')]

        if args.test_stride=='': # not defined, do default 50%
            args.test_stride = np.maximum(1, args.test_size - args.model_io_size // 2)
        else:
            args.test_stride = [int(x) for x in args.test_stride.split(',')]
    print(args)
    return args
