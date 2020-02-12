import argparse
import os, datetime
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
    parser.add_argument('-o','--output', default='result/train/',
                        help='Output path')


    # data layout: h5 or folders of tiles 
    parser.add_argument('-dt','--do-tile',  type=int, default=0,
                        help='Data in tile format or not')
    parser.add_argument('-dcn','--data-chunk-num', type=str,  default='1,1,1,1',
                        help='Chunk parameters for tile format: chunk_num, chunk_stride')
    parser.add_argument('-dcni','--data-chunk-num-ind', type=str,  default='',
                        help='Predefined data chunk to iterate through')
    parser.add_argument('-dci','--data-chunk-iter', type=int,  default='1000',
                        help='Chunk parameters for tile format: chunk_iter_num')

    parser.add_argument('-dn', '--img-name',  default='im_uint8.h5',
                        help='Image data path')
    parser.add_argument('-dp', '--data-pad', type=str,  default='',
                        help='Pad size of the input data for maximum usage of gt data')
    parser.add_argument('-ds', '--data-scale', type=str,  default='1,1,1',
                        help='Scale size of the input data for different resolutions')

    parser.add_argument('-daz','--data-aug-ztrans', type=int,  default=0,
                        help='apply xz transponse for data augmentation (for isotropic data)')

    parser.add_argument('-mi', '--model-input', type=str,  default='18,160,160',
                        help='Input size of deep network')
    parser.add_argument('-mo', '--model-output', type=str,  default='18,160,160',
                        help='Output size of deep network')
    parser.add_argument('-ma', '--architecture', help='model architecture')  
    parser.add_argument('-mf', '--model-filters', type=str,  default='28,36,48,64,80',
                        help='number of filters per unet block')
    parser.add_argument('-mhd', '--model-head-depth', type=int,  default=1,
                        help='last decoder head depth')
    parser.add_argument('-mcm', '--model-conv-mode', type=str,  default='rep,bn,elu',
                        help='convolution layer mode: padding,normalization,activation')
    parser.add_argument('-me', '--model-embedding', type=int, default=1,
                        help='do 2d embedding or not')  

    # model option
    parser.add_argument('-pm', '--pre-model', type=str, default='',
                        help='Pre-trained model path')      
    parser.add_argument('-oc', '--out-channel', type=int, default=3,
                        help='Number of output channel(s).')

    # machine option
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='Batch size')

    parser.add_argument('-vtb','--visualize-tensorboard', type=int,  default=1,
                        help='tensorboard visualization. default: 1')
    parser.add_argument('-vtxt','--visualize-txt', type=int,  default=0,
                        help='txt logging. default: 0')

    if mode == 'train':
        parser.add_argument('-ln','--label-name',  default='seg-groundtruth2-malis.h5',
                            help='Ground-truth label path')
        parser.add_argument('-le','--label-erosion', type=int,  default=0,
                            help='Half Patch size for 2D label erosion')
        parser.add_argument('-lb','--label-binary', type=bool,  default=False,
                            help='if it is binary label')

        parser.add_argument('-vm','--valid-mask', default=None,
                            help='Mask for the train images')

        parser.add_argument('-ft','--finetune', type=str, default='',
                            help='Fine-tune suffix for model saving')

        # loss/optimization option
        parser.add_argument('-lt', '--loss-type', type=int, default=0,
                            help='Loss function')
        parser.add_argument('-lw', '--loss-weight-opt', type=int, default=0,
                            help='Type of weight for rebalancing')
        parser.add_argument('-lwv', '--loss-weight-val', type=str, default='1,1,1,1',
                            help='weight value')
        parser.add_argument('-lr', type=float, default=0.0001,
                            help='Learning rate')
        parser.add_argument('-it', '--iteration-total', type=int, default=1000,
                            help='Total number of iteration')
        parser.add_argument('-isa', '--iteration-save', type=int, default=100,
                            help='Number of iteration to save')
        parser.add_argument('-ist', '--iteration-step', type=int, default=1,
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
    time_now = str(datetime.datetime.now()).split(' ')
    date = time_now[0]
    time = time_now[1].split('.')[0].replace(':','-')
    args.output += '/log'+date+'_'+time+'/'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # I/O size in (z,y,x), no specified channel number
    args.model_io_size = np.array([int(x) for x in args.model_input.split(',')])

    # select training machine
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.data_chunk_stride = int(args.data_chunk_num[-1:])==1
    args.data_chunk_num = np.array([int(x) for x in args.data_chunk_num.split(',')[:-1]])
    args.data_chunk_num_ind = np.array([int(x) for x in args.data_chunk_num_ind.split(',')]) if len(args.data_chunk_num_ind)>0 else []

    if args.data_pad=='':
        args.pad_size = args.model_io_size//2
    else:
        args.pad_size = [int(x) for x in args.data_pad.split(',')]

    args.filters = [int(x) for x in args.model_filters.split(',')]
    args.model_pad_mode,args.model_norm_mode,args.model_act_mode = args.model_conv_mode.split(',')

    args.data_scale = np.array([int(x) for x in args.data_scale.split(',')])

    if mode == 'train':
        args.loss_weight_val = np.array([int(x) for x in args.loss_weight_val.split(',')])
    elif mode == 'test':
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
