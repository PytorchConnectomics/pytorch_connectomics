import argparse
import os, datetime
import torch
import numpy as np

def get_args(mode='train', do_output=True):
    assert mode in ['train', 'test']
    if mode == 'train':
        parser = argparse.ArgumentParser(description='Specify model training arguments.')
    else:
        parser = argparse.ArgumentParser(description='Specify model inference arguments.')
    
    # I/O
    parser.add_argument('-i','--input-path',  default='/n/home12/zijzhao/CREMI/',
                        help='Input folder (train)')
    parser.add_argument('-o','--output-path', default='',
                        help='Output path')

    # data layout: h5 or folders of tiles 
    parser.add_argument('-dct','--do-chunk-tile',  type=int, default=0,
                        help='Data in tile format or not')
    parser.add_argument('-dcn','--data-chunk-num', type=str,  default='1,1,1,1',
                        help='Chunk parameters for tile format: chunk_num (z,y,x), chunk_stride')
    parser.add_argument('-dcni','--data-chunk-num-ind', type=str,  default='',
                        help='Predefined data chunk to iterate through')
    parser.add_argument('-dci','--data-chunk-iter', type=int,  default='1000',
                        help='Chunk parameters for tile format: chunk_iter_num')

    parser.add_argument('-din', '--img-name',  default='im_uint8.h5',
                        help='Image data path')
    parser.add_argument('-dp', '--data-pad', type=str,  default='',
                        help='Pad size of the input data for maximum usage of gt data')
    parser.add_argument('-ds', '--data-scale', type=str,  default='1,1,1',
                        help='Scale size of the input data for different resolutions')

    parser.add_argument('-dam','--data-aug-mode', type=int,  default=2,
                        help='data augmentation mode. 0: none, 1: no shape change, 2: all')
    parser.add_argument('-daz','--data-aug-ztrans', type=int,  default=0,
                        help='apply xz transponse for data augmentation (for isotropic data)')
    parser.add_argument('-dvt','--data-invalid-thres', type=str,  default='0,0',
                        help='number of voxel to exceed for a valid sample')

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
    parser.add_argument('-mpt', '--pre-model', type=str, default='',
                        help='Pre-trained model path')      
    parser.add_argument('-mpi', '--pre-model-iter', type=int, default=0,
                        help='Pre-trained model iteration')      
    parser.add_argument('-mpl', '--pre-model-layer', type=str, default='',
                        help='Pre-trained model layers to be changed')
    parser.add_argument('-mpls', '--pre-model-layer-select', type=str, default='-1',
                        help='Pre-trained model channels to be selected')

    parser.add_argument('-mi', '--model-input', type=str,  default='18,160,160',
                        help='Input size of deep network')
    parser.add_argument('-mo', '--model-output', type=str,  default='',
                        help='Output size of deep network')
    parser.add_argument('-moc', '--model-out-channel', type=int, default=3,
                        help='Number of output channel(s).')

    if mode == 'train':
        parser.add_argument('-dln','--label-name',  default='seg-groundtruth2-malis.h5',
                            help='Ground-truth label path')
        parser.add_argument('-dle','--label-erosion', type=int,  default=0,
                            help='Half Patch size for 2D label erosion')
        parser.add_argument('-dlb','--label-binary', type=bool,  default=False,
                            help='if it is binary label')

        parser.add_argument('-vm','--valid-mask', default=None,
                            help='Mask for the train images')

        parser.add_argument('-ft','--finetune', type=str, default='',
                            help='Fine-tune suffix for model saving')

        # loss/optimization option
        # for each target representation, we have a list of weight, loss-type, loss-weight
        # label, bd, bd-bg, small obj
        parser.add_argument('-to', '--target-opt', type=str, default='0',
                            help='target opt')
        parser.add_argument('-lo', '--loss-opt', type=str, default='1',
                            help='loss function for each target')
        parser.add_argument('-wo', '--weight-opt', type=str, default='0',
                            help='weight for each loss')
        parser.add_argument('-lw', '--loss-weight', type=str, default='1',
                            help='weight for each loss function')
        parser.add_argument('-lro', '--regu-opt', type=str, default='',
                            help='regularization function for prediction')
        parser.add_argument('-lrw', '--regu-weight', type=str, default='',
                            help='regularization weight for each reguluarization function')

        parser.add_argument('-lr', type=float, default=0.001,
                            help='Learning rate')
        parser.add_argument('-it', '--iteration-total', type=int, default=1000,
                            help='Total number of iteration')
        parser.add_argument('-isa', '--iteration-save', type=int, default=100,
                            help='Number of iteration to save')
        parser.add_argument('-ist', '--iteration-step', type=int, default=1,
                            help='Number of steps to update')
    elif mode == 'test':
        parser.add_argument('-tsd', '--test-stride', type=str, default='',
                            help='stride during inference')
        parser.add_argument('-tam', '--test-aug-mode', type=str, default='min',
                            help='use data augmentation at test time: "mean", "min"')
        parser.add_argument('-tan', '--test-aug-num', type=int, default=0,
                            help='use data augmentation 4-fold, 16-fold')
        parser.add_argument('-tid', '--test-id', type=int, default=0,
                            help='test worker id')
        parser.add_argument('-tn', '--test-num', type=int, default=1,
                            help='number of test workers')

    # machine option
    parser.add_argument('-g','--num-gpu', type=int,  default=1,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=1,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=1,
                        help='Batch size')

    parser.add_argument('-ulo','--mon-log-opt', type=str,  default='1,1,0',
                        help='monitor log option: [tensorboard,txt]. default: 1,1,0')
    parser.add_argument('-uvo','--mon-vis-opt', type=str,  default='0,8',
                        help='monitor vis option. default: 0,8')
    parser.add_argument('-ui','--mon-iter-num', type=str,  default='10,500',
                        help='monitor iteration for update: avg-loss,vis. default: 10,500')


    args = parser.parse_args()

    ## pre-process
    if do_output:
        if args.output_path!='': # new folder
            time_now = str(datetime.datetime.now()).split(' ')
            date = time_now[0]
            time = time_now[1].split('.')[0].replace(':','-')
            args.output_path = os.path.join(args.output_path, 'log'+date+'_'+time)
        else:
            if args.pre_model!='':
                args.output_path = args.pre_model[:args.pre_model.rfind('/')]
                if mode =='test': # create test folder
                    args.output_path = os.path.join(args.output_path, 'test_%d'%(args.pre_model_iter))
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)

    # I/O size in (z,y,x), no specified channel number
    args.model_input_size = np.array([int(x) for x in args.model_input.split(',')])
    if args.model_output=='':
        args.model_output_size = args.model_input_size
    else:
        args.model_output_size = np.array([int(x) for x in args.model_output.split(',')])

    # select training machine
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.data_chunk_stride = int(args.data_chunk_num[-1:])==1
    args.data_chunk_num = np.array([int(x) for x in args.data_chunk_num.split(',')[:-1]])
    args.data_chunk_num_ind = np.array([int(x) for x in args.data_chunk_num_ind.split(',')]) if len(args.data_chunk_num_ind)>0 else []

    if args.data_pad=='':
        args.pad_size = args.model_input_size//4
    else:
        args.pad_size = np.array([int(x) for x in args.data_pad.split(',')])

    args.filters = [int(x) for x in args.model_filters.split(',')]
    args.model_pad_mode,args.model_norm_mode,args.model_act_mode = args.model_conv_mode.split(',')
    args.data_scale = np.array([int(x) for x in args.data_scale.split(',')])
    args.data_invalid_thres = np.array([float(x) for x in args.data_invalid_thres.split(',')])
    args.pre_model_layer = args.pre_model_layer.split('@')
    args.pre_model_layer_select = np.array([int(x) for x in args.pre_model_layer_select.split('@')])

    if mode == 'train':
        # each target has a list of loss/weight/loss-weight
        args.target_opt = [x for x in args.target_opt.split(',')]
        # further split by '-'
        args.weight_opt = [[y for y in x.split('-')] for x in args.weight_opt.split(',')]
        args.loss_opt = [[y for y in x.split('-')] for x in args.loss_opt.split(',')]

        args.loss_weight = [[float(y) for y in x.split('-')] for x in args.loss_weight.split(',')]
        assert(len(args.target_opt)==len(args.loss_opt))
        assert(len(args.target_opt)==len(args.loss_weight))

        args.regu_opt = [] if len(args.regu_opt)==0 else [float(x) for x in args.regu_opt.split(',')]
        args.regu_weight = [] if len(args.regu_opt)==0 else [float(x) for x in args.regu_weight.split(',')]
        assert(len(args.regu_opt)==len(args.regu_weight))

        args.mon_log_opt = [int(x) for x in args.mon_log_opt.split(',')]
        args.mon_vis_opt = [int(x) for x in args.mon_vis_opt.split(',')]
        args.mon_iter_num = np.array([int(x) for x in args.mon_iter_num.split(',')])*args.iteration_step
        
    elif mode == 'test':
        # test stride
        if args.test_stride=='': # if not defined, overlap by args.pad_size
            args.test_stride = np.maximum(1, args.model_input_size - args.pad_size)
        else:
            args.test_stride = [int(x) for x in args.test_stride.split(',')]
    print(args)
    return args
