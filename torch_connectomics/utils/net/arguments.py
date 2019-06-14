import argparse

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
    parser.add_argument('-mi','--model-input', type=str,  default='31,204,204',
                        help='Input size of deep network')
    parser.add_argument('-mo','--model-output', type=str,  default='3,116,116',
                        help='Output size of deep network')
    parser.add_argument('-mp','--model-pad', type=str,  default='0,0,0',
                        help='Pad size of the input data for maximum usage of gt data')

    # model option
    parser.add_argument('-ac','--architecture', help='model architecture')  
    parser.add_argument('-lm','--load-model', type=bool, default=False,
                        help='Use pre-trained model')                
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
        parser.add_argument('-ln','--seg-name',  default='seg-groundtruth2-malis.h5',
                            help='Ground-truth label path')

        parser.add_argument('-vm','--valid-mask', default=None,
                            help='Mask for the train images')

        parser.add_argument('-ft','--finetune', type=bool, default=False,
                            help='Fine-tune on previous model [Default: False]')

        # optimization option
        parser.add_argument('-lt', '--loss', type=int, default=1,
                            help='Loss function')
        parser.add_argument('-lr', type=float, default=0.0001,
                            help='Learning rate')
        parser.add_argument('--iteration-total', type=int, default=1000,
                            help='Total number of iteration')
        parser.add_argument('--iteration-save', type=int, default=100,
                            help='Number of iteration to save')
    elif mode == 'test':
        parser.add_argument('--test-stride', type=str, default='',
                            help='stride during inference')

    args = parser.parse_args()
    return args
