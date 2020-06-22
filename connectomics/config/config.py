import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()

_C.SYSTEM.NUM_GPUS = 4

_C.SYSTEM.NUM_CPUS = 4

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# Model architectures defined in the package: unet_super, super, fpn, unet_residual_3d
_C.MODEL.ARCHITECTURE = 'unet_residual_3d' 

# Number of filters per unet block
_C.MODEL.FILTERS = [28, 36, 48, 64, 80] 

_C.MODEL.TARGET_OPT = ['0']

_C.MODEL.WEIGHT_OPT = [['1']]

# Choose the right loss function for each target: 
# 'WeightedMSE', 'WeightedBCE', 'JaccardLoss', 'DiceLoss'
_C.MODEL.LOSS_OPTION = [['WeightedBCE']]

# Weight for each loss function
_C.MODEL.LOSS_WEIGHT = [[1.0]]

# Define the number of input channels. Usually EM images are
# single-channel gray-scale image. 
_C.MODEL.IN_PLANES = 1 

# Define the number of output channels.
_C.MODEL.OUT_PLANES = 1 

# Padding mode, possible options: 'zeros','circular', 'rep'
_C.MODEL.PAD_MODE = 'rep' 

# Normalization mode, possible options: 'bn', 'abn', 'in', 'bin'
_C.MODEL.NORM_MODE = 'bn'

# Activation mode, possible options: 'relu', 'elu', 'leaky'
_C.MODEL.ACT_MODE = 'elu'

# If MODEL.EMBEDDING = 1 will do embedding
_C.MODEL.EMBEDDING = 1

# Last decoder head depth
_C.MODEL.HEAD_DEPTH = 1

_C.MODEL.INPUT_SIZE = [8, 256, 256]

_C.MODEL.OUTPUT_SIZE = [8, 256, 256]

_C.MODEL.REGU_OPT = []

_C.MODEL.REGU_WEIGHT = []

# Fine-tune suffix for model saving
_C.MODEL.FINETUNE = ''

# Exact matching: the weights shape in pretrain model and current model are identical
_C.MODEL.EXACT = True

_C.MODEL.SIZE_MATCH = True

_C.MODEL.PRE_MODEL = ''

_C.MODEL.PRE_MODEL_LAYER = ''

_C.MODEL.PRE_MODEL_ITER = 0

_C.MODEL.PRE_MODEL_LAYER_SELECT = -1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

# Scale ratio of the input data for different resolutions.
# Using a DATA_SCALE of [1., 0.5, 0.5] will downsample the 
# original image by two times (e.g., 4nm -> 8nm).
_C.DATASET.DATA_SCALE = [1., 1., 1.]

# Scaling factor for super resolution
_C.DATASET.SCALE_FACTOR = [2, 3, 3]

# Specify the data path in the *.yaml files for different experiments.
_C.DATASET.IMAGE_NAME = ''

_C.DATASET.LABEL_NAME = ''

_C.DATASET.INPUT_PATH = ''

_C.DATASET.OUTPUT_PATH = ''

# Padding size for the input volumes
_C.DATASET.PAD_SIZE = [2, 64, 64] 

# Half Patch size for 2D label erosion
_C.DATASET.LABEL_EROSION = 0

# If it's a binary label
_C.DATASET.LABEL_BINARY = False

_C.DATASET.LABEL_MAG = 0

# Data in tile format or not.
_C.DATASET.DO_CHUNK_TITLE = 0

# Chunk parameters for tile format: chunk_num (z,y,x), chunk_stride
_C.DATASET.DATA_CHUNK_NUM = [1, 1, 1]

# Predefined data chunk to iterate through
_C.DATASET.DATA_CHUNK_NUM_IND = []

# Boolean variable, euqal to 'int(args.data_chunk_num[-1:])==1'
_C.DATASET.DATA_CHUNK_STRIDE = True

# Chunk parameters for tile format: chunk_iter_num
_C.DATASET.DATA_CHUNK_ITER = 1000

# Number of voxel to exceed for a valid sample
_C.DATASET.DATA_INVALID_THRES = [0., 0.]

_C.DATASET.PRE_LOAD_DATA = [None,None,None]

# For some datasets the foreground mask is sparse in the volume. Therefore
# we perform reject sampling to decrease (all completely avoid) regions
# without foreground masks. Set REJECT_SAMPLING.SIZE_THRES = -1 to disable.
_C.DATASET.REJECT_SAMPLING = CN()
_C.DATASET.REJECT_SAMPLING.SIZE_THRES = -1
# By default, we conduct rejection sampling before data augmentation to
# save data loading time. However, the final output after augmentation
# may not satisfy the SIZE_THRES. Thus some tasks require AFTER_AUG=True.
_C.DATASET.REJECT_SAMPLING.AFTER_AUG = False
_C.DATASET.REJECT_SAMPLING.P = 0.95

# -----------------------------------------------------------------------------
# Augmentor
# -----------------------------------------------------------------------------
_C.AUGMENTOR = CN()

# The nearest interpolation for the label mask during data augmentation
# can result in masks with coarse boundaries. Thus we apply Gaussian filtering
# to smooth the object boundary (default: True).
_C.AUGMENTOR.SMOOTH = True

_C.AUGMENTOR.ROTATE = CN({"ENABLED": True})
# Probability of applying the rotation augmentation
_C.AUGMENTOR.ROTATE.P = 0.5

_C.AUGMENTOR.RESCALE = CN({"ENABLED": True})
# Probability of applying the rescale augmentation
_C.AUGMENTOR.RESCALE.P = 0.5

_C.AUGMENTOR.FLIP = CN({"ENABLED": True})
# Probability of applying the flip augmentation (always applied by default)
_C.AUGMENTOR.FLIP.P = 1.0
# Conducting x-z and y-z flip only when the dataset is isotropic. 
_C.AUGMENTOR.FLIP.DO_ZTRANS = 0

_C.AUGMENTOR.ELASTIC = CN({"ENABLED": True})
# Maximum pixel-moving distance of elastic transformation
_C.AUGMENTOR.ELASTIC.ALPHA = 16.0
# Standard deviation of the Gaussian filter
_C.AUGMENTOR.ELASTIC.SIGMA = 4.0
# Probability of applying the elastic augmentation
_C.AUGMENTOR.ELASTIC.P = 0.75

_C.AUGMENTOR.GRAYSCALE = CN({"ENABLED": True})
# Probability of applying the grayscale augmentation
_C.AUGMENTOR.GRAYSCALE.P = 0.75

_C.AUGMENTOR.MISSINGPARTS = CN({"ENABLED": True})
# Probability of applying the missingparts augmentation
_C.AUGMENTOR.MISSINGPARTS.P = 0.9

_C.AUGMENTOR.MISSINGSECTION = CN({"ENABLED": True})
# Probability of applying the missingsection augmentation
_C.AUGMENTOR.MISSINGSECTION.P = 0.5

_C.AUGMENTOR.MISALIGNMENT = CN({"ENABLED": True})
# Probability of applying the misalignment augmentation
_C.AUGMENTOR.MISALIGNMENT.P = 0.5
# Maximum pixel displacement in each direction (x and y) (int)
_C.AUGMENTOR.MISALIGNMENT.DISPLACEMENT = 16
_C.AUGMENTOR.MISALIGNMENT.ROTATE_RATIO = 0.5

_C.AUGMENTOR.MOTIONBLUR = CN({"ENABLED": True})
# Probability of applying the motion blur augmentation
_C.AUGMENTOR.MOTIONBLUR.P = 0.5
# Number of sections along z dimension to apply motion blur
_C.AUGMENTOR.MOTIONBLUR.SECTIONS = 2
# Kernel size of motion blur
_C.AUGMENTOR.MOTIONBLUR.KERNEL_SIZE = 11

_C.AUGMENTOR.CUTBLUR = CN({"ENABLED": True})
# Probability of applying the CutBlur augmentation
_C.AUGMENTOR.CUTBLUR.P = 0.5
_C.AUGMENTOR.CUTBLUR.LENGTH_RATIO = 0.4
_C.AUGMENTOR.CUTBLUR.DOWN_RATIO_MIN = 2.0
_C.AUGMENTOR.CUTBLUR.DOWN_RATIO_MAX = 8.0
_C.AUGMENTOR.CUTBLUR.DOWNSAMPLE_Z = False

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

# Specify the learning rate scheduler.
_C.SOLVER.LR_SCHEDULER_NAME = "MultiStepLR"

_C.SOLVER.ITERATION_STEP = 1

_C.SOLVER.ITERATION_SAVE = 5000

_C.SOLVER.ITERATION_TOTAL = 40000

# Whether or not to restart training from iteration 0 regardless
# of the 'iteration' key in the checkpoint file. This option only 
# works when a pretrained checkpoint is loaded (default: False).
_C.SOLVER.ITERATION_RESTART = False

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.MOMENTUM = 0.9

# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY = 0.0001

_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

# The iteration number to decrease learning rate by GAMMA
_C.SOLVER.GAMMA = 0.1

# should be a tuple like (30000,)
_C.SOLVER.STEPS = (30000, 35000)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000

_C.SOLVER.WARMUP_ITERS = 1000

_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of samples per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.SOLVER.SAMPLES_PER_BATCH = 16

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# -----------------------------------------------------------------------------
# Monitor
# -----------------------------------------------------------------------------
_C.MONITOR = CN()

_C.MONITOR.LOG_OPT = [1, 1, 0]

_C.MONITOR.VIS_OPT = [0, 8]

_C.MONITOR.ITERATION_NUM = [10, 50]

# # -----------------------------------------------------------------------------
# # Inference
# # -----------------------------------------------------------------------------
_C.INFERENCE = CN()

_C.INFERENCE.INPUT_SIZE = [8, 256, 256]

_C.INFERENCE.OUTPUT_SIZE = [8, 256, 256]

_C.INFERENCE.IMAGE_NAME = ''

_C.INFERENCE.OUTPUT_PATH = ''

_C.INFERENCE.OUTPUT_NAME = 'result.h5'

_C.INFERENCE.PAD_SIZE = [8, 64, 64]

_C.INFERENCE.STRIDE = [1, 192, 192]

_C.INFERENCE.AUG_MODE = 'mean'

_C.INFERENCE.AUG_NUM = 4

_C.INFERENCE.DO_EVAL = True

_C.INFERENCE.DO_3D = True

# If not None then select channel of output
_C.INFERENCE.MODEL_OUTPUT_ID = [None] 

# Number of test workers
_C.INFERENCE.TEST_NUM = 1 

# Test worker id
_C.INFERENCE.TEST_ID = 0 

# Batchsize for inference
_C.INFERENCE.SAMPLES_PER_BATCH = 32

#######################################################
# Util functions
#######################################################

def update_inference_cfg(cfg):
    r"""Update configurations (cfg) when running mode is inference.
    """
    cfg.MODEL.INPUT_SIZE = cfg.INFERENCE.INPUT_SIZE
    cfg.MODEL.OUTPUT_SIZE = cfg.INFERENCE.OUTPUT_SIZE
    cfg.DATASET.IMAGE_NAME = cfg.INFERENCE.IMAGE_NAME
    cfg.DATASET.OUTPUT_PATH = cfg.INFERENCE.OUTPUT_PATH
    cfg.DATASET.PAD_SIZE = cfg.INFERENCE.PAD_SIZE

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def save_all_cfg(cfg, output_dir):
    """Save configs in the output directory."""
    # Save config.yaml in the experiment directory after combine all 
    # non-default configurations from yaml file and command line.
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print("Full config saved to {}".format(path))
