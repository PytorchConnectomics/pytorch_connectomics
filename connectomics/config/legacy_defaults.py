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
# Run distributed training using DistributedDataparallel model
_C.SYSTEM.DISTRIBUTED = False
_C.SYSTEM.PARALLEL = 'DP'
_C.SYSTEM.DISTRIBUTED_BACKEND = 'nccl'
# Debug mode is for tackle cases where this is no errors
# but the model behavior are unexpected.
_C.SYSTEM.DEBUG = False

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# Model architectures defined in the package: unet_super, super, fpn, unet_residual_3d
_C.MODEL.ARCHITECTURE = 'unet_3d'
_C.MODEL.BLOCK_TYPE = 'residual'
_C.MODEL.BACKBONE = 'resnet'
_C.MODEL.DEPLOY_MODE = False

# Number of filters per unet block
_C.MODEL.FILTERS = [28, 36, 48, 64, 80]
_C.MODEL.BLOCKS = [2, 2, 2, 2]
_C.MODEL.KERNEL_SIZES = [3, 3, 5, 3, 3] #used only in effnet for now

_C.MODEL.ATTENTION = 'squeeze_excitation'

_C.MODEL.ISOTROPY = [False, False, False, True, True]

_C.MODEL.TARGET_OPT = ['0']
_C.MODEL.TARGET_OPT_MULTISEG_SPLIT = None

_C.MODEL.LABEL_EROSION = None # erode masks
_C.MODEL.LABEL_DILATION = None # dilate masks

_C.MODEL.WEIGHT_OPT = [['1']]

# Choose the right loss function for each target:
# 'WeightedMSE', 'WeightedBCE', 'JaccardLoss', 'DiceLoss'
_C.MODEL.LOSS_OPTION = [['WeightedBCE']]
# activation for the output in loss calculation
_C.MODEL.OUTPUT_ACT = [['none']]

# Weight for each loss function
_C.MODEL.LOSS_WEIGHT = [[1.0]]
_C.MODEL.LOSS_KWARGS_KEY = None
_C.MODEL.LOSS_KWARGS_VAL = None

# Define the number of input channels. Usually EM images are
# single-channel gray-scale image.
_C.MODEL.IN_PLANES = 1

# Define the number of output channels.
_C.MODEL.OUT_PLANES = 1

# Padding mode, possible options: 'zeros','circular', 'reflect', 'replicate'
_C.MODEL.PAD_MODE = 'replicate'

# Normalization mode, possible options: 'bn', 'sync_bn', 'in', 'gn', 'none'
_C.MODEL.NORM_MODE = 'bn'

# Activation mode, possible options: 'relu', 'elu', 'leaky'
_C.MODEL.ACT_MODE = 'elu'

# Use pooling layer for downsampling
_C.MODEL.POOLING_LAYER = False

# Mixed-precision training
_C.MODEL.MIXED_PRECESION = False

# If MODEL.EMBEDDING = 1 will do embedding
_C.MODEL.EMBEDDING = 1

# Last decoder head depth
_C.MODEL.HEAD_DEPTH = 1

_C.MODEL.INPUT_SIZE = [8, 256, 256]

_C.MODEL.OUTPUT_SIZE = [8, 256, 256]

_C.MODEL.REGU_OPT = None
_C.MODEL.REGU_TARGET = None
_C.MODEL.REGU_WEIGHT = None

# Fine-tune suffix for model saving
_C.MODEL.FINETUNE = ''

# Exact matching: the weights shape in pretrain model and current model are identical
_C.MODEL.EXACT = True

_C.MODEL.SIZE_MATCH = True

_C.MODEL.PRE_MODEL = ''

_C.MODEL.PRE_MODEL_LAYER = ''

_C.MODEL.PRE_MODEL_ITER = 0

# Return specified feature maps (only works with 3D U-Net and child classes)
_C.MODEL.RETURN_FEATS = None
# Predict an auxiliary output (only works with 2D DeeplabV3)
_C.MODEL.AUX_OUT = False


# Configurations for Swin UNETR
# Dimension of network feature size.
_C.MODEL.SWIN_UNETR_FEATURE_SIZE = 48

# Number of layers in each stage.
_C.MODEL.DEPTHS = (2, 2, 2, 2)

# Number of attention heads.
_C.MODEL.SWIN_UNETR_NUM_HEADS = (3, 6, 12, 24)

# Feature normalization type and arguments.
_C.MODEL.NORM_NAME = 'instance'

# Dropout rate.
_C.MODEL.SWIN_UNETR_DROPOUT_RATE = 0.0

# Attention dropout rate.
_C.MODEL.ATTN_DROP_RATE = 0.0

# Dropout path rate.
_C.MODEL.DROPOUT_PATH_RATE = 0.0

# Normalize output intermediate features in each stage.
_C.MODEL.NORMALIZE = True

# Use gradient checkpointing for reduced memory usage.
_C.MODEL.USE_CHECKPOINT = False

# Number of spatial dims.
_C.MODEL.SPATIAL_DIMS = 3

# Module used for downsampling, available options are `"mergingv2"`, `"merging"` and a user-specified `nn.Module`
_C.MODEL.DOWNSAMPLE = 'merging'

# Use swinunetr_v2, which adds a residual convolution block at the beggining of each swin stage.
_C.MODEL.USE_V2 = False


# Configurations for UNETR
# Dimension of network feature size.
_C.MODEL.UNETR_FEATURE_SIZE = 16

# Dimension of hidden layer.
_C.MODEL.HIDDEN_SIZE = 768

# Dimension of feedforward layer.
_C.MODEL.MLP_DIM = 3072

# Number of attention heads.
_C.MODEL.UNETR_NUM_HEADS = 12

# Position embedding layer type.
_C.MODEL.POS_EMBED = 'perceptron'

# Feature normalization type and arguments.
_C.MODEL.NORM_NAME = 'instance'

# bool argument to determine if convolutional block is used.
_C.MODEL.CONV_BLOCK = True

# bool argument to determine if residual block is used.
_C.MODEL.RES_BLOCK = True

# action of the input units to drop.
_C.MODEL.UNETR_DROPOUT_RATE = 0.0


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

# Scale ratio of the input data for different resolutions.
# Using a DATA_SCALE of [1., 0.5, 0.5] will downsample the
# original image by two times (e.g., 4nm -> 8nm).
_C.DATASET.DATA_SCALE = [1., 1., 1.]
_C.DATASET.IMAGE_SCALE = None
_C.DATASET.LABEL_SCALE = None
_C.DATASET.VALID_MASK_SCALE = None

# Scaling factor for super resolution
_C.DATASET.SCALE_FACTOR = [2, 3, 3]

# Specify the data path in the *.yaml files for different experiments.
_C.DATASET.IMAGE_NAME = None
_C.DATASET.LABEL_NAME = None
_C.DATASET.VALID_MASK_NAME = None

_C.DATASET.VAL_IMAGE_NAME = None
_C.DATASET.VAL_LABEL_NAME = None
_C.DATASET.VAL_VALID_MASK_NAME = None
_C.DATASET.VAL_PAD_SIZE = [0, 0, 0]

_C.DATASET.LABEL_VAST = False
_C.DATASET.INPUT_PATH = 'path/to/input'
_C.DATASET.OUTPUT_PATH = 'path/to/output'
_C.DATASET.IS_ABSOLUTE_PATH = False

# Specify whether the data is isotropic or not.
_C.DATASET.IS_ISOTROPIC = False

# 2d or 3d dataset
_C.DATASET.DO_2D = False
_C.DATASET.LOAD_2D = False

# Specify whether to drop channels in multi-channel images/volumes
_C.DATASET.DROP_CHANNEL = False

# Reduce the the mask indicies in a sampled label volume
_C.DATASET.REDUCE_LABEL = True

# Padding size for the input volumes
# Due to the center crop in the data augmentor, regions close to the volume
# border will never be sampled. Therefore we pad the input volume. For
# large-scale dataset there is no need for padding.
_C.DATASET.PAD_SIZE = [2, 64, 64]
_C.DATASET.PAD_MODE = 'reflect'  # reflect, constant, symmetric

# Upsample the input to at least the required sample size. If data 
# augmentor is used, the min_size is augmentor.sample_size, else is
# MODEL.INPUT_SIZE.
_C.DATASET.ENSURE_MIN_SIZE = False

# Normalize the image and cast to uint8 format
_C.DATASET.NORMALIZE_RANGE = True

# If it's a binary label
_C.DATASET.LABEL_BINARY = False

_C.DATASET.LABEL_MAG = 0

# Data in tile format or not.
_C.DATASET.DO_CHUNK_TITLE = 0

# Chunk parameters for tile format: chunk_num (z,y,x), chunk_stride
_C.DATASET.DATA_CHUNK_NUM = [1, 1, 1]

# Predefined data chunk to iterate through
_C.DATASET.DATA_CHUNK_IND = None
_C.DATASET.CHUNK_IND_SPLIT = None

# For TileDataset, specify the coordintate range of data to use.
# If not None, should be a list of format [z0, z1, y0, y1, x0, x1] applied to all volumes,
# or List[List[int]] with a range for each input TileDataset volume.
_C.DATASET.DATA_COORD_RANGE = None

# Boolean variable, euqal to 'int(args.data_chunk_num[-1:])==1'
_C.DATASET.DATA_CHUNK_STRIDE = True

# Chunk parameters for tile format: chunk_iter_num
_C.DATASET.DATA_CHUNK_ITER = 1000

# Handle dataset with partial annotation.
_C.DATASET.VALID_RATIO = 0.5

# For some datasets the foreground mask is sparse in the volume. Therefore
# we perform reject sampling to decrease (all completely avoid) regions
# without foreground masks. Set REJECT_SAMPLING.SIZE_THRES = -1 to disable.
# Note that reject sampling only works when label is given.
_C.DATASET.REJECT_SAMPLING = CN()
_C.DATASET.REJECT_SAMPLING.SIZE_THRES = -1
_C.DATASET.REJECT_SAMPLING.DIVERSITY = -1
_C.DATASET.REJECT_SAMPLING.P = 0.95
_C.DATASET.REJECT_SAMPLING.NUM_TRIAL = 50

# Normalize model inputs (the images are assumed to be gray-scale).
_C.DATASET.MEAN = 0.5
_C.DATASET.STD = 0.5
_C.DATASET.MATCH_ACT = 'none'

_C.DATASET.DISTRIBUTED = False

# -----------------------------------------------------------------------------
# Augmentor
# -----------------------------------------------------------------------------
_C.AUGMENTOR = CN({"ENABLED": True})

# The nearest interpolation for the label mask during data augmentation
# can result in masks with coarse boundaries. Thus we apply Gaussian filtering
# to smooth the object boundary (default: False).
# WARNING: applying label smoothing can erase the segmentation masks of thin 
# structures like spine necks and wrinkle artifacts.
_C.AUGMENTOR.SMOOTH = False

# CfgNodes can only contain a limited set of valid types:
# _VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
_C.AUGMENTOR.ADDITIONAL_TARGETS_NAME = ['label']
_C.AUGMENTOR.ADDITIONAL_TARGETS_TYPE = ['mask']

# _C.AUGMENTOR.[xxx].SKIP specify the sample
# key to skip for that augmentation
_C.AUGMENTOR.ROTATE = CN({"ENABLED": True})
_C.AUGMENTOR.ROTATE.ROT90 = True
_C.AUGMENTOR.ROTATE.P = 1.0
_C.AUGMENTOR.ROTATE.SKIP = []

_C.AUGMENTOR.RESCALE = CN({"ENABLED": True})
_C.AUGMENTOR.RESCALE.FIX_ASPECT = False
_C.AUGMENTOR.RESCALE.P = 0.5
_C.AUGMENTOR.RESCALE.SKIP = []

_C.AUGMENTOR.FLIP = CN({"ENABLED": True})
_C.AUGMENTOR.FLIP.P = 1.0
# Conducting x-z and y-z flip only when the dataset is isotropic
# and the input is cubic.
_C.AUGMENTOR.FLIP.DO_ZTRANS = 0
_C.AUGMENTOR.FLIP.SKIP = []

_C.AUGMENTOR.ELASTIC = CN({"ENABLED": True})
_C.AUGMENTOR.ELASTIC.P = 0.75
# Maximum pixel-moving distance of elastic transformation
_C.AUGMENTOR.ELASTIC.ALPHA = 16.0
# Standard deviation of the Gaussian filter
_C.AUGMENTOR.ELASTIC.SIGMA = 4.0
_C.AUGMENTOR.ELASTIC.SKIP = []

_C.AUGMENTOR.GRAYSCALE = CN({"ENABLED": True})
_C.AUGMENTOR.GRAYSCALE.P = 0.75
_C.AUGMENTOR.GRAYSCALE.SKIP = []

# Randomly mask out some input regions
_C.AUGMENTOR.MISSINGPARTS = CN({"ENABLED": True})
_C.AUGMENTOR.MISSINGPARTS.P = 0.9
_C.AUGMENTOR.MISSINGPARTS.ITER = 64
_C.AUGMENTOR.MISSINGPARTS.SKIP = []

_C.AUGMENTOR.MISSINGSECTION = CN({"ENABLED": True})
_C.AUGMENTOR.MISSINGSECTION.P = 0.5
_C.AUGMENTOR.MISSINGSECTION.NUM_SECTION = 2
_C.AUGMENTOR.MISSINGSECTION.SKIP = []

_C.AUGMENTOR.MISALIGNMENT = CN({"ENABLED": True})
_C.AUGMENTOR.MISALIGNMENT.P = 0.5
# Maximum pixel displacement in each direction (x and y) (int)
_C.AUGMENTOR.MISALIGNMENT.DISPLACEMENT = 16
# The ratio of mis-alignment by rotation among all mis-alignment augmentations.
_C.AUGMENTOR.MISALIGNMENT.ROTATE_RATIO = 0.5
_C.AUGMENTOR.MISALIGNMENT.SKIP = []

_C.AUGMENTOR.MOTIONBLUR = CN({"ENABLED": True})
_C.AUGMENTOR.MOTIONBLUR.P = 0.5
# Number of sections along z dimension to apply motion blur
_C.AUGMENTOR.MOTIONBLUR.SECTIONS = 2
# Kernel size of motion blur
_C.AUGMENTOR.MOTIONBLUR.KERNEL_SIZE = 11
_C.AUGMENTOR.MOTIONBLUR.SKIP = []

_C.AUGMENTOR.CUTBLUR = CN({"ENABLED": True})
_C.AUGMENTOR.CUTBLUR.P = 0.5
_C.AUGMENTOR.CUTBLUR.LENGTH_RATIO = 0.4
_C.AUGMENTOR.CUTBLUR.DOWN_RATIO_MIN = 2.0
_C.AUGMENTOR.CUTBLUR.DOWN_RATIO_MAX = 8.0
_C.AUGMENTOR.CUTBLUR.DOWNSAMPLE_Z = False
_C.AUGMENTOR.CUTBLUR.SKIP = []

_C.AUGMENTOR.CUTNOISE = CN({"ENABLED": True})
_C.AUGMENTOR.CUTNOISE.P = 0.75
_C.AUGMENTOR.CUTNOISE.LENGTH_RATIO = 0.4
_C.AUGMENTOR.CUTNOISE.SCALE = 0.3
_C.AUGMENTOR.CUTNOISE.SKIP = []

_C.AUGMENTOR.COPYPASTE = CN({"ENABLED": False})
_C.AUGMENTOR.COPYPASTE.AUG_THRES = 0.7
_C.AUGMENTOR.COPYPASTE.P = 0.8
_C.AUGMENTOR.COPYPASTE.SKIP = []

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

# Specify the name of the optimizer
_C.SOLVER.NAME = "SGD"  # "SGD", "Adam", "AdamW"

# Specify the learning rate scheduler.
_C.SOLVER.LR_SCHEDULER_NAME = "MultiStepLR"

_C.SOLVER.ITERATION_STEP = 1

# Save a checkpoint after every this number of iterations.
_C.SOLVER.ITERATION_SAVE = 5000
_C.SOLVER.ITERATION_TOTAL = 40000
_C.SOLVER.ITERATION_VAL = 5000

# Whether or not to restart training from iteration 0 regardless
# of the 'iteration' key in the checkpoint file. This option only
# works when a pretrained checkpoint is loaded (default: False).
_C.SOLVER.ITERATION_RESTART = False

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.BETAS = (0.9, 0.999)  # Adam and AdamW

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

# Number of samples per GPU. If we have 8 GPUs and SAMPLES_PER_BATCH = 2,
# then each GPU will see 2 samples and the effective batch size is 16.
_C.SOLVER.SAMPLES_PER_BATCH = 2

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

# Stochastic Weight Averaging
_C.SOLVER.SWA = CN({"ENABLED": False})
_C.SOLVER.SWA.LR_FACTOR = 0.05
_C.SOLVER.SWA.START_ITER = 90000
_C.SOLVER.SWA.MERGE_ITER = 10
_C.SOLVER.SWA.BN_UPDATE_ITER = 2000

# -----------------------------------------------------------------------------
# Monitor
# -----------------------------------------------------------------------------
_C.MONITOR = CN()

_C.MONITOR.LOG_OPT = [1, 1, 0]

_C.MONITOR.VIS_OPT = [0, 16]

_C.MONITOR.ITERATION_NUM = [20, 200]

# # -----------------------------------------------------------------------------
# # Inference
# # -----------------------------------------------------------------------------
_C.INFERENCE = CN()

_C.INFERENCE.INPUT_SIZE = None
_C.INFERENCE.OUTPUT_SIZE = None

_C.INFERENCE.TENSORSTORE_PATH = None
_C.INFERENCE.INPUT_PATH = None
_C.INFERENCE.IMAGE_NAME = None
_C.INFERENCE.OUTPUT_PATH = ""
_C.INFERENCE.OUTPUT_NAME = 'result.h5'
_C.INFERENCE.IS_ABSOLUTE_PATH = None
_C.INFERENCE.DO_CHUNK_TITLE = None

# Do inference one-by-one (load a volume when needed).
_C.INFERENCE.DO_SINGLY = False
_C.INFERENCE.DO_SINGLY_START_INDEX = 0
_C.INFERENCE.DO_SINGLY_STEP = 1

_C.INFERENCE.PAD_SIZE = None
_C.INFERENCE.UNPAD = True
# activation for the output for inference and visualization
_C.INFERENCE.OUTPUT_ACT = ['sigmoid']

_C.INFERENCE.STRIDE = [4, 128, 128]

# Blending function for overlapping inference.
_C.INFERENCE.BLENDING = 'gaussian'

_C.INFERENCE.AUG_MODE = 'mean'
_C.INFERENCE.AUG_NUM = None

_C.INFERENCE.DATA_SCALE = None # Overwrites DATASET.DATA_SCALE if is not None
_C.INFERENCE.OUTPUT_SCALE = [1., 1., 1.] # Rescale after model prediction

# Run the model forward pass with model.eval() if DO_EVAL is True, else
# run with model.train(). Layers like batchnorm and dropout will be affected.
_C.INFERENCE.DO_EVAL = True

# Number of test workers
_C.INFERENCE.TEST_NUM = 1

# Test worker id
_C.INFERENCE.TEST_ID = 0

# Number of samples per GPU (for inference). If we have 8 GPUs and
# SAMPLES_PER_BATCH = 4, then each GPU will see 2 samples and the
# effective batch size is 32.
_C.INFERENCE.SAMPLES_PER_BATCH = 4

# Change MODEL.RETURN_FEATS at inference time.
_C.INFERENCE.MODEL_RETURN_FEATS = None


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
