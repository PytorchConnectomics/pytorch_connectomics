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

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.SEPARATE = False
_C.MODEL.SYNC_BN = True

_C.ENCODER = CN()
_C.ENCODER.BLOCK_TYPE = 'basicblock'
_C.ENCODER.LAYERS = [2, 2, 2, 2, 2]
_C.ENCODER.SYNC_BN = True
_C.ENCODER.ZERO_INIT_RESIDUAL = True
_C.ENCODER.ISOTROPIC = [False, False, False, True, True]
_C.ENCODER.IMAGE_CHANNELS = 1
# The number of channels of the feature maps at different
# ResNet stages.
_C.ENCODER.NUM_CHANNELS = [32, 64, 128, 256, 256]

_C.DECODER = CN()
_C.DECODER.PLANES = 32
_C.DECODER.SYNC_BN = True
_C.DECODER.COORD_CONV = False
_C.DECODER.USE_ASPP = False

# -----------------------------------------------------------------------------
# Loss Function
# -----------------------------------------------------------------------------

_C.LOSS = CN()
_C.LOSS.POS_WEIGHT = 2.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

_C.DATASET = CN()
# Dataset name and path is not specified in the defaults.
# Overwrite those variables in the custom config file.
_C.DATASET.NAME = ""
_C.DATASET.IMAGE_PATH = ""
_C.DATASET.LABEL_PATH = ""

# Image volumes can be isotropic or anisotropic due to
# different imaging techniques. This entry specify the 
# resolution in nanometer scale in Z,Y,X (int only).
_C.DATASET.RESOLUTION = (1.0, 1.0, 1.0)
_C.DATASET.ISOTROPIC = True

_C.DATASET.INPUT_SIZE = (32, 256, 256)
# Due to the center crop in the data augmentor, regions close to the volume
# border will never be sampled. Therefore we pad the input volume. For
# large-scale dataset there is no need for padding.
_C.DATASET.PADDING_SIZE = (8, 128, 128)
_C.DATASET.SAMPLE_STRIDE = (1, 1, 1)

_C.DATASET.LABEL_CONNECTED = False
_C.DATASET.DATA_AUGMENTATION = True

# Sample only the top k largest instances in the volume during training.
# Set to 0 to disable.
_C.DATASET.TOP_K_LARGEST = 0

# To prevent the seed from being to close to the boundary, 
# erode the mask first. However, when the object mask is very small,
# erosion may result in an empty target mask.
_C.DATASET.EROSION = False
_C.DATASET.EROSION_STRUCT = (3, 9, 9)

# Instead of taking the binary seed mask as input, we take the normalized 
# distance map following common practice in interactive segmentation.
_C.DATASET.DISTANCE_MAP = False
_C.DATASET.MAXIMUM_DISTANCE = 100.0

# -----------------------------------------------------------------------------
# Augmentor
# -----------------------------------------------------------------------------
_C.AUGMENTOR = CN()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN()
_C.DATA_LOADER.SHUFFLE = True
_C.DATA_LOADER.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 50000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Number of samples per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.SOLVER.SAMPLES_PER_BATCH = 16

# -----------------------------------------------------------------------------
# Misc options
# -----------------------------------------------------------------------------
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()