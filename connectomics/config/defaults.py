"""
Modern configuration defaults for PyTorch Connectomics.

This configuration only includes settings needed for:
- nnUNet model architectures
- Lightning training framework
- MONAI transforms
"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_CPUS = 4
_C.SYSTEM.PARALLEL = 'DP'  # Data Parallel (DP) or Distributed Data Parallel (DDP)

# -----------------------------------------------------------------------------
# Model (nnUNet architectures only)
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# Modern nnUNet architectures
_C.MODEL.ARCHITECTURE = 'monai_basic_unet3d'  # Default to MONAI Basic UNet (most likely to be available)

# Model I/O dimensions
_C.MODEL.INPUT_SIZE = [128, 128, 128]
_C.MODEL.OUTPUT_SIZE = [128, 128, 128]
_C.MODEL.IN_PLANES = 1
_C.MODEL.OUT_PLANES = 1

# Loss configuration
_C.MODEL.LOSS_OPTION = [["DiceLoss", "WeightedBCEWithLogitsLoss"]]
_C.MODEL.OUTPUT_ACT = [["sigmoid", "none"]]
_C.MODEL.LOSS_WEIGHT = [[1.0, 1.0]]
_C.MODEL.TARGET_OPT = ["0"]  # 0: semantic segmentation
_C.MODEL.WEIGHT_OPT = [["1", "0"]]

# Model-specific parameters (optional, will use defaults if not specified)
_C.MODEL.FILTERS = (32, 64, 128, 256, 512, 1024)  # For MONAI models (6 levels required)
_C.MODEL.UNETR_FEATURE_SIZE = 16  # For UNETR models
_C.MODEL.HIDDEN_SIZE = 768  # For transformer models
_C.MODEL.MLP_DIM = 3072  # For transformer models
_C.MODEL.UNETR_NUM_HEADS = 12  # For UNETR models
_C.MODEL.UNETR_DROPOUT_RATE = 0.0  # For UNETR models

# -----------------------------------------------------------------------------
# Model compatibility settings (for make_parallel function)
_C.MODEL.NORM_MODE = "batch"  # Normalization mode

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

# Data paths
_C.DATASET.INPUT_PATH = "datasets/"
_C.DATASET.OUTPUT_PATH = "outputs/"
_C.DATASET.IMAGE_NAME = "train_image.h5"
_C.DATASET.LABEL_NAME = "train_label.h5"

# Data preprocessing
_C.DATASET.PAD_SIZE = [8, 32, 32]
_C.DATASET.DO_2D = False

# Validation data (optional)
_C.DATASET.VAL_IMAGE_NAME = None
_C.DATASET.VAL_LABEL_NAME = None

# -----------------------------------------------------------------------------
# Solver (Training)
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

# Optimizer
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.SAMPLES_PER_BATCH = 2

# Training schedule
_C.SOLVER.ITERATION_TOTAL = 50000
_C.SOLVER.ITERATION_STEP = 1
_C.SOLVER.ITERATION_SAVE = 2000
_C.SOLVER.VAL_CHECK_INTERVAL = 2000

# Modern training features
_C.SOLVER.GRAD_CLIP = 1.0
_C.SOLVER.GRAD_ACCUMULATE = 1

# -----------------------------------------------------------------------------
# Lightning
# -----------------------------------------------------------------------------
_C.LIGHTNING = CN()

# Precision and performance
_C.LIGHTNING.PRECISION = "16-mixed"  # Mixed precision training
_C.LIGHTNING.STRATEGY = "auto"  # Auto-select strategy
_C.LIGHTNING.ACCELERATOR = "auto"  # Auto-detect GPU/CPU

# Logging and monitoring
_C.LIGHTNING.LOGGER_TYPE = "tensorboard"  # tensorboard or wandb
_C.LIGHTNING.LOG_EVERY_N_STEPS = 50
_C.LIGHTNING.ENABLE_PROGRESS_BAR = True

# Checkpointing
_C.LIGHTNING.SAVE_TOP_K = 3
_C.LIGHTNING.MONITOR = "val_loss"
_C.LIGHTNING.MODE = "min"

# Early stopping
_C.LIGHTNING.EARLY_STOPPING = False
_C.LIGHTNING.EARLY_STOPPING_PATIENCE = 10

# Profiling and debugging
_C.LIGHTNING.PROFILER = None  # "simple", "advanced", or None
_C.LIGHTNING.DETECT_ANOMALY = False

# -----------------------------------------------------------------------------
# MONAI Transforms
# -----------------------------------------------------------------------------
_C.MONAI = CN()

# Transform settings
_C.MONAI.USE_TRANSFORMS = True
_C.MONAI.SPATIAL_SIZE = [128, 128, 128]

# Augmentation settings
_C.MONAI.RAND_FLIP_PROB = 0.5
_C.MONAI.RAND_ROTATE_PROB = 0.5
_C.MONAI.RAND_SCALE_PROB = 0.5
_C.MONAI.RAND_ELASTIC_PROB = 0.5
_C.MONAI.RAND_INTENSITY_PROB = 0.5

# Normalization
_C.MONAI.NORMALIZE_INTENSITY = True
_C.MONAI.INTENSITY_RANGE = (0.0, 1.0)

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()

# Inference data
_C.INFERENCE.INPUT_PATH = None
_C.INFERENCE.IMAGE_NAME = "test_image.h5"
_C.INFERENCE.OUTPUT_PATH = "outputs/inference/"
_C.INFERENCE.OUTPUT_NAME = "prediction.h5"

# Inference parameters
_C.INFERENCE.INPUT_SIZE = [128, 128, 128]
_C.INFERENCE.OUTPUT_SIZE = [128, 128, 128]
_C.INFERENCE.STRIDE = [64, 64, 64]
_C.INFERENCE.SAMPLES_PER_BATCH = 4
_C.INFERENCE.OUTPUT_ACT = ["sigmoid"]

# Test-time augmentation
_C.INFERENCE.AUG_MODE = "mean"
_C.INFERENCE.AUG_NUM = 4

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the modern config."""
    return _C.clone()