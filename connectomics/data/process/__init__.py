# Core processing functions
from .bbox_processor import *  # New: unified bbox processing framework

# MONAI-native transforms and composition

# Pipeline builder (primary entry point for label transforms)
from .build import create_label_transform_pipeline
