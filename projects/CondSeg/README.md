# Conditional Segmentation

## Overview

This repository contains the implementation of a PyTorch-Connectomics's based conditional segmentation 3D UNet. It provides a comprehensive setup for conditional segmentation tasks, with a focus on flexibility and extensibility in model configuration and data handling. It is suitable for researchers and practitioners in fields requiring precise segmentation capabilities.

## Structure

The repository is structured into several key directories and files, each serving a distinct purpose:

### Directories and Key Files

- `condseg/`: Core directory containing the main codebase.
  - `build.py`: Handles the setup and management of data loading and preprocessing for the deep learning model.
  - `collate.py`: Contains custom collate functions for PyTorch data loaders, crucial for batching strategies in training and inference.
  - `config.py`: Utilizes `YACS` for hierarchical configuration management, specifically designed for setting up conditional segmentation parameters.
  - `trainer.py`: Defines the `TrainerCondSeg` class for training and testing the model, integrating various components like data loading, training loops, and model evaluation.

- `configs/`: Contains YAML configuration files.
  - `JWR15/`: A subdirectory for specific configurations.
    - `JWR15-Synapse-Base.yaml`: Provides a base configuration for the model, including system settings, model architecture, dataset paths, and solver configurations.
    - `JWR15-Synapse-BCE-DICE.yaml`: Specifies loss function parameters (Binary Cross Entropy and Dice Loss) and output path for a model variant.

- `main.py`: The entry point of the project, orchestrating the training and inference processes.

## How it Works

1. **Data Preparation**: `build.py` in the `condseg` directory is responsible for loading, processing, and preparing data. It uses PyTorch's DataLoader to efficiently handle data in batches suitable for training and testing.

2. **Configuration Management**: The `config.py` file and YAML files in the `configs` directory allow for flexible and extensive configuration of model parameters, training settings, and dataset specifics. 

3. **Model Training and Inference**: The `trainer.py` file provides the `TrainerCondSeg` class, encapsulating the logic for setting up the model, training the model, conducting validation, and performing inference. It integrates the model with the data loaders, handles the training loop, loss computation, and implements various functionalities like checkpoint saving and learning rate scheduling.

4. **Execution**: `main.py` serves as the starting point for running the model. It sets up the device configuration, initializes the trainer, and triggers either the training or inference process based on the provided command-line arguments.

## Usage

To use this repository:

1. **Setup**: Setup up PyTorch-Connetomics as specified in the main README.

2. **Data Preparation**: Organize your volumetric data and corresponding labels in the required format and structure.

3. **Configuration**: Adjust the parameters in the provided YAML configuration files in the `configs` directory to suit your specific requirements.

4. **Run**: Execute `main.py` with the desired mode (training or inference) and other necessary command-line arguments. 

5. **Results**: Output, such as trained models and inference results, will be saved in the specified output directories.

## Data

### Type of Data Required

1. **Volumetric Data**: The primary type of data required is volumetric data, which are essentially 3D images or volumes. These could be medical scans like MRI or CT images, 3D microscopy images, or any other form of 3D data that requires segmentation.

2. **Labels/Annotations**: For supervised learning, you'll need corresponding labels or annotations for your volumetric data. These labels should indicate the segments or regions of interest in the volumes. In conditional segmentation, these labels might also include specific conditions or attributes for each segment.

### Data Format

- The data (both images and labels) should be in the `.h5`  (HDF5) or the `.tif` (TIFF) file format to be compatible with the tools used in the repository.
- While the system handles certain normalization and preprocessing steps, users should still perform a thorough inspection and preprocessing of their data to ensure compatibility and optimal processing within the system.

### Data Structure

- The system can handle both single complete volumes and subvolumes (chunks). This is configured by the `DATA_CHUNK_NUM` parameter in the YAML configuration files.
- For inference, there’s a distinction between processing volumes "singly" or in "chunks," as indicated by the `INFERENCE.DO_SINGLY` configuration parameter. The choice between these options would depend on the size of your data and available computational resources.

## Configuration Files

### Base Configuration

The `JWR15-Synapse-Base.yaml` file definiens the operational parameters of your model and its training/inference process. It covers a broad range of settings, from the architectural details of the model to how the data is handled and how the training process is managed.

1. **`SYSTEM`**:
   - `NUM_GPUS` and `NUM_CPUS`: Specifies the number of GPUs and CPUs to be used.

2. **`MODEL`**:
   - Various parameters define the model architecture and properties, such as `ARCHITECTURE` (set to `unet_plus_3d`), `BLOCK_TYPE`, `INPUT_SIZE`, `OUTPUT_SIZE`, `IN_PLANES`, `OUT_PLANES`, etc.
   - `FILTERS`: Defines the number of filters in each layer of the network.
   - `NORM_MODE`: Specifies the normalization mode, set to `sync_bn` (synchronized batch normalization).

3. **`CONDITIONAL`**:
   - `LABEL_TYPE`: The type of labels used, set to `syn` for synapse.
   - `INFERENCE_CONDITIONAL`: Specifies the conditional file used during inference.

4. **`DATASET`**:
   - Contains paths and names for image and label data, input path, output path, padding size, and other dataset-specific configurations.
   - `DISTRIBUTED`: Indicates if the dataset should be distributed (useful for multi-GPU training).
   - `REJECT_SAMPLING`: Parameters for handling reject sampling.

5. **`SOLVER`**:
   - Specifies the learning rate scheduler (`LR_SCHEDULER_NAME`), base learning rate (`BASE_LR`), and other solver-related settings like iteration steps, save intervals, total iterations, and samples per batch.

6. **`INFERENCE`**:
   - Settings for the inference process, including input and output sizes, output activation function, paths, and augmentation mode.

### Loss Configuration

The `JWR15-Synapse-BCE-DICE.yaml` configuration file defines specific aspects of the model's loss functions and output management, which are key to the model's learning and performance.

1. **`MODEL`**:
   - `LOSS_OPTION`: Specifies a list of loss functions to be used. In this case, it's a combination of `WeightedBCEWithLogitsLoss` and `DiceLoss`. These are common loss functions for segmentation tasks, where BCE (Binary Cross Entropy) handles binary classification at each pixel and Dice Loss is used for overlap measurement between the predicted segmentation and the ground truth.
   - `LOSS_WEIGHT`: Assigns weights to the specified loss functions. This parameter is important for balancing the influence of different loss functions during training. Here, `1.0` is assigned to `WeightedBCEWithLogitsLoss` and `0.5` to `DiceLoss`.
   - `WEIGHT_OPT`: Provides additional options for weighting, potentially for the specified losses.
   - `OUTPUT_ACT`: Specifies the activation functions to be applied to the output of the model. The configuration indicates no activation (`"none"`) for the first output and a sigmoid activation (`"sigmoid"`) for the second output. This is likely tailored to the specific needs of the segmentation task.

2. **`DATASET`**:
   - `OUTPUT_PATH`: Defines the path where the output of this particular model configuration (using BCE and Dice losses) will be saved. This allows for organized storage of results, especially when experimenting with different model configurations.
