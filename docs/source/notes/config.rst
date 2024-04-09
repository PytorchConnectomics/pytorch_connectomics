Configurations
================

`PyTorch Connectomics <https://github.com/zudi-lin/pytorch_connectomics>`_ uses a key-value based configuration system 
that can be adjusted to carry out standard and commonly used tasks. The configuration system is built with `YACS <https://github.com/rbgirshick/yacs>`_
that uses YAML, a human-readable data-serialization language, to manage options.

.. note::
   The system has ``_C.KEY:VALUE``  field, which will use a pre-defined configurations first. Values in the pre-defined config will 
   be overwritten in sub-configs, if there are any according to the user requirements. We provided several base configs for standard tasks
   in connectomics research as ``<task>.yaml`` files at `pytorch_connectomics/configs/ <https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs>`_.

We do not expect all features in the package to be available through configs, which will make it too complicated. If you need 
to add options that are not available in the current version, please modify the keys-value pairs in ``/connectomics/config/config.py``

Basic Usage
-------------

Some basic usage of the ``CfgNode`` object in `YACS <https://github.com/rbgirshick/yacs>`_ is shown below:

.. code-block:: python

   from yacs.config import CfgNode as CN
   _C = CN()            # config definition
   _C.SYSTEM = CN()     # config definition for GPU and CPU
   _C.SYSTEM.NUM_GPUS = 4 

   _C.MODEL = CN()      # Model architectures defined in the package
   _C.MODEL.ARCHITECTURE = 'unet_residual_3d' 

   # specify the name and type of additional targets in the augmentor
   _C.AUGMENTOR = CN()
   _C.AUGMENTOR.ADDITIONAL_TARGETS_NAME = ['label']
   _C.AUGMENTOR.ADDITIONAL_TARGETS_TYPE = ['mask']
   
The configs in PyTorch Connectomics also accepts command line configuration overwriting, *i.e.*, key-value pairs provided in the command line will 
overwrite the existing values in the config file. For example, we can add arguments when executing ``scripts/main.py``:

.. code-block:: none

    python -u scripts/main.py \
    --config-file configs/Lucchi-Mitochondria.yaml SOLVER.ITERATION_TOTAL 30000
  
To see the list of all available configurations in the current PyTorch Connectomics package and their default values, please check the `config references <https://github.com/zudi-
lin/pytorch_connectomics/blob/master/connectomics/config/config.py>`_. All configuration options after command line overwriting will be saved to the experiment directory for future reference.


Multiple Losses for a Single Learning Target
----------------------------------------------

Sometimes training with a single loss function does not produce favorable predictions. Thus we provide a simple way to specify multiple loss functions
for training the segmentation models. For example, to use the weighted binary cross-entropy loss (``WeightedBCE``) and the soft Sørensen–Dice  
loss (``DiceLoss``) at the same time, we can change the key-value pairs of ``LOSS_OPTION`` in the ``config.yaml`` file by doing:

.. code-block:: yaml

   MODEL:
     LOSS_OPTION: [['WeightedBCE', 'DiceLoss']]
     LOSS_WEIGHT: [[1.0, 0.5]]
     WEIGHT_OPT: [['1', '0']]

``LOSS_OPTION``: the loss criterions to be used during training.
``LOSS_WEIGHT``: the relative weight of each loss function.
``WEIGHT_OPT``: the option for generating pixel-wise loss mask (set to '0' disable).

If you only want to use weighted binary cross-entropy loss, do:

.. code-block:: yaml

   MODEL:
     LOSS_OPTION: [['WeightedBCE']]
     LOSS_WEIGHT: [[1.0]]
     WEIGHT_OPT: [['1']]

Multitask Learning
--------------------

To conduct multitask learning, which predicts multiple targets given a image volume, we can further adjust the ``TARGET_OPT`` option.
For example, to conduct instance segmentation of mitochondria, we can predict not only the binary foreground mask but also the instance
boundary to distinguish closely touching objects. Specifically, we can use the following options:

.. code-block:: yaml

   MODEL:
     TARGET_OPT: ['0', '4-2-1']
     LOSS_OPTION: [['WeightedBCE', 'DiceLoss'], ['WeightedBCE']]
     LOSS_WEIGHT: [[1.0, 1.0], [1.0]]
     WEIGHT_OPT: [['1', '0'], ['1']]

``TARGET_OPT``: a list of the targets to learn.

Currently seven types of ``TARGET_OPT`` are supported:

- ``'0'``: binary foreground mask (used in the `mitochondria semantic segmentation tutorial <../tutorials/mito.html#semantic-segmentation>`_).

- ``'1'``: synaptic polarity mask (used in the `synaptic polairty tutorial <../tutorials/synapse.html#synaptic-polarity-detection>`_).

- ``'2'``: affinity map (used in the `neuron segmentation tutorial <../tutorials/neuron.html>`_).

- ``'3'``: masks of small objects only.

- ``'4'``: instance boundaries (used in the `mitochondria instance segmentation tutorial <../tutorials/mito.html#instance-segmentation>`_).

- ``'5'``: distance transform. This target represents each pixel as the (quantized) distance to the instance boundaries. By default the distance is calculated for each slice in a given volume. To calculate the distance transform for 3D objects, set the option to ``'5-3d'``.

- ``'9'``: generic segmantic segmentation. Supposing there are 12 classes (including one background class) to predict, we need to set ``MODEL.OUT_PLANES: 12`` and ``MODEL.TARGET_OPT: ['9-12']``. Here ``9`` represent the multi-class semantic segmentation task, while ``12`` in ``['9-12']`` represents the 12 semantic classes.

The list of learning targets here can be outdated. To check the latest version of supported learning targets, please see the 
``seg_to_targets`` function in this `file <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/data/utils/data_segmentation.py>`_.

Inference
-----------

Most of the config options are shared by training and inference. However, there are
several options to be adjusted at inference time by the ``update_inference_cfg`` function:

.. code-block:: python

   def update_inference_cfg(cfg: CfgNode):
      r"""Overwrite configurations (cfg) when running mode is inference. Please 
      note that None type is only supported in YACS>=0.1.8.
      """
      # dataset configurations
      if cfg.INFERENCE.INPUT_PATH is not None:
         cfg.DATASET.INPUT_PATH = cfg.INFERENCE.INPUT_PATH
      cfg.DATASET.IMAGE_NAME = cfg.INFERENCE.IMAGE_NAME
      cfg.DATASET.OUTPUT_PATH = cfg.INFERENCE.OUTPUT_PATH

      if cfg.INFERENCE.PAD_SIZE is not None:
         cfg.DATASET.PAD_SIZE = cfg.INFERENCE.PAD_SIZE
      if cfg.INFERENCE.IS_ABSOLUTE_PATH is not None:
         cfg.DATASET.IS_ABSOLUTE_PATH = cfg.INFERENCE.IS_ABSOLUTE_PATH

      if cfg.INFERENCE.DO_CHUNK_TITLE is not None:
         cfg.DATASET.DO_CHUNK_TITLE = cfg.INFERENCE.DO_CHUNK_TITLE

      # model configurations
      if cfg.INFERENCE.INPUT_SIZE is not None:
         cfg.MODEL.INPUT_SIZE = cfg.INFERENCE.INPUT_SIZE
      if cfg.INFERENCE.OUTPUT_SIZE is not None:
         cfg.MODEL.OUTPUT_SIZE = cfg.INFERENCE.OUTPUT_SIZE

      # output file name(s)
      if cfg.DATASET.DO_CHUNK_TITLE or cfg.DATASET.INFERENCE.DO_SINGLY:
         out_name = cfg.INFERENCE.OUTPUT_NAME
         name_lst = out_name.split(".")
         assert len(name_lst) <= 2, \
               "Invalid output file name is given."
         if len(name_lst) == 2:
               cfg.INFERENCE.OUTPUT_NAME = name_lst[0]

      for topt in cfg.MODEL.TARGET_OPT:
         # For multi-class semantic segmentation and quantized distance
         # transform, no activation function is applied at the output layer
         # during training. For inference where the output is assumed to be
         # in (0,1), we apply softmax.
         if topt[0] in ['5', '9'] and cfg.MODEL.OUTPUT_ACT == 'none':
               cfg.MODEL.OUTPUT_ACT = 'softmax'
               break

There are also several options exclusive for inference. For example:

.. code-block:: yaml

   INFERENCE:
     AUG_MODE: 'mean' # options for test augmentation
     AUG_NUM: 4
     BLENDING: 'gaussian' # blending function for overlapping inference
     STRIDE: (4, 128, 128) # sampling stride for inference
     SAMPLES_PER_BATCH: 4 # per GPU batchsize for inference 

Since at test time the model only runs forward pass, a larger mini-batch size is recommended for higher inference throughput. 

2D Models
-----------

Our package is mainly developed for volumetric data, but also supported 2D trainin and inference. There are a bunch of 
configuration options to update for 2D functionalities:

.. code-block:: yaml

   MODEL:
     ARCHITECTURE: unet_2d # specify a 2D architecture
     INPUT_SIZE: [1, 513, 513] # the z-dimension will be ignored
     OUTPUT_SIZE: [1, 513, 513]
   DATASET:
     DO_2D: True # stream 2D samples
     LOAD_2D: True # directly load 2D data if True, else load 3D and sample 2D patches
