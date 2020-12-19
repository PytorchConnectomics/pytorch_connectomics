Configurations
===============

.. contents::
   :local:

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
    _C.MODEL = CN()      # Model architectures defined in the package
    _C.MODEL.ARCHITECTURE = 'unet_residual_3d' 
   
The configs in PyTorch Connectomics also accepts command line configuration overwrite, i.e.: Key-value pairs provided in the command line will 
overwrite the existing values in the config file. For example, we can add arguments when executing ``scripts/main.py``:

.. code-block:: none

    python -u scripts/main.py \
    --config-file configs/Lucchi-Mitochondria.yaml SOLVER.ITERATION_TOTAL 30000
  
To see a list of available configs in PyTorch Connectomics and what they mean, check `Config References <https://github.com/zudi-
lin/pytorch_connectomics/blob/master/connectomics/config/config.py>`_.


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

Currently five kinds of ``TARGET_OPT`` are supported:

- ``'0'``: binary foreground mask (used in the `mitochondria segmentation tutorial <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/lucchi.html>`_).

- ``'1'``: synaptic polarity mask (used in the `synaptic polairty tutorial <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/synaptic_partner.html>`_).

- ``'2'``: affinity map (used in the `neuron segmentation tutorial <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_).

- ``'3'``: masks of small objects only.

- ``'4'``: instance boundaries (used in the `mitochondria segmentation tutorial <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/lucchi.html>`_).

- ``'9'``: generic segmantic segmentation. Supposing there are 12 classes (including one background class) to predict, we need to set ``MODEL.OUT_PLANES: 12`` and ``MODEL.TARGET_OPT: ['9-12']``. Here ``9`` represent the multi-class semantic segmentation task, while ``12`` in ``['9-12']`` represents the 12 semantic classes.

More options will be provided soon!

Inference
-----------

Most of the config options are shared by training and inference. However, there are
several options to be adjusted at inference time by the ``update_inference_cfg`` function:

.. code-block:: python

   def update_inference_cfg(cfg):
      r"""Update configurations (cfg) when running mode is inference.

      Note that None type is not supported in current release of YACS (0.1.7), but will be 
      supported soon according to this pull request: https://github.com/rbgirshick/yacs/pull/18.
      Therefore a re-organization of the configurations using None type will be done when YACS
      0.1.8 is released.
      """
      # Dataset configurations:
      if len(cfg.INFERENCE.INPUT_PATH) != 0:
         cfg.DATASET.INPUT_PATH = cfg.INFERENCE.INPUT_PATH
      cfg.DATASET.IMAGE_NAME = cfg.INFERENCE.IMAGE_NAME
      cfg.DATASET.OUTPUT_PATH = cfg.INFERENCE.OUTPUT_PATH
      if len(cfg.INFERENCE.PAD_SIZE) != 0:
         cfg.DATASET.PAD_SIZE = cfg.INFERENCE.PAD_SIZE

      # Model configurations:
      if len(cfg.INFERENCE.INPUT_SIZE) != 0:
         cfg.MODEL.INPUT_SIZE = cfg.INFERENCE.INPUT_SIZE
      if len(cfg.INFERENCE.OUTPUT_SIZE) != 0:
         cfg.MODEL.OUTPUT_SIZE = cfg.INFERENCE.OUTPUT_SIZE

There are also several options exclusive for inference. For example:

.. code-block:: yaml

   INFERENCE:
     AUG_MODE: 'mean' # options for test augmentation
     AUG_NUM: 4
     BLENDING: 'gaussian' # blending function for overlapping inference
     STRIDE: (4, 128, 128) # sampling stride for inference
     SAMPLES_PER_BATCH: 16 # batchsize for inference