Configurations
===============

.. contents::
   :local:

`PyTorch Connectomics <https://github.com/zudi-lin/pytorch_connectomics>`_ uses a key-value based configuration system 
that can be adjusted to carry out standard and commonly used tasks. The configuration system is built with `YACS <https://github.com/rbgirshick/yacs>`_
that uses YAML, a human-readable data-serialization language, to manage options.

#. The config have ``_C.key:value``  field, which will use a pre-defined config first. Values in the pre-defined config will 
   be overwritten in sub-configs, if there are any according to requirements. We provided several base configs for standard tasks
   in connectomics research as ``task.yaml`` files at `pytorch_connectomics/configs/ <https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs>`_.

We do not expect all features in the package to be available through configs. If you need 
to add some options that are not available in the version, please modify the keys-value pairs in ``/connectomics/config/config.py``

Basic Usage
-------------

Some basic usage of the ``CfgNode`` object is shown here.

.. code-block:: none

    from yacs.config import CfgNode as CN
    _C = CN()            # config definition
    _C.SYSTEM = CN()     # config definition for GPU and CPU
    _C.MODEL = CN()      # Model architectures defined in the package
    _C.MODEL.ARCHITECTURE = 'unet_residual_3d' 
   
The configs in PyTorch Connectomics also accepts command line configuration overwrite, i.e.: Key-value pairs provided in the command line will 
overwrite the existing values in the config file. For example, ``main.py`` can be used with to modify input file name :

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
-----------------------

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

- ``'0'``: binary foreground mask (see more details in the `mitochondria segmentation tutorial <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/lucchi.html>`_).

- ``'1'``: synaptic polarity mask (see more details in the `synaptic polairty tutorial <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/synaptic_partner.html>`_).

- ``'2'``: affinity map (see more details in the `neuron segmentation tutorial <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_).

- ``'3'``: small object masks.

- ``'4'``: instance boundaries.
