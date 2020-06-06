Configurations
===============

Connectomics uses a key-value based configuration system that can be used to carry outstandard and commonly used tasks.

The config system of Connectomics uses YAML, a human-readable data-serialization language and yacs package, a simple experiment 
configuration system for research. In addition to the general tasks, that access and update a config, we provide the following extra 
functionalities:

#. The config have ``_C.key:value``  field, which will use a pre-defined config first. Values in the pre-defined config will 
   be overwritten in sub-configs, if there are any according to requirements. We provided several base configs for standard model 
   architectures as ``task.yaml`` files.

``config`` has a very limited abstraction. We do not expect all features in Connectomics to be available through configs. If you need 
something that’s not available in the config space, please modify the keys-value pairs in the file at 
``/connectomics/config/config.py``

Basic Usage
============

Some basic usage of the ``CfgNode`` object is shown here.

.. code-block:: none

    from yacs.config import CfgNode as CN
    _C = CN()            # config definition
    _C.SYSTEM = CN()     # config definition for GPU and CPU
    _C.MODEL = CN()      # Model architectures defined in the package
    _C.MODEL.ARCHITECTURE = 'unet_residual_3d' 
   
The configs in Connectomics also accepts command line configuration overwrite, i.e.: Key-value pairs provided in the command line will 
overwrite the existing values in the config file. For example, ``main.py`` can be used with to modify input file name :

.. code-block:: none

    python -u scripts/main.py \
    --config-file configs/Lucchi-Mitochondria.yaml DATASET.IMAGE_NAME ‘img/train_im.h5’
  
To see a list of available configs in Connectomics and what they mean, check `Config References <https://github.com/zudi-
lin/pytorch_connectomics/blob/master/connectomics/config/config.py>`_


Best Practice with Configs
==========================

#. Treat the configs you write as “code”: avoid copying them or duplicating them; use ``_BASE_`` to share common parts between 
configs.

#. Keep the configs you write simple: don’t include keys that do not affect the experimental setting.


Specifing multiple loss for a single learning target
=========================================================

We can also specify multiple loss functions as a criterion for training the segmentation models, i.e Binary cross-entropy loss, Dice 
loss by changing the key-value pairs of ``_LOSS_OPTION`` in the ``config.yaml`` file or by giving it explicitly as command-line 
arguments, multiple loss functions can be put to use for a single learning target.


#. Say, you want to use Weighted Binary cross-entropy loss as well as Dice loss as criterions for mitochondria segmentation task.

.. code-block:: none

   python -u scripts/main.py --config-file configs/Lucchi-Mitochondria.yaml \
   MODEL.LOSS_OPTION [[‘WeightedBCE’, ‘DiceLoss’]] MODEL.LOSS_WEIGHT [[1.0, 1.0]]
   
 
``LOSS_OPTION`` specifies the loss criterions to be used while training
``LOSS_WEIGHT`` specifies the weight or emphasis to be given to each loss criterion, i.e In the above case both the loss criterion
will contribute equally to the overall loss.


#. Say, you want to use Weighted Binary cross-entropy loss as well as Dice loss as criterions with a weight of 1.0, 0.5 .

.. code-block:: none

   python -u scripts/main.py --config-file configs/Lucchi-Mitochondria.yaml \
   MODEL.LOSS_OPTION [[‘WeightedBCE’, ‘DiceLoss’]] MODEL.LOSS_WEIGHT [[1.0, 0.5]]
   
 
``LOSS_OPTION`` specifies the loss criterions to be used while training
``LOSS_WEIGHT`` specifies the weight or emphasis to be given to each loss criterion, i.e. In the above case cross-entropy will 
contribute twice times more than dice loss to the overall loss.



#. Say, you want to use only Weighted Binary cross-entropy loss as a criterion for mitochondria segmentation task.

.. code-block:: none

   python -u scripts/main.py --config-file configs/Lucchi-Mitochondria.yaml \
   MODEL.LOSS_OPTION [[‘WeightedBCE’]] MODEL.LOSS_WEIGHT [[1.0]]
   
 
``LOSS_OPTION`` specifies only W Binary cross-entropy as loss criterions to be used while training.
``LOSS_WEIGHT`` specifies the weight emphasis to be given to each loss criterion, i.e. In the above case, only cross-entropy will 
contribute to the overall loss.


















