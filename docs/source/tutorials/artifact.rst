Artifacts Detection (Draft)
============================

Wrinkle Detection
--------------------

For pre-processing, we create a folder ``datasets/Wrinkle/train/``, and put (or soft link) the training images and wrinkle labels under
the folder, which perserves a directory structure like ``images/**/*.png`` and ``wrinkles/**/*.png``. The model will randomly sample patches
from the large images at training time and run sliding-window inference at test time, with the image resolution unchanged.

The config file for training wrinkle detection model from EM images is ``configs/misc/Wrinkle-Deeplab-Binary-2D.yaml``. We have tested
the training with 2 Nvidia V100 GPUs and 16 CPU cores. After intalling the package, run

.. code-block:: none

    source activate py3_torch
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.run \
    --nproc_per_node=2 --master_port=1234 scripts/main.py --distributed \
    --config-file configs/misc/Wrinkle-Deeplab-Binary-2D.yaml

The model checkpoints will be saved to ``outputs/Wrinkle_Deeplab/``. For inference on test images, run

.. code-block:: none

    source activate py3_torch
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py --inference \
    --config-file configs/misc/Wrinkle-Deeplab-Binary-2D.yaml \
    --checkpoint outputs/Wrinkle_Deeplab_new/checkpoint_100000.pth.tar
