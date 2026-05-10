Lucchi++ (Semantic Segmentation)
==================================

This tutorial provides step-by-step guidance for mitochondria segmentation
with the EM benchmark dataset released by
`Lucchi et al. (2012) <https://cvlab.epfl.ch/research/page-90578-en-html/research-medical-em-mitochondria-index-php/>`__.
We approach the task as a **semantic segmentation** task and predict the
mitochondria pixels with encoder-decoder ConvNets similar to the models
used for affinity prediction in :doc:`../neuron/snemi3d`. The evaluation of
the mitochondria segmentation results is based on the F1 score and
Intersection over Union (IoU).

    .. note:: Unlike other EM connectomics datasets used in these tutorials, the dataset released by Lucchi et al. is an isotropic dataset, which means the spatial resolution along all three axes is the same. Therefore a completely 3D U-Net and data augmentation along x-z and y-z planes (alongside the standard practice of applying augmentation along the x-y plane) is applied.

The scripts needed for this tutorial can be found at ``scripts/main.py``. The corresponding configuration file is ``tutorials/monai_lucchi++.yaml``.

.. figure:: ../../_static/img/lucchi_qual.png
    :align: center
    :width: 800px

A benchmark model's qualitative results on the Lucchi dataset, presented without any post-processing

0 - Setup environment
^^^^^^^^^^^^^^^^^^^^^

Activate the PyTorch Connectomics environment:

.. code-block:: bash

    source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

1 - Get the data
^^^^^^^^^^^^^^^^

The Lucchi++ dataset is available at:

.. code-block:: bash

    /projects/weilab/weidf/lib/pytorch_connectomics/datasets/Lucchi++

For description of the data please check `the author page <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.

2 - Visualize the data
^^^^^^^^^^^^^^^^^^^^^^

Before training, you can visualize the dataset using Neuroglancer:

.. code-block:: bash

    just visualize tutorials/monai_lucchi++.yaml --mode train

This will launch a Neuroglancer instance to explore the training data.

3 - Run training
^^^^^^^^^^^^^^^^

**On SLURM cluster** (recommended for multi-GPU training):

.. code-block:: bash

    just slurm weilab 8 4 "train monai lucchi++"

This launches a training job on the ``weilab`` partition with 8 GPUs and 4 CPUs per GPU.

**On local machine** (single or multi-GPU):

.. code-block:: bash

    just train monai lucchi++

The training script automatically uses PyTorch Lightning with distributed data-parallel (DDP) training for multiple GPUs, enabling synchronized batch normalization (SyncBN) and efficient distributed training.

4 - Monitor training progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can monitor the training progress with TensorBoard:

.. code-block:: bash

    just tensorboard monai_lucchi++

This will launch TensorBoard and display training metrics, losses, and validation results in real-time.

5 - Test the model
^^^^^^^^^^^^^^^^^^

After training completes, test the model on the test set:

.. code-block:: bash

    just test monai lucchi++ outputs/lucchi++_monai_unet/20251012_011259/checkpoints/last.ckpt

Replace the checkpoint path with your actual trained model checkpoint. The checkpoint is typically saved in ``outputs/lucchi++_monai_unet/{timestamp}/checkpoints/last.ckpt``.

6 - Visualize results
^^^^^^^^^^^^^^^^^^^^^

After testing, visualize the prediction results using Neuroglancer:

.. code-block:: bash

    just visualize tutorials/monai_lucchi++.yaml test --port 5005 \
        --volumes pred:image:outputs/lucchi++_monai_unet/results/test_im_prediction.h5:5-5-5

This will launch a Neuroglancer instance on port 5005 displaying the predicted segmentation overlaid on the test images.

.. note::
    - ``pred``: Layer name in Neuroglancer
    - ``image``: Volume type (can be ``image`` for raw data or ``segmentation`` for labels)
    - Path to the prediction HDF5 file
    - ``5-5-5``: Voxel resolution in nm (z-y-x)

7 - Run evaluation
^^^^^^^^^^^^^^^^^^

Since the ground-truth label of the test set is public, we can run the evaluation locally:

.. code-block:: python

    from connectomics.metrics import adapted_rand
    from connectomics.data.io import read_hdf5

    # Load prediction and ground truth
    pred = read_hdf5('outputs/lucchi++_monai_unet/results/test_im_prediction.h5')
    gt = read_hdf5('datasets/Lucchi++/test_label.h5')

    # Prepare for evaluation
    pred = (pred / 255).astype(np.uint8)  # output is casted to uint8 with range [0,255]
    gt = (gt != 0).astype(np.uint8)
    thres = [0.4, 0.6, 0.8]  # evaluate at multiple thresholds
    scores = get_binary_jaccard(pred, gt, thres)

The prediction can be further improved by conducting median filtering to remove noise:

.. code-block:: python

    from connectomics.utils.evaluate import get_binary_jaccard
    from connectomics.utils.process import binarize_and_median

    pred = (pred / 255).astype(np.uint8)  # output is casted to uint8 with range [0,255]
    pred = binarize_and_median(pred, size=(7,7,7), thres=0.8)
    gt = (gt != 0).astype(np.uint8)
    scores = get_binary_jaccard(pred, gt)  # prediction is already binarized

Our pretrained model achieves a foreground IoU and IoU of **0.892** and **0.943** on the test set, respectively. The results are better or on par with state-of-the-art approaches. Please check `BENCHMARK.md <https://github.com/zudi-lin/pytorch_connectomics/blob/master/BENCHMARK.md>`_ for detailed performance comparison and the pre-trained models.
