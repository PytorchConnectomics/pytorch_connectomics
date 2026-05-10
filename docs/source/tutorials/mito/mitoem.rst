MitoEM (Instance Segmentation)
================================

This tutorial provides step-by-step guidance for mitochondria segmentation
with the
`MitoEM <https://donglaiw.github.io/page/mitoEM/index.html>`_ dataset
released by
`Wei et al. <https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf>`__
in 2020. We approach the task as a 3D **instance segmentation** task and
provide three different configurations of the model output. We utilize the
``UNet3D`` model similar to the one used in :doc:`../neuron/snemi3d`. The
evaluation of the segmentation results is based on AP-75 (average
precision with an IoU threshold of 0.75).

.. figure:: ../../_static/img/mito_complex.png
    :align: center
    :width: 800px

Complex mitochondria in the MitoEM dataset:(**a**) mitochondria-on-a-string (MOAS), and (**b**) dense tangle of touching instances. Those challenging cases are prevalent but not covered in previous datasets.

    .. note:: The MitoEM dataset has two sub-datasets **MitoEM-Rat** and **MitoEM-Human** based on the source of the tissues. Three training configuration files on **MitoEM-Rat** are provided in ``pytorch_connectomics/configs/MitoEM/`` for different learning setting as described in this `paper <https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf>`_.

..

   .. note:: Since the dataset is very large and cannot always be directly loaded into memory, use lazy HDF5/Zarr loading or filename-based datasets from :mod:`connectomics.data.datasets` instead of the removed ``TileDataset`` path.

..

    .. note:: A benchmark evaluation with validation data and pretrained weights is provided for users at `this Colab notebook <https://colab.research.google.com/drive/1ll3a0F2VbmmKBTQ_RBqSrEsU3gpTUdam>`_.

1 - Dataset introduction
^^^^^^^^^^^^^^^^^^^^^^^^

The dataset is publicly available at both the `project <https://donglaiw.github.io/page/mitoEM/index.html>`_ page and
the `MitoEM Challenge <https://mitoem.grand-challenge.org/>`_ page. To provide a brief description of the dataset:

- ``im``: includes 1,000 single-channel ``*.png`` files (**4096x4096**) of raw EM images (with a spatial resolution of **30x8x8** nm).
  The 1,000 images are splited into 400, 100 and 500 slices for training, validation and inference, respectively.

- ``mito_train/``: includes 400 single-channel ``*.png`` files (**4096x4096**) of instance labels for training. Similarly, the ``mito_val/`` folder contains 100 slices for validation. The ground-truth annotation of the test set (rest 500 slices) is not publicly provided but can be evaluated online at the `MitoEM challenge page <https://mitoem.grand-challenge.org>`_.

2 - Model configuration
^^^^^^^^^^^^^^^^^^^^^^^

Multiple ``*.yaml`` configuration files are provided at ``configs/MitoEM`` for different learning targets:

- ``MitoEM-R-A.yaml``: output 3 channels for predicting the affinty between voxels.

- ``MitoEM-R-AC.yaml``: output 4 channels for predicting both affinity and instance contour.

- ``MitoEM-R-BC.yaml``: output 2 channels for predicting both the binary foreground mask and instance contour.

The lattermost configuration achieves the best overall performance according to our `experiments <https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf>`_. This tutorial will move forward using this configuration file.

3 - Run training
^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -u scripts/main.py \
    --config-base configs/MitoEM/MitoEM-R-Base.yaml \
    --config-file configs/MitoEM/MitoEM-R-BC.yaml

..

    .. note:: By default the path of images and labels are not specified. To run the training scripts, please revise the ``DATASET.IMAGE_NAME``, ``DATASET.LABEL_NAME``, ``DATASET.OUTPUT_PATH`` and ``DATASET.INPUT_PATH`` options in ``configs/MitoEM/MitoEM-R-*.yaml``. The options can also be given as command-line arguments without changing of the ``yaml`` configuration files.

4 (*optional*) - Visualize the training progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    tensorboard --logdir outputs/MitoEM_R_BC/

5 - Run inference
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -u scripts/main.py \
    --config-base configs/MitoEM/MitoEM-R-Base.yaml \
    --config-file configs/MitoEM/MitoEM-R-BC.yaml --inference \
    --checkpoint outputs/MitoEM_R_BC/checkpoint_100000.pth.tar

..

   .. note:: If training on personal data, please change the ``INFERENCE.IMAGE_NAME`` ``INFERENCE.OUTPUT_PATH`` ``INFERENCE.OUTPUT_NAME`` options in ``configs/MitoEM-R-*.yaml`` based on your own data path.

6 - Post-process
^^^^^^^^^^^^^^^^

The post-processing step requires merging output volumes and applying watershed segmentation. As mentioned before, the dataset is very large and cannot be directly loaded into memory for processing. Therefore our code run prediction on smaller chunks sequentially, which produces multiple ``*.h5`` files with the coordinate information. To merge the chunks into a single volume and apply the segmentation algorithm:

.. code-block:: python

    import glob
    import numpy as np
    from connectomics.data.io import read_hdf5
    from connectomics.utils.process import bc_watershed

    output_files = 'outputs/MitoEM_R_BC/test/*.h5' # output folder with chunks
    chunks = glob.glob(output_files)

    # Mitochondria Segmentation
    vol_shape = (2, 500, 4096, 4096) # MitoEM test set
    pred = np.ones(vol_shape, dtype=np.uint8)
    for x in chunks:
        pos = x.strip().split("/")[-1]
        print("process chunk: ", pos)
        pos = pos.split("_")[1].split("-")
        pos = list(map(int, pos))
        chunk = readvol(x)
        pred[:, pos[0]:pos[1], pos[2]:pos[3], pos[4]:pos[5]] = chunk

    # This function process the array in numpy.float64 format.
    # Please allocate enough memory for processing.
    segm = bc_watershed(pred, thres1=0.85, thres2=0.6, thres3=0.8, thres_small=1024)

..

   .. note:: The decoding parameters for the watershed step are a set of reasonable thresholds but not optimal given different segmentation models. We suggest conducting a hyper-parameter search on the validation set to decide the decoding parameters.

The generated segmentation map should be ready for submission to the `MitoEM <https://mitoem.grand-challenge.org/>`_ challenge website for evaluation. Please note that this tutorial only outlines training on **MitoEM-Rat** subset. Results on the **MitoEM-Human** subset, which can be generated using a similar pipeline as above, also need to be provided for online evaluation.

7 (*optional*)- Evaluate on the validation set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Performance on the MitoEM test data subset can only be evaluated on the Grand Challenge website. Users are encouraged to experiment with the metric code on the validation data subset to optimize performance and understand the Challenge's evaluation process. Evaluation is performed with the ``demo.py`` file provided by the `mAP_3Dvolume <https://github.com/ygCoconut/mAP_3Dvolume/tree/master>`__ repository. The ground truth ``.h5`` file can be generated from the 2D images using the following script:

.. code-block:: python

  import glob
  import numpy as np
  from connectomics.data.io import read_hdf5, write_hdf5

  gt_path = "datasets/MitoEM_R/mito_val/*.tif"
  files = sorted(glob.glob(gt_path))

  data = []
  for i, file in enumerate(files):
      print("process chunk: ", i)
      data.append(readvol(file))

  data = np.array(data)
  writeh5("validation_gt.h5", data)

The resulting scores can then be obtained by executing ``python demo.py -gt {path to validation ground truth}.h5 -p {path to segmentation result}.h5``
