EM-R50 (Synaptic Polarity Detection)
======================================

This tutorial provides step-by-step guidance for synaptic polarity
detection with the EM-R50 dataset released by
`Lin et al. <http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630103.pdf>`__
in 2020. This task differs from synaptic cleft detection in two ways:
this one requires distinguishing different synapses, while cleft
detection only needs the binary foreground mask for evaluation; and the
polarity detection task requires separated pre-synaptic and post-synaptic
masks. The evaluation metric is an IoU-based F1 score. The sparsity and
diversity of synapses make this task challenging.

    .. note::
        We tackle the task using a bottom-up approach that first generates the segmentation masks of synaptic regions and then apply post-processing algorithms like connected component labeling to separate individual synapses. Our segmentation model uses a model target of three channels. The three channels are **pre-synaptic region**, **post-synaptic region** and **synaptic region** (union of the first two channels), respectively.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``.
Synaptic partner datasets are configured through ``data.train``/``data.val``/``data.test`` and loaded by
:func:`connectomics.training.lightning.create_datamodule`.

.. figure:: ../../_static/img/polarity_qual.png
    :align: center
    :width: 800px

Qualitative results of the synaptic polarity prediction on the EM-R50 dataset. The three-channel outputs that consist of pre-synaptic region, post-synaptic region and their
union (synaptic region) are visualizd in color on the EM images. The single flows from the magenta sides to the cyan sides between neurons.

1 - Get the dataset
^^^^^^^^^^^^^^^^^^^^^

Download the example dataset for synaptic polarity detection from our server:

.. code-block:: none

    wget http://rhoana.rc.fas.harvard.edu/dataset/jwr15_synapse.zip

2 - Run training
^^^^^^^^^^^^^^^^^^

The training and inference script can take a list of volumes (or a long string of paths that can be separated by `'@'`)
in either the yaml config file or by command-line arguments.

.. code-block:: none

    source activate py3_torch
    python -u scripts/main.py \
    --config-base configs/JWR15/synapse/JWR15-Synapse-Base.yaml \
    --config-file configs/JWR15/synapse/JWR15-Synapse-BCE.yaml
..

   .. tip::
    We add **higher weights** to the foreground pixels and apply **rejection sampling** to reject samples without synapes during training to heavily penalize false negatives. This is beneficial for down-stream proofreading and analysis as correcting false positives is much easier than finding missing synapses in the vast volumes.

3 - Visualize the training progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

    tensorboard --logdir outputs/Synaptic_Polarity_UNet

4 - Run inference
^^^^^^^^^^^^^^^^^^

.. code-block:: none

    source activate py3_torch
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
    --config-file configs/Synaptic-Polarity.yaml --inference \
    --checkpoint outputs/Synaptic_Polarity_UNet/volume_100000.pth.tar

..

   .. note::
    The path to images for inference/testing are not specified in the configuration file. Please change the ``INFERENCE.IMAGE_NAME`` option in ``configs/Synaptic-Polarity.yaml``.

5 - Post-process
^^^^^^^^^^^^^^^^^

Then convert the predicted probability into segmentation masks in post-processing. Specifically,
we use :func:`connectomics.utils.process.polarity2instance` to convert the predictions into instance or semantic
masks based on the downstream application.

6 - Learning exclusive polarity masks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The tutorial shown above predicts three channels *independently* with binary cross-entropy losses (BCE) using
the following model configurations:

.. code-block:: yaml

    MODEL:
      TARGET_OPT: ["1"]
      LOSS_OPTION: [["WeightedBCEWithLogitsLoss"]]
      LOSS_WEIGHT: [[1.0]]
      WEIGHT_OPT: [["1"]]
      OUTPUT_ACT: [["none"]]
    INFERENCE:
      OUTPUT_ACT: ["sigmoid"]

Because the three channels are not exclusive, overlap can happen between pre- and post-synaptic masks. Therefore we
also provide a config file to conduct standard semantic segmentation with exclusive masks. The main configurations are

.. code-block:: yaml

    MODEL:
      TARGET_OPT: ["1-1"] # exclusive pos and neg masks
      LOSS_OPTION: [["WeightedCE"]]
      LOSS_KWARGS_KEY: [[["class_weight"]]]
      LOSS_KWARGS_VAL: [[[[1.0, 10.0, 10.0]]]] # class weights
      LOSS_WEIGHT: [[1.0]]
      OUTPUT_ACT: [["none"]]
    INFERENCE:
      OUTPUT_ACT: ["softmax"]

The prediction of the non-exclusive synaptic masks can also be converted into instance masks to identify individual
synapse instances using :func:`connectomics.utils.process.polarity2instance` with the option ``exclusive=True``.
