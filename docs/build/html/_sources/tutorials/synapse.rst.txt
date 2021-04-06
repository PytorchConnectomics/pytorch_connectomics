Synapse Detection
==================

Introduction
-------------

A `synapse <https://en.wikipedia.org/wiki/Synapse>`__ is an essential structure in the nervous system that allows an electric or chemical signal to be
passed to another neuron or an effector cell (*e.g.*, muscle fiber). Identification of synapses is important for reconstructing the wiring diagram of 
neurons to enable new insights into the workings of the brain, which is the long-term goal of the connectomics area. Signal flows in one direction
at a synapse, therefore each synapse usually consists of a pre-synaptic region and a post-synaptic region.

This tutorial has two parts. In the first part, you will learn how to detect **synaptic clefts** by predicting the synaptic cleft pixels on the 
`CREMI Challenge <https://cremi.org>`__ dataset from adult *Drosophila melanogaster* brain tissue. This dataset is released in 2016. In the second part, 
you will learn how to predict the **synaptic polarity masks** to demonstrate the signal flow between neurons using the dataset released 
by `Lin et al. <http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630103.pdf>`__ in 2020. The brain sample is collected from Layer II/III in 
the primary visual cortex of an adult rat.

Synaptic Cleft Detection
-------------------------

This tutorial provides step-by-step guidance for synaptic cleft detection with `CREMI <https://cremi.org>`_ benchmark datasets.
We consider the task as a semantic segmentation task and predict the synapse pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. 
The evaluation of the synapse detection results is based on the F1 score and average distance. See `CREMI metrics <https://cremi.org/metrics/>`_
for more details.

.. note::

    We preform re-alignment of the original CREMI image stacks and also remove the crack artifacts. Please reverse 
    the alignment before submitting the test prediction to the CREMI challenge.

Script needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. The *YAML* configuration files can be found at ``pytorch_connectomics/configs/``, which 
stores the common settings for model training and inference. Other default configuration options can be found at ``pytorch_connectomics/connectomics/config/``. The pytorch 
dataset class of the synaptic cleft detection task is :class:`connectomics.data.dataset.VolumeDataset`.

.. figure:: ../_static/img/cremi_qual.png
    :align: center
    :width: 800px

    Qualitative results of the synaptic cleft prediction (red segments) on the CREMI challenge test volumes. The three images from left to right are
    cropped from volume A+, B+, and C+, respectively.

#. Get the dataset:

    Download the dataset from the Harvard RC server:

        .. code-block:: none

            wget http://rhoana.rc.fas.harvard.edu/dataset/cremi.zip
    
    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.

    .. note::
        If you use the original CREMI challenge datasets or the data processed by yourself, the file names can be
        different from the default ones. In such case, please change the corresponding entries, including ``IMAGE_NAME``, 
        ``LABEL_NAME`` and ``INPUT_PATH`` in the `CREMI config file <https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/CREMI-Synaptic-Cleft.yaml>`_.

#. Run the main.py script for training. This script can take a list of volumes and conduct training/inference at the same time.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/CREMI-Synaptic-Cleft.yaml

    - ``config-file``: configuration setting for the current experiment.

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run the main.py script for inference:

    .. code-block:: none

        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/CREMI-Synaptic-Cleft.yaml \
          --checkpoint outputs/CREMI_syn_baseline/volume_50000.pth.tar \
          --inference

    - ``config-file``: configuration setting for current experiments.
    - ``inference``: will run inference when given, otherwise will run training instead.
    - ``checkpoint``: the pre-trained checkpoint file for inference.

Synaptic Polarity Detection
----------------------------

This tutorial provides step-by-step guidance for synaptic polarity detection with the EM-R50 dataset released by `Lin et al. <http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630103.pdf>`__ in 2020. 
This task is different from the synaptic cleft detection task in two aspects. First, this one requires distinguishing different synapses, while the cleft detection task
only needs the binary foreground mask for evaluation. Second, the polarity detection task also requires separated pre-synaptic and post-synaptic masks. 
The evaluation metric of the synaptic polarity detection results is an IoU-based F1 score. The sparsity and diversity of synapses make the task challenging. 

.. note::
    We tackle the task using a bottom-up approach that first generates the segmentation masks of synaptic regions and then apply post-processing algorithms like 
    connected component labeling to separate individual synapses. Our segmentation model uses a model target of three channels. The three channels 
    are **pre-synaptic region**, **post-synaptic region** and **synaptic region** (union of the first two channels), respectively. 

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``.  
The pytorch dataset class of synaptic partners is :class:`connectomics.data.dataset.VolumeDataset`.

.. figure:: ../_static/img/polarity_qual.png
    :align: center
    :width: 800px

    Qualitative results of the synaptic polarity prediction on the EM-R50 dataset. The three-channel outputs that consist of pre-synaptic region, post-synaptic region and their
    union (synaptic region) are visualizd in color on the EM images. The single flows from the magenta sides to the cyan sides between neurons.

#. Get the dataset:

    Download the example dataset for synaptic polarity detection from our server:

        .. code-block:: none

            wget http://rhoana.rc.fas.harvard.edu/dataset/jwr15_synapse.zip

#. Run the training script. The training and inference script can take a list of volumes (separated by '@') in either the yaml config file or by command-line arguments.

    .. note::
        By default the path of images and labels are not specified. To 
        run the training scripts, please revise the ``IMAGE_NAME``, ``LABEL_NAME``
        and ``INPUT_PATH`` options in ``configs/Synaptic-Polarity.yaml``.
        The options can also be given as command-line arguments without changing of the ``yaml`` configuration files.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Synaptic-Polarity.yaml
          
    .. note::
        We add **higher weights** to the foreground pixels and apply **rejection sampling** to reject samples without synapes during training to heavily penalize
        false negatives. This is beneficial for down-stream proofreading and analysis as correcting false positives is much easier than finding missing synapses in the
        vast volumes.

#. Visualize the training progress. More info `here <https://vcg.github.io/newbie-wiki/build/html/computation/machine_rc.html>`_:

    .. code-block:: none

        $ tensorboard --logdir outputs/synaptic_polarity

#. Run inference on image volumes:

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Synaptic-Polarity.yaml --inference \
          --checkpoint outputs/synaptic_polarity/volume_xxxxx.pth.tar

    .. note::
        By default the path of images for inference are not specified. Please change 
        the ``INFERENCE.IMAGE_NAME`` option in ``configs/Synaptic-Polarity.yaml``.

#. Apply post-processing algorithms. Use the ``polarity2instance`` function (`link <https://zudi-lin.github.io/pytorch_connectomics/build/html/modules/utils.html#connectomics.utils.process.polarity2instance>`_) to 
   convert the probability map into instance/semantic segmentation masks based on the application.
