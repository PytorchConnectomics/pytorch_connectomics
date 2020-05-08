Synaptic Partner Detection
==========================

This tutorial provides step-by-step guidance for synaptic partner detection with the benchmark datasets.
We consider the task as a semantic segmentation task and predict both the pre-synaptic and post-synaptic pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. 
The evaluation of the synapse detection results is based on the F1 score. 

.. note::
    Our segmentation task consists of 2 targets and 3 channels. Targets are background and synapses whereas channels are pre-synapse, post-synapse and background.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``.  The pytorch dataset class of synapses is :class:`torch_connectomics.data.dataset.SynapseDataset`.


#. Dataset can be found on the Harvard RC server here:

        Image:

        .. code-block:: none

            /n/pfister_lab2/Lab/vcg_connectomics/cerebellum_P7/gt-syn/syn_0922_im_orig.h5
        Label:

        .. code-block:: none

            /n/pfister_lab2/Lab/vcg_connectomics/cerebellum_P7/gt-syn/syn_0922_seg.h5

#. Run the training script. The training and inference script can take a list of volumes (separated by '@' in the code below) and conduct training/inference at the same time.

    .. code-block:: none

        $ source activate py3_torch
        $ python -u train.py -i /n/pfister_lab2/Lab/vcg_connectomics/cerebellum_P7/ \
         -din gt-syn/syn_0922_im_orig.h5 -dln gt-syn/syn_0922_seg.h5 \
         -o outputs/unetv0_syn -lr 1e-03 --iteration-total 50000 --iteration-save 5000 -mi 8,256,256 \
         -ma unet_residual_3d -moc 3 -to 1.2,0 -lo 1,1 -wo 1,1 -lw 1,1 -g 4 -c 4 -b 8


    - data: ``i/o/din/dln`` (input folder/output folder/train volume/train label)
    - optimization: ``lr/iteration-total/iteration-save`` (learning rate/total #iterations/#iterations to save)
    - model: ``mi/ma/moc`` (input size/architecture/#output channel)
    - loss: ``to/lo/wo`` (target option/loss option/weight option)
    - system: ``g/c/b`` (#GPU/#CPU/batch size)

#. Visualize the training progress. More info `here <https://vcg.github.io/newbie-wiki/build/html/computation/machine_rc.html>`_:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on image volumes:

    .. code-block:: none

        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u test.py \
        -i /n/pfister_lab2/Lab/vcg_connectomics/cerebellum_P7/ -din gt-syn/syn_0922_im_orig.h5 \
        -o outputs/unetv0_syn/result/ -mi 8,256,256 -ma unet_residual_3d -moc 1 -g 4 -c 4 -b 32 \
        -mpt outputs/unetv0_syn/log2020-04-17_18-55-44/volume_49999.pth -mpi 49999 -dp 8,64,64 -tam mean -tan 4

    - pre-train model: ``mpt/mpi`` (model path/iteration number)
    - test configuration: ``dp/tam/tan`` (data padding/augmentation mode/augmentation number)

#. Evaluate your model using F1 score as metric:

    .. code-block:: none

        $ python /scripts/synapse_evaluate.py

    The evaluation script can be found `here <https://github.com/zudi-lin/pytorch_connectomics/blob/master/tools/evaluation/evaluate_seg.py>`_

#. Calculate connected components of your inferred results:

    .. code-block:: none

        $ python CC.py

    Note: CC.py can be found `here <https://github.com/geekswaroop/EM-ConnectedComponent/blob/master/CC.py>`_.

#. Load your results onto Neuroglancer!

    .. code-block:: none

        $ python neuroG.py

    Note: neuroG.py can be found `here <https://github.com/aarushgupta/NeuroG/blob/master/neuroG.py>`_.