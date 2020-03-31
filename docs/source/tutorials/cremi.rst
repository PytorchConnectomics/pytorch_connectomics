Synaptic Cleft Detection
==========================

This tutorial provides step-by-step guidance for synaptic cleft detection with `CREMI <https://cremi.org>`_ benchmark datasets.
We consider the task as a semantic segmentation task and predict the synapse pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. 
The evaluation of the synapse detection results is based on the F1 score and average distance. See `CREMI metrics <https://cremi.org/metrics/>`_
for more details.

.. note::
    We preform re-alignment of the original CREMI image stacks and also remove the crack artifacts. Please reverse 
    the alignment before submitting the test prediction to the CREMI challenge.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. Need to pass the argument ``--task 1``
when executing the ``train.py`` and ``test.py`` scripts. The pytorch dataset class of synapses is :class:`torch_connectomics.data.dataset.SynapseDataset`.

#. Get the dataset:

    #. Download the dataset from our server:

        .. code-block:: none

            wget http://hp03.mindhackers.org/rhoana_product/dataset/cremi.zip
    
    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.

#. Run the training script. The training and inference script can take a list of volumes and conduct training/inference at the same time.

    .. code-block:: none

        $ source activate py3_torch
        $ python -u train.py -i /path/to/CREMI/ \
          -din image/im_A_v2_200.h5@image/im_B_v2_200.h5@image/im_C_v2_200.h5 \
          -dln gt-syn/syn_A_v2_200.h5@gt-syn/syn_B_v2_200.h5@gt-syn/syn_C_v2_200.h5 \
          -o outputs/unetv0_syn -lr 1e-03 --iteration-total 50000 --iteration-save 5000 \
          -mi 8,256,256 -ma unet_residual_3d -moc 1 \
          -to 0 -lo 1 -wo 1 -g 4 -c 4 -b 8 

    - data: ``i/o/din/dln`` (input folder/output folder/train volume/train label)
    - optimization: ``lr/iteration-total/iteration-save`` (learning rate/total #iterations/#iterations to save)
    - model: ``mi/ma/moc`` (input size/architecture/#output channel)
    - loss: ``to/lo/wo`` (target option/loss option/weight option)
    - system: ``g/c/b`` (#GPU/#CPU/batch size)


#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on image volumes:

    .. code-block:: none

        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u test.py -i /path/to/CREMI/ \
          -din image/im_A_v2_200.h5@image/im_B_v2_200.h5@image/im_C_v2_200.h5 \
          -o outputs/unetv0_syn/result -mi 8,256,256 -ma unet_residual_3d -moc 1 \
          -g 4 -c 4 -b 32 
          -mpt outputs/unetv0_syn/volume_49999.pth -mpi 49999 -dp 8,64,64 -tam mean -tan 4

    - pre-train model: ``mpt/mpi`` (model path/iteration number)
    - test configuration: ``dp/tam/tan`` (data padding/augmentation mode/augmentation number)
