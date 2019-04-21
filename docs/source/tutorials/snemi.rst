Neuron Segmentation
=======================

This tutorial provides step-by-step guidance for neuron segmentation with SENMI3D benchmark datasets.
Dense neuron segmentation in electronic microscopy (EM) images belongs to the category of instance segmentation.
The methodology is to first predict the affinity map of pixels with an encoder-decoder ConvNets and 
then generate the segmentation map using a segmentation algorithm (e.g. watershed). 

.. note::
    Before running neuron segmentation, please take a look at the `demo <https://github.com/zudi-lin/pytorch_connectomics/tree/master/demo>`_
    to get familiar with the datasets and have a sense of how the affinity graphs look like.

#. Get the dataset:

#. Run the training script:

#. Visualize the training progress:

#. Run inference on image volumes:

#. Gnerate segmentation and run evaluation:

#. Conduct error analysis: