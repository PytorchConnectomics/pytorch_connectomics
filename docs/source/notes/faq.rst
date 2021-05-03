FAQ
===================================

*1. Why the model input sizes are 2^n+1 (33, 129, 257, etc.) instead of 2^n (32, 128, 256, etc.)?*

Based on the Figure 11 of *Mind the Pad -- CNNs can Develop Blind Spots* (`arXiv <https://arxiv.org/abs/2010.02178>`_), 
using 2^n+1 input sizes for models with zero-padding layers can give a symmetric foveation map, but 2^n leads to 
an *asymmetric* foveation map.

*2. Why the activation functions during training and inference can be different?*

During training, loss functions like ``CrossEntropyLoss`` and ``BCEWithLogitsLoss`` do not require *softmax* or *sigmoid*
activations, but during inference (and visualization) those activations are needed. Besides, multiple losses can be applied
to a single target during training with different activations.
