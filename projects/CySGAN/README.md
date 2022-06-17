## Instance Segmentation in Unlabeled Modalities via Cyclic Segmentation GAN

[[arXiv](https://arxiv.org/abs/2204.03082)] [[Project Page](https://connectomics-bazaar.github.io/proj/CySGAN/index.html)]

### Abstract

Instance segmentation for unlabeled imaging modalities is a challenging but essential task as collecting expert annotation can be expensive and time-consuming. Existing works segment a new modality by either deploying a pre-trained model optimized on diverse training data or conducting domain translation and image segmentation as two independent steps. In this work, we propose a novel *Cyclic Segmentation* Generative Adversarial Network (**CySGAN**) that conducts image translation and instance segmentation jointly using a unified framework. Besides the CycleGAN losses for image translation and supervised losses for the annotated source domain, we introduce additional self-supervised and segmentation-based adversarial objectives to improve the model performance by leveraging unlabeled target domain images. We benchmark our approach on the task of 3D neuronal nuclei segmentation with annotated electron microscopy (EM) images and unlabeled expansion microscopy (ExM) data. Our CySGAN outperforms both pre-trained generalist models and the baselines that sequentially conduct image translation and segmentation. Our implementation and the newly collected, densely annotated ExM nuclei dataset, named *NucExM*, are publicly available. 

### Unique Configurations

We show below the list of configurations exclusive for semi-supervised segmentation using
GAN losses, which extends the basic configurations in PyTorch Connectomics (PyTC).

```yaml
NEW_DOMAIN:
  IMAGE_NAME: imageY.tif # image file name for the new domain
  SEMI_SUP: True # use semi-supervised segmentation loss. default: True
```

This work focus on nuclei segmentation using the BCD representation as in the [NucMM Dataset](https://arxiv.org/abs/2107.05840) paper. Other learning targets defined in PyTC, including the 3D affinity map, can also be easily configured to build a CySGAN model.

### Training Data Preparation

CySGAN training requires an annotated source domain and an unlabeled target domain. We recommend the users rescale the source and target volumes to have visually similar object sizes for optimal performance. Please remember to update the source and target file names in the [CySGAN-Base.yaml](configs/CySGAN-Base.yaml) file:

```yaml
DATASET:
  IMAGE_NAME: imageX.tif # image file name (source domain)
  LABEL_NAME: labelX.tif # label file name (source domain)
NEW_DOMAIN:
  IMAGE_NAME: imageY.tif # image file name for the new domain
```

### Command

Training command (after `source activate py3_torch`) using distributed data parallel:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch \
--nproc_per_node=2 --master_port=9977 projects/CySGAN/main.py --distributed \
--config-base projects/CySGAN/configs/CySGAN-Base.yaml \
--config-file projects/CySGAN/configs/CySGAN-UNet-BCD.yaml
```

Inference command using data parallel (remember to change the checkpoint path if you are using a pretrained model):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u projects/CySGAN/main.py \
--inference --config-base projects/CySGAN/configs/CySGAN-Base.yaml \
--config-file projects/CySGAN/configs/CySGAN-UNet-BCD.yaml \
--checkpoint outputs/CySGAN_BCD/checkpoint_100000.pth.tar
```

To convert the predicted probability maps into instance segmentation, use the following post-processing code. Please note that the translated image is not required for segmentation, only the BCD representations are needed.

```python
from connectomics.data.utils import readvol, savevol
from connectomics.utils.process import bcd_watershed
pred = readvol("CySGAN_BCD/test/pred.h5") # your own path to prediction

segm_pred = pred[1:] # the 1st channel is translated image
segm, seed = bcd_watershed(
    segm_pred, thres1=0.9, thres2=0.2, thres3=0.7, thres4=0.5, thres5=-0.1, 
    return_seed=True, seed_thres=32, thres_small=256)
savevol("segm.h5", segm) # your own path to save segm
```

### Citation

```bibtex
@article{lauenburg2022instance,
  title={Instance Segmentation of Unlabeled Modalities via Cyclic Segmentation GAN},
  author={Lauenburg, Leander and Lin, Zudi and Zhang, Ruihan and Santos, M{\'a}rcia dos and Huang, Siyu and Arganda-Carreras, Ignacio and Boyden, Edward S and Pfister, Hanspeter and Wei, Donglai},
  journal={arXiv preprint arXiv:2204.03082},
  year={2022}
}
```

Please also cite the [PyTorch Connectomics](https://github.com/zudi-lin/pytorch_connectomics#citation) package if you find our codebase useful.

### Acknowledgements

This work has been partially supported by NSF awards IIS-1835231 and IIS-2124179 and NIH grant 5U54CA225088-03.
