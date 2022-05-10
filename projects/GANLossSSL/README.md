## Semi-supervised Segmentation with GAN Loss

### Unique Configurations

We show below the list of configurations exclusive for semi-supervised segmentation using
GAN losses, which extends the basic configurations in PyTorch Connectomics.

```yaml
UNLABELED:
  IMAGE_NAME: unlabeled.h5 # name of unlabeled image volume
  GAN_UNLABELED_ONLY: True # apply GAN loss to only the unlabeled
  GAN_WEIGHT: 0.1 # weight of the GAN loss
  D_DILATION: 1 # dilation rate of the CNN discriminator
  SAMPLES_PER_BATCH: 2 # batch size of the unlabeled data
```

There are also configurations for feature-level distribution alignment, under `UNLABELED`:

```yaml
UNLABELED:
  FEATURE_ALIGNMENT:
    ENABLED: False # whether to enable feature-level alignment
    D_IS_ISOTROPIC: None # isotropy of feature discriminator
    D_FILTERS: None # filters of feature discriminator 
    STRIDE_LIST: None # strides of feature discriminator
```

For each of the feature discriminator configurations, setting it to `None` means using the defaults.

### Command

Training command (after `source activate py3_torch`) using distributed data parallel:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.run \
--nproc_per_node=2 --master_port=9967 projects/GANLossSSL/main.py --distributed \
--config-base projects/GANLossSSL/configs/SNEMI/SNEMI-Base.yaml \
--config-file projects/GANLossSSL/configs/SNEMI/SNEMI-Affinity-UNet.yaml
```

Inference command using data parallel:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u projects/GANLossSSL/main.py \
--inference --config-base projects/GANLossSSL/configs/SNEMI/SNEMI-Base.yaml \
--config-file projects/GANLossSSL/configs/SNEMI/SNEMI-Affinity-UNet.yaml \
--checkpoint outputs/SNEMI_UNet_GANLoss/checkpoint_150000.pth.tar
```
