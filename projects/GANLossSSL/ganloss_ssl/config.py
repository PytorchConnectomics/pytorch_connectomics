from yacs.config import CfgNode as CN

def add_ganloss_config(cfg):
    cfg.UNLABELED = CN()
    cfg.UNLABELED.IMAGE_NAME = "unlabeled.h5"
    cfg.UNLABELED.GAN_UNLABELED_ONLY = True
    cfg.UNLABELED.GAN_WEIGHT = 0.1
    cfg.UNLABELED.D_DILATION = 1
    cfg.UNLABELED.SAMPLES_PER_BATCH = None

    # feature-level distribution alignment
    cfg.UNLABELED.FEATURE_ALIGNMENT = CN({"ENABLED": False})
    cfg.UNLABELED.FEATURE_ALIGNMENT.D_IS_ISOTROPIC = None
    cfg.UNLABELED.FEATURE_ALIGNMENT.D_FILTERS = None
    cfg.UNLABELED.FEATURE_ALIGNMENT.STRIDE_LIST = None
