from yacs.config import CfgNode as CN

def add_twostream_config(cfg):
    cfg.TWOSTREAM = CN()
    cfg.TWOSTREAM.IMAGE_VAE = True
    cfg.TWOSTREAM.LABEL_TYPE = "syn"
    cfg.TWOSTREAM.LATENT_DIM = 512
    cfg.TWOSTREAM.HIDDEN_DIMS = [32, 64, 128, 256, 256, 512]
    cfg.TWOSTREAM.WIDTH = 128 # width of the square input patchs
    cfg.TWOSTREAM.KLD_WEIGHT = 1e-4 # weight of the KL divergence
