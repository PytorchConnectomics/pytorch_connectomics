from yacs.config import CfgNode as CN

def add_twostream_config(cfg):
    cfg.TWOSTREAM = CN()
    cfg.TWOSTREAM.LATENT_DIM = 512
    cfg.TWOSTREAM.HIDDEN_DIMS = [32, 64, 128, 256, 256, 512]
    cfg.TWOSTREAM.WIDTH = 128 # width of the square input patchs
    cfg.TWOSTREAM.KLD_WEIGHT = 0.01 # weight of the KL divergence
