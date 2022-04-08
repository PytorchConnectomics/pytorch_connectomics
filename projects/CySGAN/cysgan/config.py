from yacs.config import CfgNode as CN

def add_cysgan_config(cfg):
    cfg.NEW_DOMAIN = CN()
    cfg.NEW_DOMAIN.SEMI_SUP = True
    cfg.NEW_DOMAIN.IMAGE_NAME = 'imageY.tif'
    