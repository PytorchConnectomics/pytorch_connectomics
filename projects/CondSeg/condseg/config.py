from yacs.config import CfgNode as CN


def add_condseg_config(cfg):
    r''' Conditional Segmentation specific configurations'''

    cfg.CONDITIONAL = CN()
    cfg.CONDITIONAL.LABEL_TYPE = None # Type of the labels: seg | syn; defaults to seg
    cfg.CONDITIONAL.INFERENCE_CONDITIONAL = None # Name of the conditional/target file for inferencing