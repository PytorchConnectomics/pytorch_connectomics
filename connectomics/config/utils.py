import os
import warnings
import argparse
from yacs.config import CfgNode
from .defaults import get_cfg_defaults


def load_cfg(args: argparse.Namespace, freeze=True, add_cfg_func=None):
    """Load configurations.
    """
    # Set configurations
    cfg = get_cfg_defaults()
    if add_cfg_func is not None:
        add_cfg_func(cfg)
    if args.config_base is not None:
        cfg.merge_from_file(args.config_base)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Overwrite options given configs with higher priority.
    if args.inference:
        update_inference_cfg(cfg)
    overwrite_cfg(cfg, args)

    if freeze:
        cfg.freeze()
    else:
        warnings.warn("Configs are mutable during the process, "
                      "please make sure that is expected.")
    return cfg


def save_all_cfg(cfg: CfgNode, output_dir: str):
    r"""Save configs in the output directory.
    """
    # Save config.yaml in the experiment directory after combine all
    # non-default configurations from yaml file and command line.
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print("Full config saved to {}".format(path))


def update_inference_cfg(cfg: CfgNode):
    r"""Overwrite configurations (cfg) when running mode is inference. Please
    note that None type is only supported in YACS>=0.1.8.
    """
    # dataset configurations
    if cfg.INFERENCE.INPUT_PATH is not None:
        cfg.DATASET.INPUT_PATH = cfg.INFERENCE.INPUT_PATH
    cfg.DATASET.IMAGE_NAME = cfg.INFERENCE.IMAGE_NAME
    cfg.DATASET.OUTPUT_PATH = cfg.INFERENCE.OUTPUT_PATH

    if cfg.INFERENCE.PAD_SIZE is not None:
        cfg.DATASET.PAD_SIZE = cfg.INFERENCE.PAD_SIZE
    if cfg.INFERENCE.IS_ABSOLUTE_PATH is not None:
        cfg.DATASET.IS_ABSOLUTE_PATH = cfg.INFERENCE.IS_ABSOLUTE_PATH

    if cfg.INFERENCE.DO_CHUNK_TITLE is not None:
        cfg.DATASET.DO_CHUNK_TITLE = cfg.INFERENCE.DO_CHUNK_TITLE
    if cfg.INFERENCE.DATA_SCALE is not None:
        cfg.DATASET.DATA_SCALE = cfg.INFERENCE.DATA_SCALE

    # model configurations
    if cfg.INFERENCE.INPUT_SIZE is not None:
        cfg.MODEL.INPUT_SIZE = cfg.INFERENCE.INPUT_SIZE
    if cfg.INFERENCE.OUTPUT_SIZE is not None:
        cfg.MODEL.OUTPUT_SIZE = cfg.INFERENCE.OUTPUT_SIZE
    # specify feature maps to return as inference time
    cfg.MODEL.RETURN_FEATS = cfg.INFERENCE.MODEL_RETURN_FEATS

    # output file name(s)
    out_name = cfg.INFERENCE.OUTPUT_NAME
    name_lst = out_name.split(".")
    if cfg.DATASET.DO_CHUNK_TITLE or cfg.INFERENCE.DO_SINGLY:
        assert len(name_lst) <= 2, \
            "Invalid output file name is given."
        if len(name_lst) == 2:
            cfg.INFERENCE.OUTPUT_NAME = name_lst[0]
    else:
        if len(name_lst) == 1:
            cfg.INFERENCE.OUTPUT_NAME = name_lst[0] + '.h5'

    for topt in cfg.MODEL.TARGET_OPT:
        # For multi-class semantic segmentation and quantized distance
        # transform, no activation function is applied at the output layer
        # during training. For inference where the output is assumed to be
        # in (0,1), we apply softmax.
        if topt[0] in ['5', '9'] and cfg.MODEL.OUTPUT_ACT == 'none':
            cfg.MODEL.OUTPUT_ACT = 'softmax'
            break


def overwrite_cfg(cfg: CfgNode, args: argparse.Namespace):
    r"""Overwrite some configs given configs or args with higher priority.
    """
    # Distributed training:
    if args.distributed:
        cfg.SYSTEM.DISTRIBUTED = True
        cfg.SYSTEM.PARALLEL = 'DDP'

    # Update augmentation options when valid masks are specified
    if cfg.DATASET.VALID_MASK_NAME is not None:
        assert cfg.DATASET.LABEL_NAME is not None, \
            "Using valid mask is only supported when target label is given."
        assert cfg.AUGMENTOR.ADDITIONAL_TARGETS_NAME is not None
        assert cfg.AUGMENTOR.ADDITIONAL_TARGETS_TYPE is not None

        cfg.AUGMENTOR.ADDITIONAL_TARGETS_NAME += ['valid_mask']
        cfg.AUGMENTOR.ADDITIONAL_TARGETS_TYPE += ['mask']

    # Model I/O size
    for x in cfg.MODEL.INPUT_SIZE:
        if x % 2 == 0 and not cfg.MODEL.POOLING_LAYER:
            warnings.warn(
                "When downsampling by stride instead of using pooling "
                "layers, the cfg.MODEL.INPUT_SIZE are expected to contain "
                "numbers of 2n+1 to avoid feature mis-matching, "
                "but get {}".format(cfg.MODEL.INPUT_SIZE))
            break
        if x % 2 == 1 and cfg.MODEL.POOLING_LAYER:
            warnings.warn(
                "When downsampling by pooling layers the cfg.MODEL.INPUT_SIZE "
                "are expected to contain even numbers to avoid feature mis-matching, "
                "but get {}".format(cfg.MODEL.INPUT_SIZE))
            break

    # Mixed-precision training (only works with DDP)
    cfg.MODEL.MIXED_PRECESION = (
        cfg.MODEL.MIXED_PRECESION and args.distributed)

    # Scaling factors for image, label and valid mask
    if cfg.DATASET.IMAGE_SCALE is None:
        cfg.DATASET.IMAGE_SCALE = cfg.DATASET.DATA_SCALE
    if cfg.DATASET.LABEL_SCALE is None:
        cfg.DATASET.LABEL_SCALE = cfg.DATASET.DATA_SCALE
    if cfg.DATASET.VALID_MASK_SCALE is None:
        cfg.DATASET.VALID_MASK_SCALE = cfg.DATASET.DATA_SCALE

    # Disable label reducing for semantic segmentation to avoid class shift
    for topt in cfg.MODEL.TARGET_OPT:
        if topt[0] == '9': # semantic segmentation mode
            cfg.DATASET.REDUCE_LABEL = False
            break


def validate_cfg(cfg: CfgNode):
    num_target = len(cfg.MODEL.TARGET_OPT)
    assert len(cfg.INFERENCE.OUTPUT_ACT) == num_target, \
        "Activations need to be specified for each learning target."


def convert_cfg_markdown(cfg):
    """Converts given cfg node to markdown for tensorboard visualization.
    """
    r = ""
    s = []

    def helper(cfg):
        s_indent = []
        for k, v in sorted(cfg.items()):
            seperator = " "
            attr_str = "  \n{}:{}{}  \n".format(str(k), seperator, str(v))
            s_indent.append(attr_str)
        return s_indent

    for k, v in sorted(cfg.items()):
        seperator = "&nbsp;&nbsp;&nbsp;" if isinstance(v, str) else "  \n"
        val = helper(v)
        val_str = ""
        for line in val:
            val_str += line
        attr_str = "##{}:{}{}  \n".format(str(k), seperator, val_str)
        s.append(attr_str)
    for line in s:
        r += "  \n" + line + "  \n"
    return r
