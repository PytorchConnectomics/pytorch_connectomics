import os
import warnings
import argparse
from yacs.config import CfgNode
from .defaults import get_cfg_defaults

def load_cfg(args: argparse.Namespace):
    """Load configurations.
    """
    # Set configurations
    cfg = get_cfg_defaults()
    if args.config_base is not None:
        cfg.merge_from_file(args.config_base)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Overwrite options given configs with higher priority.
    if args.inference:
        update_inference_cfg(cfg)
    overwrite_cfg(cfg, args)
    cfg.freeze()
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
    # Dataset configurations:
    if cfg.INFERENCE.INPUT_PATH is not None:
        cfg.DATASET.INPUT_PATH = cfg.INFERENCE.INPUT_PATH
    cfg.DATASET.IMAGE_NAME = cfg.INFERENCE.IMAGE_NAME
    cfg.DATASET.OUTPUT_PATH = cfg.INFERENCE.OUTPUT_PATH

    if cfg.INFERENCE.PAD_SIZE is not None:
        cfg.DATASET.PAD_SIZE = cfg.INFERENCE.PAD_SIZE
    if cfg.INFERENCE.IS_ABSOLUTE_PATH is not None:
        cfg.DATASET.IS_ABSOLUTE_PATH = cfg.INFERENCE.IS_ABSOLUTE_PATH

    # Model configurations:
    if cfg.INFERENCE.INPUT_SIZE is not None:
        cfg.MODEL.INPUT_SIZE = cfg.INFERENCE.INPUT_SIZE
    if cfg.INFERENCE.OUTPUT_SIZE is not None:
        cfg.MODEL.OUTPUT_SIZE = cfg.INFERENCE.OUTPUT_SIZE

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

    # Target options:
    for topt in cfg.MODEL.TARGET_OPT:
        if topt[0] == '5': # quantized distance transform
            cfg.MODEL.OUT_PLANES = 11
            assert len(cfg.MODEL.TARGET_OPT) == 1, \
                "Multi-task learning with quantized distance transform " \
                "is currently not supported."

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
        if x % 2 == 0 and not cfg.MODEL.POOING_LAYER:
            warnings.warn(
                "When downsampling by stride instead of using pooling " \
                "layers, the cfg.MODEL.INPUT_SIZE are expected to contain " \
                "numbers of 2n+1 to avoid feature mis-matching, " \
                "but get {}".format(cfg.MODEL.INPUT_SIZE))
            break
        if x % 2 == 1 and cfg.MODEL.POOING_LAYER:
            warnings.warn(
                "When downsampling by pooling layers the cfg.MODEL.INPUT_SIZE " \
                "are expected to contain even numbers to avoid feature mis-matching, " \
                "but get {}".format(cfg.MODEL.INPUT_SIZE))
            break

def validate_cfg(cfg: CfgNode):
    num_target = len(cfg.MODEL.TARGET_OPT)
    assert len(cfg.INFERENCE.OUTPUT_ACT) == num_target

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

def convert_model_to_markdown(model):
    def extra_repr_func() -> str:
        return ''

    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * '  \t') + line for line in s]
        s = '  \n'.join(s)
        s = first + '  \n' + s
        return s
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = extra_repr_func()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    # for key, module in self._modules.items():
    for key, module in model._modules.items():
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        tmp_str = "#####({}".format(key) + '):'
        child_lines.append(tmp_str + mod_str)
    lines = extra_lines + child_lines

    main_str = model.__class__.__name__ + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '  \n' + '  \n'.join(lines) + '  \n'

    main_str += ')'
    return main_str