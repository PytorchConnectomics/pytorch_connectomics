import torch
import torch.nn as nn

from .arch.nnunet_models import create_nnunet_model, NNUNET_MODEL_REGISTRY

# Modern nnUNet models only
MODEL_MAP = {
    'mednext_s': lambda **kwargs: create_nnunet_model('mednext_s', **kwargs),
    'mednext_b': lambda **kwargs: create_nnunet_model('mednext_b', **kwargs),
    'mednext_m': lambda **kwargs: create_nnunet_model('mednext_m', **kwargs),
    'mednext_l': lambda **kwargs: create_nnunet_model('mednext_l', **kwargs),
    'monai_unet3d': lambda **kwargs: create_nnunet_model('monai_unet3d', **kwargs),
    'monai_basic_unet3d': lambda **kwargs: create_nnunet_model('monai_basic_unet3d', **kwargs),
    'monai_unetr': lambda **kwargs: create_nnunet_model('monai_unetr', **kwargs),
    'monai_swin_unetr': lambda **kwargs: create_nnunet_model('monai_swin_unetr', **kwargs),
}


def build_model(cfg, device=None, rank=None):
    """
    Build modern nnUNet model from configuration.
    """
    model_arch = cfg.MODEL.ARCHITECTURE
    assert model_arch in MODEL_MAP.keys(), f"Unknown architecture: {model_arch}. Available: {list(MODEL_MAP.keys())}"

    # All models are nnUNet-based
    kwargs = {
        'in_channels': cfg.MODEL.IN_PLANES,
        'out_channels': cfg.MODEL.OUT_PLANES,
    }

    # Add specific parameters for different model types
    if 'unetr' in model_arch or 'swin' in model_arch:
        kwargs['img_size'] = tuple(cfg.MODEL.INPUT_SIZE)

    if 'mednext' in model_arch:
        # MedNeXt specific parameters
        kwargs['kernel_size'] = getattr(cfg.MODEL, 'KERNEL_SIZE', 3)
        kwargs['deep_supervision'] = getattr(cfg.MODEL, 'DEEP_SUPERVISION', False)

    if 'unetr' in model_arch and 'swin' not in model_arch:
        # UNETR specific parameters
        kwargs['feature_size'] = getattr(cfg.MODEL, 'UNETR_FEATURE_SIZE', 16)
        kwargs['hidden_size'] = getattr(cfg.MODEL, 'HIDDEN_SIZE', 768)
        kwargs['mlp_dim'] = getattr(cfg.MODEL, 'MLP_DIM', 3072)
        kwargs['num_heads'] = getattr(cfg.MODEL, 'UNETR_NUM_HEADS', 12)
        kwargs['dropout_rate'] = getattr(cfg.MODEL, 'UNETR_DROPOUT_RATE', 0.0)

    if 'swin' in model_arch:
        # Swin UNETR specific parameters
        kwargs['feature_size'] = getattr(cfg.MODEL, 'SWIN_UNETR_FEATURE_SIZE', 48)
        kwargs['drop_rate'] = getattr(cfg.MODEL, 'SWIN_UNETR_DROPOUT_RATE', 0.0)
        kwargs['attn_drop_rate'] = getattr(cfg.MODEL, 'ATTN_DROP_RATE', 0.0)
        kwargs['dropout_path_rate'] = getattr(cfg.MODEL, 'DROPOUT_PATH_RATE', 0.0)
        kwargs['use_checkpoint'] = getattr(cfg.MODEL, 'USE_CHECKPOINT', False)

    if 'monai_unet3d' in model_arch and 'basic' not in model_arch:
        # MONAI UNet3D specific parameters
        kwargs['features'] = getattr(cfg.MODEL, 'FILTERS', (32, 64, 128, 256, 512))
        kwargs['strides'] = getattr(cfg.MODEL, 'STRIDES', (1, 2, 2, 2))
        kwargs['kernel_size'] = getattr(cfg.MODEL, 'KERNEL_SIZE', 3)
        kwargs['num_res_units'] = getattr(cfg.MODEL, 'NUM_RES_UNITS', 2)

    if 'basic' in model_arch:
        # Basic UNet specific parameters
        kwargs['features'] = getattr(cfg.MODEL, 'FILTERS', (32, 64, 128, 256))
        kwargs['dropout'] = getattr(cfg.MODEL, 'DROPOUT', 0.1)

    # Create model
    model = MODEL_MAP[model_arch](**kwargs)

    print('model: ', model.__class__.__name__)
    return make_parallel(model, cfg, device, rank)


def make_parallel(model, cfg, device, rank=None, find_unused_parameters=False):
    if cfg.SYSTEM.PARALLEL == 'DDP':
        print('Parallelism with DistributedDataParallel.')
        # Currently SyncBatchNorm only supports DistributedDataParallel (DDP)
        # with single GPU per process. Use torch.nn.SyncBatchNorm.convert_sync_batchnorm()
        # to convert BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
        if cfg.MODEL.NORM_MODE == "sync_bn":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = model.to(device)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        assert rank is not None
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)

    elif cfg.SYSTEM.PARALLEL == 'DP':
        gpu_device_ids = list(range(cfg.SYSTEM.NUM_GPUS))
        print('Parallelism with DataParallel.')
        model = nn.DataParallel(model, device_ids=gpu_device_ids)
        model = model.to(device)

    else:
        print('No parallelism across multiple GPUs.')
        model = model.to(device)

    return model.to(device)


def update_state_dict(cfg, model_dict, mode='train'):
    """Modern state dict processing for nnUNet models."""
    if 'n_averaged' in model_dict.keys():
        print("Number of models averaged in SWA: ", model_dict['n_averaged'])

    return model_dict
