import os
import argparse

import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

__all__ = [
    'get_args',
    'init_devices',
]


def get_args():
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)
    parser.add_argument('--inference', action='store_true',
                        help='inference mode')
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to load the checkpoint')
    parser.add_argument('--manual-seed', type=int, default=None)
    parser.add_argument('--local_world_size', type=int, default=1,
                        help='number of GPUs each process.')
    parser.add_argument('--local_rank', type=int, default=None,
                        help='node rank for distributed training')
    parser.add_argument('--debug', action='store_true',
                        help='run the scripts in debug mode')
    # Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_devices(args, cfg):
    if args.distributed:  # parameters to initialize the process group
        assert torch.cuda.is_available(), \
            "Distributed training without GPUs is not supported!"

        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK",
                        "LOCAL_RANK", "WORLD_SIZE")}
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(cfg.SYSTEM.DISTRIBUTED_BACKEND, init_method='env://')
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )

        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        n = torch.cuda.device_count() // args.local_world_size
        device_ids = list(
            range(args.local_rank * n, (args.local_rank + 1) * n))

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        print(
            f"[{os.getpid()}] rank = {dist.get_rank()} ({args.rank}), "
            + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
        )

        manual_seed = args.local_rank if args.manual_seed is None \
            else args.manual_seed
    else:
        manual_seed = 0 if args.manual_seed is None else args.manual_seed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("rank: {}, device: {}, seed: {}".format(args.local_rank, device, manual_seed))
    # use manual_seed seeds for reproducibility
    init_seed(manual_seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    return device
