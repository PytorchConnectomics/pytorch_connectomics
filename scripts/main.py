import os, sys
import argparse
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import numpy as np
from connectomics.config import *
from connectomics.engine import Trainer

def get_args():
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--inference', action='store_true', help='inference mode')
    parser.add_argument('--distributed', action='store_true', help='distributed training')
    parser.add_argument('--local_rank', type=int, help='node rank for distributed training', default=None)
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    # Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.local_rank==0 or args.local_rank is None:
        print("Command line arguments: ", args)

    # Set seeds
    manual_seed = 0 if args.local_rank is None else args.local_rank
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Set configurations
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Overwrite options given configs with higher priority.
    if args.inference:
        update_inference_cfg(cfg)
    overwrite_cfg(cfg, args)
    cfg.freeze()

    if args.local_rank==0 or args.local_rank is None:
        # In distributed training, only print and save the 
        # configurations using the node with local_rank=0.
        print(cfg)

        if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
            print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
            os.makedirs(cfg.DATASET.OUTPUT_PATH)
            save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    if args.distributed:
        assert torch.cuda.is_available(), \
            "Distributed training without GPUs is not supported!"
        dist.init_process_group("nccl", init_method='env://')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)        
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device: ", device)
    cudnn.enabled = True
    cudnn.benchmark = True

    mode = 'test' if args.inference else 'train'
    trainer = Trainer(cfg, device, mode, 
                      rank=args.local_rank,
                      checkpoint=args.checkpoint)

    # Start training or inference:
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        if args.inference:
            trainer.test()
        else:
            trainer.train()
    else:
        trainer.run_chunk(mode)

if __name__ == "__main__":
    main()
