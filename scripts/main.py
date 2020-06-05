import os, sys
import argparse
import torch

from connectomics.config import get_cfg_defaults, save_all_cfg, update_inference_cfg
from connectomics.engine import Trainer

def get_args():
    r"""Get args from command lines.
    """
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--inference', action='store_true', help='inference mode')
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
    r"""Main function.
    """
    # arguments
    args = get_args()
    print("Command line arguments:")
    print(args)

    # configurations
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.inference:
        update_inference_cfg(cfg)

    cfg.freeze()
    print("Configuration details:")
    print(cfg)

    if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
        print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
        os.makedirs(cfg.DATASET.OUTPUT_PATH)
        save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    mode = 'test' if args.inference else 'train'
    trainer = Trainer(cfg, device, mode, args.checkpoint)
    if args.inference:
        trainer.test()
    else:
        trainer.train()

if __name__ == "__main__":
    main()
