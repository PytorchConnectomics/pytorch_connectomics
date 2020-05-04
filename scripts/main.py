import os, sys
import argparse
import torch

from connectomics.config import get_cfg_defaults
from connectomics.engine import Trainer


def get_args():
    parser = argparse.ArgumentParser(description="Fast Training")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--output', type=str, help='output path')
    parser.add_argument('--inference', action='store_true', help='inference mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    args = parser.parse_args()
    return args

def main():
    # arguments
    args = get_args()
    print(args)

    # configurations
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print("Configuration details:")
    print(cfg)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    mode = 'test' if args.inference else 'train'
    trainer = Trainer(cfg, device, mode, args.output, args.checkpoint)
    if args.inference:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
  main()
