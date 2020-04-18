import os, sys
import argparse
import torch


from connectomics.config import get_cfg_defaults
from connectomics.run import trainer


def get_args():
    parser = argparse.ArgumentParser(description="Fast Training")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--output', type=str, help='output path')
    parser.add_argument('--inference', action='store_true', help='inference mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    args = parser.parse_args()
    return args

def main():
    # load configurations
    args = get_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    print(cfg)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    trainer = Trainer(cfg, device, args.output, inference=args.inference)
    if args.inference:
        trainer.test()
    else:
        trainer.train()

if __name__ == "__main__":
   main()
