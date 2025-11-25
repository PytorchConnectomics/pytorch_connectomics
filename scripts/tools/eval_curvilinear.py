#!/usr/bin/env python
"""
Evaluation script for curvilinear structures.

This script provides command-line interface for evaluating segmentation
of curvilinear structures (neurons, blood vessels) using skeleton-based metrics.

Metrics computed:
    - Correctness: Precision on skeletonized predictions
    - Completeness: Recall on skeletonized predictions
    - Quality: F-measure combining correctness and completeness
    - Foreground IoU: Intersection over Union of foreground regions

Based on:
    Mosinska et al., "Beyond the Pixel-Wise Loss for Topology-Aware Delineation"
    https://arxiv.org/abs/1712.02190

Usage:
    python scripts/tools/eval_curvilinear.py \
        --gt-path path/to/groundtruth/ \
        --pd-path path/to/predictions/ \
        --thres 128 \
        --max-index 200

    # Or use as module:
    from connectomics.metrics import evaluate_directory
    results = evaluate_directory(pred_dir, gt_dir)
"""

import argparse
from connectomics.metrics import evaluate_directory


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Curvilinear structure evaluation using skeleton metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate predictions in directory
  python scripts/tools/eval_curvilinear.py \
      --gt-path data/groundtruth/ \
      --pd-path results/predictions/ \
      --max-index 100

  # Custom threshold and pattern
  python scripts/tools/eval_curvilinear.py \
      --gt-path data/gt/ \
      --pd-path results/pred/ \
      --thres 150 \
      --pred-pattern "pred_%04d.png" \
      --gt-pattern "gt_%04d.png"
        """
    )

    parser.add_argument(
        '--gt-path',
        type=str,
        required=True,
        help='Path to directory containing ground truth masks'
    )
    parser.add_argument(
        '--pd-path',
        type=str,
        required=True,
        help='Path to directory containing predicted structures'
    )
    parser.add_argument(
        '--thres',
        type=int,
        default=128,
        help='Threshold for binarizing predictions [0, 255]. Default: 128'
    )
    parser.add_argument(
        '--max-index',
        type=int,
        default=200,
        help='Maximum image index to evaluate. Default: 200'
    )
    parser.add_argument(
        '--pred-pattern',
        type=str,
        default='%03d_pred.png',
        help='Filename pattern for predictions (use %%d for index). Default: %%03d_pred.png'
    )
    parser.add_argument(
        '--gt-pattern',
        type=str,
        default='%03d.png',
        help='Filename pattern for ground truth (use %%d for index). Default: %%03d.png'
    )
    parser.add_argument(
        '--dilation-size',
        type=int,
        default=5,
        help='Size of square structuring element for skeleton dilation. Default: 5'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: all CPUs)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-image output'
    )

    args = parser.parse_args()
    return args


def main():
    """Main evaluation function."""
    args = get_args()

    print("="*70)
    print("Curvilinear Structure Evaluation")
    print("="*70)
    print(f"Ground truth path:    {args.gt_path}")
    print(f"Prediction path:      {args.pd_path}")
    print(f"Prediction pattern:   {args.pred_pattern}")
    print(f"Ground truth pattern: {args.gt_pattern}")
    print(f"Threshold:            {args.thres}")
    print(f"Max index:            {args.max_index}")
    print(f"Dilation size:        {args.dilation_size}")
    print(f"Num workers:          {args.num_workers or 'auto'}")
    print("="*70)

    # Run evaluation
    results = evaluate_directory(
        pred_dir=args.pd_path,
        gt_dir=args.gt_path,
        pred_pattern=args.pred_pattern,
        gt_pattern=args.gt_pattern,
        max_index=args.max_index,
        threshold=args.thres,
        dilation_size=args.dilation_size,
        num_workers=args.num_workers,
        verbose=not args.quiet,
    )

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Images evaluated:     {results['num_evaluated']}")
    print(f"Mean IoU:             {results['mean_iou']:.6f}")
    print(f"Mean Correctness:     {results['mean_correctness']:.6f}")
    print(f"Mean Completeness:    {results['mean_completeness']:.6f}")
    print(f"Mean Quality:         {results['mean_quality']:.6f}")
    print("="*70)


if __name__ == "__main__":
    main()
