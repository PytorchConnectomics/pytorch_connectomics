"""
Slurm job launcher with auto-resubmission support.

Reference: BANIS slurm_job_scheduler.py
"""

import argparse
import os
import itertools
import subprocess
from pathlib import Path
from typing import Dict, List
import yaml


def load_sweep_config(config_path: str) -> Dict:
    """Load parameter sweep configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_job_configs(sweep_config: Dict) -> List[Dict]:
    """Generate all combinations of parameters."""
    params = sweep_config['params']

    # Get all parameter combinations
    keys = list(params.keys())
    values = [params[k] for k in keys]

    jobs = []
    for combination in itertools.product(*values):
        job_config = dict(zip(keys, combination))
        jobs.append(job_config)

    return jobs


def create_slurm_script(
    job_config: Dict,
    template_path: str,
    output_dir: Path,
    exp_name: str
) -> Path:
    """Create Slurm batch script from template."""
    with open(template_path, 'r') as f:
        template = f.read()

    # Format script with job parameters
    script = template.format(
        exp_name=exp_name,
        output_dir=output_dir,
        **job_config
    )

    # Write script
    script_path = output_dir / f"job_{exp_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script)

    return script_path


def submit_job(script_path: Path, dry_run: bool = False) -> int:
    """Submit Slurm job."""
    cmd = f"sbatch {script_path}"

    if dry_run:
        print(f"[DRY RUN] Would execute: {cmd}")
        return -1

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        # Extract job ID
        job_id = int(result.stdout.strip().split()[-1])
        print(f"✓ Submitted job {job_id}: {script_path.name}")
        return job_id
    else:
        print(f"✗ Failed to submit: {script_path.name}")
        print(f"  Error: {result.stderr}")
        return -1


def main():
    parser = argparse.ArgumentParser(description="Launch Slurm parameter sweep")
    parser.add_argument("--config", type=str, required=True, help="Sweep config YAML")
    parser.add_argument("--template", type=str, default="scripts/slurm_template.sh", help="Slurm script template")
    parser.add_argument("--output-dir", type=str, default="slurm_jobs", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually submit jobs")
    parser.add_argument("--max-jobs", type=int, default=None, help="Maximum number of jobs to submit")
    args = parser.parse_args()

    # Load sweep configuration
    sweep_config = load_sweep_config(args.config)
    job_configs = generate_job_configs(sweep_config)

    print(f"Generated {len(job_configs)} job configurations")

    # Limit if requested
    if args.max_jobs is not None:
        job_configs = job_configs[:args.max_jobs]
        print(f"Limiting to {len(job_configs)} jobs")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    job_ids = []
    for i, job_config in enumerate(job_configs):
        exp_name = f"exp_{i:04d}"

        # Create script
        script_path = create_slurm_script(
            job_config,
            args.template,
            output_dir,
            exp_name
        )

        # Submit
        job_id = submit_job(script_path, dry_run=args.dry_run)
        if job_id > 0:
            job_ids.append(job_id)

    print(f"\nSubmitted {len(job_ids)} jobs")
    if job_ids:
        print(f"Job IDs: {min(job_ids)} - {max(job_ids)}")


if __name__ == "__main__":
    main()
