#!/usr/bin/env python3
"""
PyTorch Connectomics Installation Script

Automatically detects CUDA version and installs PyTorch with matching support.
Supports: nvidia-smi, nvcc, module system, and /usr/local detection.

Usage:
    python install.py
    python install.py --env-name my_env --python 3.10
    python install.py --cuda 12.4
    python install.py --cpu-only
"""

import os
import sys
import subprocess
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def disable(cls):
        """Disable colors for non-interactive terminals."""
        cls.HEADER = ''
        cls.OKBLUE = ''
        cls.OKCYAN = ''
        cls.OKGREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''


def run_command(cmd: str, check: bool = True, capture: bool = True) -> Tuple[int, str, str]:
    """Run shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=capture,
            text=True
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout if e.stdout else "", e.stderr if e.stderr else ""


def print_header(text: str):
    """Print styled header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def check_conda() -> bool:
    """Check if conda is available."""
    code, _, _ = run_command("conda --version", check=False)
    return code == 0


def detect_cuda_nvidia_smi() -> Optional[str]:
    """Detect CUDA version via nvidia-smi."""
    code, stdout, _ = run_command("nvidia-smi 2>/dev/null", check=False)
    if code == 0:
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", stdout)
        if match:
            version = match.group(1)
            print_info(f"CUDA detected via nvidia-smi: {version}")
            return version
    return None


def detect_cuda_nvcc() -> Optional[str]:
    """Detect CUDA version via nvcc."""
    code, stdout, _ = run_command("nvcc --version 2>/dev/null", check=False)
    if code == 0:
        match = re.search(r"release\s+(\d+\.\d+)", stdout)
        if match:
            version = match.group(1)
            print_info(f"CUDA detected via nvcc: {version}")
            return version
    return None


def detect_cuda_module() -> Optional[str]:
    """Detect CUDA version via module system."""
    code, stdout, _ = run_command("module avail cuda 2>&1", check=False)
    if code == 0:
        match = re.search(r"cuda/(\d+\.\d+)", stdout)
        if match:
            version = match.group(1)
            print_info(f"CUDA found in module system: {version}")
            print_info("(Note: You may need to 'module load cuda' to use it)")
            return version
    return None


def detect_cuda_local() -> Optional[str]:
    """Detect CUDA version in /usr/local."""
    if Path("/usr/local").exists():
        for path in Path("/usr/local").glob("cuda-*"):
            if path.is_dir():
                match = re.search(r"cuda-(\d+\.\d+)", path.name)
                if match:
                    version = match.group(1)
                    print_info(f"CUDA found in /usr/local: {version}")
                    return version
    return None


def detect_cuda() -> Optional[str]:
    """Detect CUDA version using multiple methods."""
    print_info("Detecting CUDA installation...")

    # Try all detection methods
    for detector in [detect_cuda_nvidia_smi, detect_cuda_nvcc,
                     detect_cuda_module, detect_cuda_local]:
        version = detector()
        if version:
            return version

    return None


def cuda_to_pytorch(cuda_version: str) -> str:
    """Map CUDA version to PyTorch wheel version."""
    try:
        major, minor = map(int, cuda_version.split('.'))

        if major == 12:
            if minor >= 4:
                return "cu124"
            elif minor >= 1:
                return "cu121"
            else:
                return "cu118"
        elif major == 11:
            return "cu118"
        else:
            return "cu118"  # Default fallback
    except (ValueError, AttributeError):
        return "cu118"


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer."""
    choices = "Y/n" if default else "y/N"
    while True:
        response = input(f"{question} [{choices}]: ").strip().lower()
        if not response:
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print_warning("Please answer 'y' or 'n'")


def get_conda_base() -> str:
    """Get conda base directory."""
    code, stdout, _ = run_command("conda info --base", check=False)
    if code == 0:
        return stdout.strip()
    return ""


def env_exists(env_name: str) -> bool:
    """Check if conda environment exists."""
    code, stdout, _ = run_command("conda env list", check=False)
    if code == 0:
        return any(line.startswith(env_name) for line in stdout.split('\n'))
    return False


def install_pytorch_connectomics(
    env_name: str = "pytc",
    python_version: str = "3.11",
    cuda_version: Optional[str] = None,
    cpu_only: bool = False,
    skip_prompts: bool = False
) -> bool:
    """Main installation function."""

    # Check conda
    if not check_conda():
        print_error("conda not found. Please install Miniconda or Anaconda first.")
        return False

    print_success("conda found")

    # Handle CUDA
    if cpu_only:
        print_warning("CPU-only installation requested")
        pytorch_install = "pip install torch torchvision"
    elif cuda_version:
        print_info(f"Using specified CUDA version: {cuda_version}")
        pytorch_cuda = cuda_to_pytorch(cuda_version)
        pytorch_install = f"pip install torch torchvision --index-url https://download.pytorch.org/whl/{pytorch_cuda}"
    else:
        detected_cuda = detect_cuda()
        if detected_cuda:
            pytorch_cuda = cuda_to_pytorch(detected_cuda)
            pytorch_install = f"pip install torch torchvision --index-url https://download.pytorch.org/whl/{pytorch_cuda}"
            cuda_version = detected_cuda
        else:
            print_warning("Could not auto-detect CUDA version")
            if skip_prompts:
                print_info("Using CPU-only installation")
                pytorch_install = "pip install torch torchvision"
            else:
                print("\nOptions:")
                print("  1. Install CPU-only PyTorch (slower, no GPU)")
                print("  2. Manually specify CUDA version")
                print("  3. Exit")

                choice = input("\nChoose option (1/2/3): ").strip()
                if choice == "1":
                    pytorch_install = "pip install torch torchvision"
                elif choice == "2":
                    cuda_version = input("Enter CUDA version (e.g., 11.8, 12.1, 12.4): ").strip()
                    pytorch_cuda = cuda_to_pytorch(cuda_version)
                    pytorch_install = f"pip install torch torchvision --index-url https://download.pytorch.org/whl/{pytorch_cuda}"
                else:
                    print_info("Installation cancelled")
                    return False

    # Print installation plan
    print_header("Installation Plan")
    print(f"  Environment: {Colors.BOLD}{env_name}{Colors.ENDC}")
    print(f"  Python: {Colors.BOLD}{python_version}{Colors.ENDC}")
    print(f"  CUDA: {Colors.BOLD}{cuda_version or 'CPU-only'}{Colors.ENDC}")
    if cuda_version:
        print(f"  PyTorch: {Colors.BOLD}with {cuda_to_pytorch(cuda_version)} support{Colors.ENDC}")
    else:
        print(f"  PyTorch: {Colors.BOLD}CPU-only{Colors.ENDC}")

    if not skip_prompts and not prompt_yes_no("\nContinue with installation?"):
        print_info("Installation cancelled")
        return False

    # Check if environment exists
    if env_exists(env_name):
        print_warning(f"Environment '{env_name}' already exists")
        if not skip_prompts and not prompt_yes_no("Remove and recreate?"):
            print_info("Installation cancelled")
            return False

        print_info(f"Removing existing environment '{env_name}'...")
        code, _, _ = run_command(f"conda env remove -n {env_name} -y", check=False)
        if code != 0:
            print_error(f"Failed to remove environment '{env_name}'")
            return False

    # Create environment
    print_header("Step 1/5: Creating Conda Environment")
    print_info(f"Creating environment '{env_name}' with Python {python_version}...")
    code, _, stderr = run_command(f"conda create -n {env_name} python={python_version} -y", check=False)
    if code != 0:
        print_error(f"Failed to create conda environment: {stderr}")
        return False
    print_success(f"Environment '{env_name}' created")

    # Get conda base for activation
    conda_base = get_conda_base()
    if not conda_base:
        print_error("Could not determine conda base directory")
        return False

    # Install scientific packages via conda (pre-built binaries, no compilation)
    print_header("Step 2/5: Installing Scientific Packages")
    print_info("Installing pre-built packages via conda-forge...")
    scientific_packages = [
        'numpy<2.0',        # Pin to 1.x for GCC compatibility
        'scipy',
        'scikit-learn',
        'scikit-image',
        'h5py',
        'cython',
        'opencv',
    ]
    packages_str = ' '.join(scientific_packages)
    code, _, stderr = run_command(
        f"conda install -n {env_name} -c conda-forge {packages_str} -y",
        check=False
    )
    if code != 0:
        print_warning(f"Some conda packages failed to install: {stderr}")
        print_info("Will try to install missing packages via pip later...")
    else:
        print_success("Scientific packages installed via conda")

    # Install PyTorch
    print_header("Step 3/5: Installing PyTorch")
    print_info(f"Running: {pytorch_install}")

    # Use conda run to execute in the environment
    code, _, stderr = run_command(f"conda run -n {env_name} {pytorch_install}", check=False)
    if code != 0:
        print_error(f"Failed to install PyTorch: {stderr}")
        return False
    print_success("PyTorch installed")

    # Install PyTorch Connectomics
    print_header("Step 4/5: Installing PyTorch Connectomics")
    print_info("Installing package in editable mode (using pre-built dependencies)...")
    code, _, stderr = run_command(
        f"conda run -n {env_name} pip install -e . --no-build-isolation",
        check=False
    )
    if code != 0:
        print_warning("Installation with --no-build-isolation failed, retrying without it...")
        code, _, stderr = run_command(f"conda run -n {env_name} pip install -e .", check=False)
        if code != 0:
            print_error(f"Failed to install PyTorch Connectomics: {stderr}")
            return False
    print_success("PyTorch Connectomics installed")

    # Verify installation
    print_header("Step 5/5: Verifying Installation")
    code, stdout, _ = run_command(
        f"conda run -n {env_name} python -c \"import torch; "
        f"print(f'PyTorch: {{torch.__version__}}'); "
        f"print(f'CUDA available: {{torch.cuda.is_available()}}'); "
        f"print(f'CUDA version: {{torch.version.cuda if torch.cuda.is_available() else \\\"N/A\\\"}}')\""
        , check=False
    )
    if code == 0:
        print_success("Installation verified")
        print("\n" + stdout.strip())
    else:
        print_warning("Could not verify installation")

    # Print usage instructions
    print_header("Installation Complete!")
    print("To use PyTorch Connectomics:\n")
    print(f"  1. Activate the environment:")
    print(f"     {Colors.BOLD}conda activate {env_name}{Colors.ENDC}\n")

    if cuda_version and run_command("command -v module", check=False)[0] == 0:
        print(f"  2. Load CUDA module (if needed):")
        print(f"     {Colors.BOLD}module load cuda/{cuda_version}{Colors.ENDC}\n")

    print(f"  3. Run training:")
    print(f"     {Colors.BOLD}python scripts/main.py --config tutorials/lucchi.yaml{Colors.ENDC}\n")

    print(f"  4. Check available models:")
    print(f"     {Colors.BOLD}python -c 'from connectomics.models.arch import list_architectures; print(list_architectures())'{Colors.ENDC}\n")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Install PyTorch Connectomics with automatic CUDA detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                           # Auto-detect everything
  python install.py --env-name my_env         # Custom environment name
  python install.py --python 3.10             # Use Python 3.10
  python install.py --cuda 12.4               # Specify CUDA version
  python install.py --cpu-only                # CPU-only installation
  python install.py --yes                     # Skip prompts (CI mode)
        """
    )

    parser.add_argument(
        '--env-name',
        default='pytc',
        help='Conda environment name (default: pytc)'
    )
    parser.add_argument(
        '--python',
        default='3.11',
        help='Python version (default: 3.11)'
    )
    parser.add_argument(
        '--cuda',
        help='CUDA version (e.g., 11.8, 12.1, 12.4)'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Install CPU-only PyTorch'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip all prompts (for CI/automation)'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )

    args = parser.parse_args()

    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Print header
    print_header("PyTorch Connectomics Installation")

    # Run installation
    success = install_pytorch_connectomics(
        env_name=args.env_name,
        python_version=args.python,
        cuda_version=args.cuda,
        cpu_only=args.cpu_only,
        skip_prompts=args.yes
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
