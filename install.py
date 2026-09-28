#!/usr/bin/env python3
"""
PyTorch Connectomics Installation Script

Automatically detects CUDA version and installs PyTorch with matching support.
Features:
- Auto-detects CUDA (nvidia-smi, nvcc, module system, /usr/local)
- Detects Apple silicon, whose standard PyTorch wheel includes MPS support
- Detects and uses current conda environment (smart installation)
- Installs pre-built packages via conda (avoids GCC issues)
- Verifies installation

Usage:
    python install.py                           # Interactive mode
    conda activate my_env && python install.py  # Use current environment
    python install.py --env-name my_env --python 3.11
    python install.py --cuda 12.6
    python install.py --cpu-only
    python install.py --yes                     # CI mode (no prompts)
"""

import os
import platform
import sys
import subprocess
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def disable(cls):
        """Disable colors for non-interactive terminals."""
        cls.HEADER = ""
        cls.OKBLUE = ""
        cls.OKCYAN = ""
        cls.OKGREEN = ""
        cls.WARNING = ""
        cls.FAIL = ""
        cls.ENDC = ""
        cls.BOLD = ""
        cls.UNDERLINE = ""


def run_command(cmd: str, check: bool = True, capture: bool = True) -> Tuple[int, str, str]:
    """Run shell command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=capture, text=True)
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout if e.stdout else "", e.stderr if e.stderr else ""


def run_streamed(cmd: str) -> int:
    """Run a long step with its output shown live, so a multi-GB download does
    not look like a hang. Returns the exit code; errors are already on screen."""
    print_info(f"$ {cmd}")
    sys.stdout.flush()
    return subprocess.run(cmd, shell=True).returncode


def conda_run(env_name: str, cmd: str) -> str:
    """`conda run` prefix that streams output instead of buffering it."""
    return f"conda run --no-capture-output -n {env_name} {cmd}"


def print_header(text: str):
    """Print styled header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}[WARN] {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}[FAIL] {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}[INFO] {text}{Colors.ENDC}")


def check_package_installed(package_name: str, env_name: str) -> tuple[bool, Optional[str]]:
    """
    Check if a package is already installed in the conda environment.

    Args:
        package_name: Name of the package to check
        env_name: Name of the conda environment

    Returns:
        Tuple of (is_installed, version) where version is None if not installed
    """
    code, stdout, _ = run_command(f"conda list -n {env_name} {package_name}", check=False)
    if code == 0 and stdout:
        # Check if package name appears in the output (not just empty list)
        lines = stdout.strip().split("\n")
        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.split()
            if parts and parts[0] == package_name:
                # Return True and version (second column)
                version = parts[1] if len(parts) > 1 else "unknown"
                return True, version
    return False, None


def check_pip_package_installed(package_name: str, env_name: str) -> tuple[bool, Optional[str]]:
    """Like check_package_installed, but asks pip (sees pip-only wheels)."""
    code, stdout, _ = run_command(f"conda run -n {env_name} pip show {package_name}", check=False)
    if code == 0:
        match = re.search(r"^Version:\s*(\S+)", stdout, re.MULTILINE)
        return True, match.group(1) if match else "unknown"
    return False, None


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
    for detector in [
        detect_cuda_nvidia_smi,
        detect_cuda_nvcc,
        detect_cuda_module,
        detect_cuda_local,
    ]:
        version = detector()
        if version:
            return version

    return None


# PyTorch wheel indices that still receive current releases, newest first, as
# ((min driver CUDA major, minor), tag). Verified 2026-09 with
# `pip index versions torch --index-url https://download.pytorch.org/whl/<tag>`:
# cu130/cu126 serve torch 2.14, cu128 stops at 2.11, cu118 at 2.7.1. cu124 and
# cu121 are frozen at 2.6/2.5 and are deliberately absent. Revisit this table
# when a PyTorch release adds or drops an index.
PYTORCH_CUDA_WHEELS = [
    ((13, 0), "cu130"),
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((11, 8), "cu118"),
]
# Blackwell (compute capability 10.x/12.x) kernels first ship in cu128 wheels.
MIN_WHEEL_FOR_BLACKWELL = "cu128"
# The default PyPI torch wheel on Linux bundles CUDA (~3 GB); the cpu index does not.
CPU_TORCH_INSTALL = (
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    if sys.platform.startswith("linux")
    else "pip install torch torchvision"
)


def cuda_to_pytorch(cuda_version: str) -> Optional[str]:
    """Map the driver's CUDA version to the newest compatible PyTorch wheel tag.

    ``nvidia-smi`` reports the highest CUDA runtime the driver supports, and
    PyTorch wheels bundle their own runtime, so the rule is: pick the newest
    wheel whose CUDA version does not exceed the driver's. Returns None when the
    driver is older than every maintained wheel.
    """
    try:
        major, minor = map(int, cuda_version.split(".")[:2])
    except (ValueError, AttributeError):
        return None
    for min_version, tag in PYTORCH_CUDA_WHEELS:
        if (major, minor) >= min_version:
            return tag
    return None


def detect_gpu_compute_capability() -> Optional[Tuple[int, int]]:
    """Return the highest compute capability among visible NVIDIA GPUs."""
    code, stdout, _ = run_command(
        "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null",
        check=False,
    )
    if code != 0:
        return None
    caps = []
    for line in stdout.strip().splitlines():
        match = re.search(r"(\d+)\.(\d+)\s*$", line)
        if match:
            caps.append((int(match.group(1)), int(match.group(2))))
    return max(caps) if caps else None


def wheel_tag_value(tag: str) -> int:
    """cu128 -> 128, for ordering wheel tags."""
    return int(tag[2:])


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer."""
    choices = "Y/n" if default else "y/N"
    while True:
        response = input(f"{question} [{choices}]: ").strip().lower()
        if not response:
            return default
        if response in ["y", "yes"]:
            return True
        if response in ["n", "no"]:
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
        # Compare whole names: a prefix match would treat "pytc" as existing
        # when only "pytc2" does.
        return any(
            line.split()[0] == env_name
            for line in stdout.split("\n")
            if line.strip() and not line.startswith("#")
        )
    return False


def get_current_conda_env() -> Optional[str]:
    """Get the name of the currently active conda environment."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env and conda_env != "base":
        return conda_env
    return None


def install_pytorch_connectomics(
    env_name: str = "pytc",
    python_version: str = "3.11",
    cuda_version: Optional[str] = None,
    cpu_only: bool = False,
    skip_prompts: bool = False,
    pip_options: str = "",
    install_type: str = "basic",
    force_recreate: bool = False,
    torch_index_url: Optional[str] = None,
) -> bool:
    """Main installation function."""

    # Check conda
    if not check_conda():
        print_error("conda not found. Please install Miniconda or Anaconda first.")
        return False

    print_success("conda found")

    # Validate Python version
    py_major, py_minor = map(int, python_version.split(".")[:2])
    if py_major != 3 or py_minor < 8 or py_minor >= 13:
        print_error(f"Python {python_version} is not supported")
        print_error("Supported versions: 3.8 to 3.12")
        return False
    else:
        print_info(f"Target Python: {python_version} (recommended: 3.11)")

    # Check if already in a conda environment
    current_env = get_current_conda_env()
    use_current_env = False

    if current_env:
        print_info(f"Detected active conda environment: {Colors.BOLD}{current_env}{Colors.ENDC}")
        if force_recreate and current_env == env_name:
            # Refuse the dangerous combination: removing an env from inside it
            # leaves conda in an inconsistent state on most platforms.
            print_error(
                f"--force-recreate cannot wipe the currently-active env '{env_name}'. "
                "Run `conda deactivate` first, then rerun this command."
            )
            return False
        if not skip_prompts:
            use_current_env = prompt_yes_no(
                f"Install in current environment '{current_env}' instead of creating '{env_name}'?",
                default=True,
            )
            if use_current_env:
                env_name = current_env
                print_success(f"Will use current environment: {env_name}")
        else:
            # In CI mode, use current env if it matches the target env name
            if current_env == env_name:
                use_current_env = True
                print_info(f"Using current environment: {env_name}")

    # Handle CUDA / Apple MPS
    apple_mps_host = sys.platform == "darwin" and platform.machine() == "arm64"
    accelerator_label = "CPU-only"
    pytorch_cuda = None
    if torch_index_url:
        print_info(f"Using explicit PyTorch index: {torch_index_url}")
        pytorch_install = f"pip install torch torchvision --index-url {torch_index_url}"
        accelerator_label = f"custom index ({torch_index_url.rstrip('/').split('/')[-1]})"
    elif cpu_only:
        print_warning("CPU-only installation requested")
        pytorch_install = CPU_TORCH_INSTALL
    else:
        if cuda_version:
            print_info(f"Using specified CUDA version: {cuda_version}")
        else:
            cuda_version = detect_cuda()
        if cuda_version:
            pytorch_cuda = cuda_to_pytorch(cuda_version)
            if pytorch_cuda is None:
                print_error(
                    f"CUDA {cuda_version} is older than every maintained PyTorch wheel "
                    f"(oldest: {PYTORCH_CUDA_WHEELS[-1][1]}). Update the NVIDIA driver, "
                    "or pass --cpu-only."
                )
                return False
            capability = detect_gpu_compute_capability()
            if capability:
                print_info(f"GPU compute capability: {capability[0]}.{capability[1]}")
            if (
                capability
                and capability[0] >= 10
                and wheel_tag_value(pytorch_cuda) < wheel_tag_value(MIN_WHEEL_FOR_BLACKWELL)
            ):
                print_error(
                    f"This GPU (compute {capability[0]}.{capability[1]}, Blackwell) needs "
                    f"{MIN_WHEEL_FOR_BLACKWELL} or newer wheels, but CUDA {cuda_version} "
                    f"only allows {pytorch_cuda}. Update the NVIDIA driver to one that "
                    "reports CUDA >= 12.8 in nvidia-smi."
                )
                return False
            pytorch_install = (
                "pip install torch torchvision --index-url "
                f"https://download.pytorch.org/whl/{pytorch_cuda}"
            )
            accelerator_label = f"CUDA {cuda_version} (driver)"
        else:
            print_warning("Could not auto-detect CUDA version")
            if apple_mps_host:
                accelerator_label = "Apple MPS"
                print_info("Apple silicon detected; the standard PyTorch wheel includes MPS")
                pytorch_install = "pip install torch torchvision"
            elif skip_prompts:
                print_info("Using CPU-only installation")
                pytorch_install = CPU_TORCH_INSTALL
            else:
                print("\nOptions:")
                print("  1. Install CPU-only PyTorch (slower, no GPU)")
                print("  2. Manually specify CUDA version")
                print("  3. Exit")

                choice = input("\nChoose option (1/2/3): ").strip()
                if choice == "1":
                    pytorch_install = CPU_TORCH_INSTALL
                elif choice == "2":
                    cuda_version = input("Enter driver CUDA version (e.g., 12.6, 13.0): ").strip()
                    pytorch_cuda = cuda_to_pytorch(cuda_version)
                    if pytorch_cuda is None:
                        print_error(f"No maintained PyTorch wheel for CUDA {cuda_version}")
                        return False
                    pytorch_install = (
                        "pip install torch torchvision --index-url "
                        f"https://download.pytorch.org/whl/{pytorch_cuda}"
                    )
                    accelerator_label = f"CUDA {cuda_version}"
                else:
                    print_info("Installation cancelled")
                    return False

    # Print installation plan
    print_header("Installation Plan")
    print(f"  Environment: {Colors.BOLD}{env_name}{Colors.ENDC}")
    print(f"  Python: {Colors.BOLD}{python_version}{Colors.ENDC}")
    print(f"  Accelerator: {Colors.BOLD}{accelerator_label}{Colors.ENDC}")
    if pytorch_cuda:
        print(f"  PyTorch: {Colors.BOLD}{pytorch_cuda} wheel{Colors.ENDC}")
    elif torch_index_url:
        print(f"  PyTorch: {Colors.BOLD}{torch_index_url}{Colors.ENDC}")
    elif accelerator_label == "Apple MPS":
        print(f"  PyTorch: {Colors.BOLD}macOS wheel with MPS support{Colors.ENDC}")
    else:
        print(f"  PyTorch: {Colors.BOLD}CPU-only{Colors.ENDC}")

    if not skip_prompts and not prompt_yes_no("\nContinue with installation?"):
        print_info("Installation cancelled")
        return False

    # Create or use existing environment
    create_env_cmd = (
        f"conda create -n {env_name} --override-channels -c conda-forge "
        f"python={python_version} -y"
    )
    if use_current_env:
        # Skip environment creation, use current one
        print_header("Step 1/6: Using Existing Environment")
        print_success(f"Using environment: {env_name}")
    elif env_exists(env_name):
        # Default to reusing the existing env. Recreate only when the user
        # explicitly opts in via --force-recreate, or interactively confirms.
        print_warning(f"Environment '{env_name}' already exists")
        recreate = False
        if force_recreate:
            recreate = True
            print_info("--force-recreate set; will remove and recreate")
        elif not skip_prompts:
            recreate = prompt_yes_no("Remove and recreate?", default=False)

        if recreate:
            print_info(f"Removing existing environment '{env_name}'...")
            code, _, _ = run_command(f"conda env remove -n {env_name} -y", check=False)
            if code != 0:
                print_error(f"Failed to remove environment '{env_name}'")
                return False

            print_header("Step 1/6: Creating Conda Environment")
            print_info(f"Creating environment '{env_name}' with Python {python_version}...")
            if run_streamed(create_env_cmd) != 0:
                print_error("Failed to create conda environment (see output above)")
                return False
            print_success(f"Environment '{env_name}' created")
        else:
            print_header("Step 1/6: Reusing Existing Environment")
            print_info("Reusing existing env. Pass --force-recreate to wipe and recreate.")
            print_success(f"Using environment: {env_name}")
    else:
        # Create environment
        print_header("Step 1/6: Creating Conda Environment")
        print_info(f"Creating environment '{env_name}' with Python {python_version}...")
        if run_streamed(create_env_cmd) != 0:
            print_error("Failed to create conda environment (see output above)")
            return False
        print_success(f"Environment '{env_name}' created")

    # Get conda base for activation
    conda_base = get_conda_base()
    if not conda_base:
        print_error("Could not determine conda base directory")
        return False

    # Install scientific packages via conda (pre-built binaries, no compilation)
    print_header("Step 2/6: Installing Scientific Packages")
    print_info("Installing pre-built packages via conda-forge...")
    print_info("This step is CRITICAL to avoid compilation errors...")

    # Install in two groups for better reliability
    # Group 1: Core numerical packages + cc3d (MUST succeed together for compatibility)
    # CRITICAL: Install cc3d (connected-components-3d) with numpy/h5py/cython to avoid
    # building from source with wrong numpy version
    core_packages = ["numpy", "h5py", "cython", "connected-components-3d"]

    # Check which packages are already installed
    already_installed = []
    to_install = []

    print_info("Checking which packages are already installed...")
    for pkg in core_packages:
        is_installed, version = check_package_installed(pkg, env_name)
        if is_installed:
            already_installed.append(f"{pkg} ({version})")
        else:
            to_install.append(pkg)

    if already_installed:
        print_success(f"Already installed: {', '.join(already_installed)}")

    if to_install:
        print_info(f"Installing: {', '.join(to_install)}")
        print_info("Note: Installing cc3d with numpy to ensure compatibility")
        code = run_streamed(
            f"conda install -n {env_name} --override-channels -c conda-forge "
            f"{' '.join(to_install)} -y"
        )
        if code != 0:
            print_error("Failed to install core packages via conda (see output above)!")
            print_error("\nThis is a critical error. These packages MUST be installed via conda")
            print_error("to avoid GCC compilation errors.")
            return False
        print_success(f"Core packages installed: {', '.join(to_install)}")
    else:
        print_success("All core packages already installed")

    if apple_mps_host:
        # The PyTorch macOS wheel bundles libomp. Conda-forge's OpenMP-flavoured
        # OpenBLAS loads a second copy and aborts even on ``import torch``.
        print_info("Selecting pthreads OpenBLAS to avoid duplicate libomp on Apple silicon...")
        code, _, stderr = run_command(
            f"conda install -n {env_name} --override-channels -c conda-forge "
            "'libopenblas=*=*pthreads*' -y",
            check=False,
        )
        if code != 0:
            print_error(f"Failed to select Apple-compatible OpenBLAS: {stderr}")
            return False
        print_success("Apple-compatible pthreads OpenBLAS installed")

    # cc3d ABI probe runs after `pip install -e .` (the editable install can pull
    # a different numpy as a transitive dep, which is the actual ABI-break case).

    # Install PyTorch
    print_header("Step 3/6: Installing PyTorch")

    print_info("Downloads ~1-3 GB for CUDA wheels; output is shown live.")
    if run_streamed(conda_run(env_name, pytorch_install)) != 0:
        print_error("Failed to install PyTorch (see pip output above)")
        return False
    print_success("PyTorch installed")

    # Install PyTorch Connectomics
    print_header("Step 4/6: Installing PyTorch Connectomics")
    print_info(f"Installing package in editable mode ({install_type} installation)...")

    # Build pip install command with appropriate extras
    pip_cmd = conda_run(env_name, "pip install -e .")
    if install_type != "basic":
        pip_cmd += f"[{install_type}]"
    if pip_options:
        pip_cmd += f" {pip_options}"

    # First try without --no-build-isolation to ensure dependencies are installed
    print_info("Installing with full dependency resolution...")
    if run_streamed(pip_cmd) != 0:
        print_warning("Standard installation failed, trying with --no-build-isolation...")
        if run_streamed(f"{pip_cmd} --no-build-isolation") != 0:
            print_error("Failed to install PyTorch Connectomics (see pip output above)")
            return False
    print_success("PyTorch Connectomics installed")

    # Envs created before the switch to opencv-python-headless still carry the
    # GUI build, which needs libGL and fails to import on headless servers. Both
    # wheels write the same cv2/ directory, so reinstall headless after removal.
    has_gui_cv2, _ = check_pip_package_installed("opencv-python", env_name)
    if has_gui_cv2:
        print_info("Replacing opencv-python with opencv-python-headless (no libGL needed)...")
        run_command(conda_run(env_name, "pip uninstall -y opencv-python"), check=False)
        if (
            run_streamed(
                conda_run(
                    env_name, "pip install --force-reinstall --no-deps opencv-python-headless"
                )
            )
            != 0
        ):
            print_warning("Could not reinstall opencv-python-headless; `import cv2` may fail")

    # cc3d ABI probe: only repair on actual import failure (numpy ABI mismatch).
    # Runs after `pip install -e .` so the probe sees pip's final numpy choice.
    print_info("Verifying cc3d ABI compatibility against installed numpy...")
    code, _, _ = run_command(f'conda run -n {env_name} python -c "import cc3d"', check=False)
    if code != 0:
        print_warning("cc3d import failed; reinstalling against current numpy ABI...")
        run_command(
            f"conda run -n {env_name} pip uninstall -y connected-components-3d", check=False
        )
        code, _, stderr = run_command(
            f"conda run -n {env_name} pip install --no-cache-dir connected-components-3d",
            check=False,
        )
        if code != 0:
            print_warning(
                "cc3d reinstall failed. See INSTALLATION.md "
                "'ABI mismatch on import cc3d' for the manual workaround."
            )
            if stderr.strip():
                print_warning(stderr.strip())
        else:
            print_success("cc3d reinstalled against current numpy")
    else:
        print_success("cc3d ABI is consistent with installed numpy")

    # Install just (command runner used by README tutorial commands)
    print_header("Step 5/6: Installing Command Runner (just)")
    print_info("Installing just command runner via conda...")

    # Check if just is already installed
    is_installed, version = check_package_installed("just", env_name)
    if is_installed:
        print_success(f"just already installed: {version}")
    else:
        code, _, stderr = run_command(
            f"conda install -n {env_name} --override-channels -c conda-forge just -y", check=False
        )
        if code != 0:
            print_warning("Failed to install just via conda")
            print_info("You can install just manually:")
            print_info("  - Rust: cargo install just")
            print_info("  - Homebrew: brew install just")
            print_info("  - Ubuntu/Debian: apt install just")
            print_info("  - Arch: pacman -S just")
        else:
            print_success("just installed successfully")

    # Verify installation
    print_header("Step 6/6: Verifying Installation")
    # Imports every core dependency and, when a GPU is visible, checks the torch
    # wheel ships kernels for it and launches one tiny kernel. A torch that
    # imports but cannot run on the GPU is a failed install, not a warning.
    check_script = Path(__file__).resolve().parent / "scripts" / "check_install.py"
    if run_streamed(conda_run(env_name, f'python "{check_script}"')) != 0:
        print_error("Installation finished but the environment check failed (see above).")
        print_error("Each [FAIL] line has a hint; details in INSTALLATION.md.")
        return False
    print_success("Installation verified")

    # Print usage instructions
    print_header("Installation Complete!")

    if use_current_env:
        print(f"{Colors.OKGREEN}You're already in the environment - ready to use!{Colors.ENDC}\n")
    else:
        print("To use PyTorch Connectomics:\n")
        print(f"  1. Activate the environment:")
        print(f"     {Colors.BOLD}conda activate {env_name}{Colors.ENDC}\n")

    step_num = 1 if use_current_env else 2

    if cuda_version and run_command("command -v module", check=False)[0] == 0:
        print(f"  {step_num}. Load CUDA module (if needed):")
        print(f"     {Colors.BOLD}module load cuda/{cuda_version}{Colors.ENDC}\n")
        step_num += 1

    print(f"  {step_num}. Run the 30-second demo (trains a tiny model on synthetic data):")
    print(f"     {Colors.BOLD}python scripts/main.py --demo{Colors.ENDC}\n")

    print(f"  {step_num + 1}. Re-check the env any time (e.g. after upgrading packages):")
    print(f"     {Colors.BOLD}python scripts/check_install.py{Colors.ENDC}\n")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Install PyTorch Connectomics with automatic CUDA detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                    # Auto-detect everything (basic, non-interactive)
  python install.py --env-name my_env         # Custom environment name
  python install.py --python 3.10             # Use Python 3.10
  python install.py --cuda 12.6               # Target a driver CUDA version
  python install.py --torch-index-url https://download.pytorch.org/whl/cu130
  python install.py --cpu-only                # CPU-only installation
  python install.py --interactive             # Enable interactive prompts
  python install.py --install-type dev        # Development installation with dev tools
  python install.py --install-type full        # Full installation with all features
  python install.py --pip-options "--no-deps" # Custom pip options
        """,
    )

    parser.add_argument("--env-name", default="pytc", help="Conda environment name (default: pytc)")
    parser.add_argument(
        "--python",
        default="3.11",
        help="Python version (default: 3.11)",
    )
    parser.add_argument(
        "--cuda",
        help="Driver CUDA version to target instead of auto-detecting (e.g. 12.6, 13.0)",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Install CPU-only PyTorch")
    parser.add_argument(
        "--torch-index-url",
        help="Install torch/torchvision from this index, bypassing CUDA detection "
        "(e.g. https://download.pytorch.org/whl/cu130)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enable interactive prompts (default: non-interactive)",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument(
        "--pip-options",
        default="",
        help='Additional options to pass to pip install -e . (e.g., "--no-deps --force-reinstall")',
    )
    parser.add_argument(
        "--install-type",
        choices=["basic", "dev", "full"],
        default="basic",
        help=(
            "Installation type: basic (core only), dev (with dev tools), "
            "full (all features) (default: basic)"
        ),
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="If the target conda env already exists, remove and recreate it. "
        "Default is to reuse an existing env.",
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
        skip_prompts=not args.interactive,
        pip_options=args.pip_options,
        install_type=args.install_type,
        force_recreate=args.force_recreate,
        torch_index_url=args.torch_index_url,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
