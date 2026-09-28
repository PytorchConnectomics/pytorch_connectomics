#!/usr/bin/env bash
# Quick install for PyTorch Connectomics. See INSTALLATION.md for details.
#
# Usage:
#   bash quickstart.sh [env_name]
#   curl -fsSL https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/master/quickstart.sh | bash
#
# After this finishes:
#   cd pytorch_connectomics  (only if the script just cloned the repo)
#   conda activate <env_name>
#   python scripts/check_install.py
#   python scripts/main.py --demo
#
# Set CONDA_PREFIX_DIR to install Miniforge somewhere other than ~/miniforge3
# (useful on clusters with a small $HOME quota).

set -e

ENV_NAME=${1:-pytc}

if ! command -v conda >/dev/null 2>&1; then
    # Miniforge: conda-forge by default, no Anaconda Terms-of-Service prompt,
    # and installers for Linux/macOS on x86_64 and arm64.
    PREFIX="${CONDA_PREFIX_DIR:-$HOME/miniforge3}"
    INSTALLER="Miniforge3-$(uname)-$(uname -m).sh"
    echo "conda not found; installing Miniforge to $PREFIX (override with CONDA_PREFIX_DIR=...)"
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/$INSTALLER" \
        -o "/tmp/$INSTALLER"
    bash "/tmp/$INSTALLER" -b -p "$PREFIX"
    rm "/tmp/$INSTALLER"
    # Put conda's python on PATH. `conda shell.bash hook` only exports condabin,
    # not bin, so we'd otherwise have no python to run install.py.
    export PATH="$PREFIX/bin:$PATH"
    eval "$("$PREFIX/bin/conda" shell.bash hook)"
fi

# Clone only when this directory is clearly NOT a PyTC checkout.
# Require both install.py AND connectomics/__init__.py — neither alone is
# specific enough to be a safe signal.
if [ ! -f "install.py" ] || [ ! -f "connectomics/__init__.py" ]; then
    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
fi

python install.py --install-type basic --python 3.11 --env-name "$ENV_NAME"

echo
echo "Done. Next steps:"
echo "  cd $(basename "$PWD")    # only if you ran this from outside the repo"
echo "  conda activate $ENV_NAME"
echo "  python scripts/main.py --demo      # use srun/sbatch on a shared GPU cluster"
