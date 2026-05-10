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
#   python scripts/main.py --demo

set -e

ENV_NAME=${1:-pytc}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; installing Miniconda to \$HOME/miniconda3..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    # Put Miniconda's python and conda on PATH. `conda shell.bash hook` only
    # exports condabin, not bin, so we'd otherwise have no python to run install.py.
    export PATH="$HOME/miniconda3/bin:$PATH"
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
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
echo "  python scripts/main.py --demo"
