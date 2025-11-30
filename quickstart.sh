#!/bin/bash
# PyTorch Connectomics Quick Start Installation
# Works on most systems with CUDA 11+ or CPU-only
#
# Usage:
#   bash quickstart.sh [env_name]
#   curl -fsSL https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/refs/heads/master/quickstart.sh | bash -s -- myenv

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Capture optional env name argument (default: pytc)
ENV_NAME=${1:-pytc}

# Main installation
main() {
    print_header "🚀 PyTorch Connectomics Quick Start"

    # Check if conda exists
    if ! command -v conda &> /dev/null; then
        print_warning "conda not found. Installing Miniconda..."

        # Download Miniconda
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        MINICONDA_INSTALLER="/tmp/miniconda.sh"

        curl -fsSL "$MINICONDA_URL" -o "$MINICONDA_INSTALLER"
        bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
        rm "$MINICONDA_INSTALLER"

        # Add to PATH
        export PATH="$HOME/miniconda3/bin:$PATH"

        # Initialize conda for bash
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

        print_success "Miniconda installed"
    else
        print_success "conda found"
    fi

    # Clone repo if not already in it
    if [ ! -f "setup.py" ]; then
        print_info "Cloning PyTorch Connectomics repository..."
        git clone https://github.com/zudi-lin/pytorch_connectomics.git
        cd pytorch_connectomics
        print_success "Repository cloned"
    else
        print_success "Already in PyTorch Connectomics directory"
    fi

    # Run automated installer (non-interactive, basic mode)
    print_info "Running automated installation (Python 3.11) into env '${ENV_NAME}'..."
    python install.py --install-type basic --python 3.11 --env-name "${ENV_NAME}"

    # Print completion message
    print_header "✅ Installation Complete!"

    echo -e "${GREEN}${BOLD}PyTorch Connectomics is ready to use!${NC}\n"

    echo -e "${BOLD}Next steps:${NC}"
    echo -e "  1. Activate the environment:"
    echo -e "     ${BOLD}conda activate ${ENV_NAME}${NC}\n"

    echo -e "  2. Run a quick demo (30 seconds):"
    echo -e "     ${BOLD}python scripts/main.py --demo${NC}\n"

    echo -e "  3. Try a tutorial (mitochondria segmentation):"
    echo -e "     ${BOLD}python scripts/main.py --config tutorials/lucchi.yaml --fast-dev-run${NC}\n"

    echo -e "${BOLD}Need help?${NC}"
    echo -e "  📚 Documentation: https://connectomics.readthedocs.io"
    echo -e "  💬 Slack: https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w"
    echo -e "  🐛 Issues: https://github.com/zudi-lin/pytorch_connectomics/issues\n"
}

# Run main function
main
