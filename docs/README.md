# PyTorch Connectomics Documentation

This directory contains the Sphinx documentation for PyTorch Connectomics.

## Quick Start

### Prerequisites

```bash
# Activate your environment
conda activate pytc

# Install documentation dependencies
pip install -r requirements.txt

# Install PyTorch Sphinx theme
git clone https://github.com/pytorch/pytorch_sphinx_theme.git
cd pytorch_sphinx_theme
pip install -e .
cd ..
```

### Build Documentation

```bash
# Build HTML documentation
make html

# Clean build
make clean && make html

# View in browser
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```

## Updating Documentation

See **[UPDATING_DOCS.md](UPDATING_DOCS.md)** for comprehensive guide on:

- Documentation structure
- Updating existing pages
- Adding new pages
- Building and testing
- ReStructuredText syntax
- Troubleshooting
- Deployment to ReadTheDocs

## Quick Reference

### Build Commands

```bash
make html          # Build documentation
make clean         # Clean build artifacts
make linkcheck     # Check for broken links
```

### Watch Mode (Auto-rebuild)

```bash
pip install sphinx-autobuild
sphinx-autobuild source build/html
# Open http://127.0.0.1:8000 in browser
```

### Serve Locally

```bash
cd build/html
python -m http.server 8000
# Open http://localhost:8000 in browser
```

## Structure

```
docs/
├── README.md              # This file
├── UPDATING_DOCS.md       # Comprehensive update guide
├── Makefile              # Build commands
├── requirements.txt      # Documentation dependencies
└── source/               # Source files
    ├── conf.py           # Sphinx configuration
    ├── index.rst         # Main page
    ├── notes/            # Getting started
    ├── tutorials/        # Tutorials
    ├── modules/          # API reference
    └── _static/          # Static files
```

## What's New in v2.0 Docs

The documentation has been updated to reflect v2.0 changes:

- ✅ Updated installation guide with Lightning/MONAI/Hydra
- ✅ New dependency information
- ✅ MedNeXt integration documentation
- ✅ Updated quick start examples
- ✅ Migration guide from v1.0
- 🔄 Configuration guide (needs Hydra examples)
- 🔄 Tutorials (need script updates)
- 🔄 API reference (needs Lightning modules)

See [UPDATING_DOCS.md](UPDATING_DOCS.md) for detailed update checklist.

## Contributing

When updating documentation:

1. Make changes in `source/` directory
2. Build locally: `make html`
3. Check for warnings: `make html 2>&1 | grep -i warning`
4. Test in browser
5. Commit and push

Documentation is automatically deployed to ReadTheDocs on push to GitHub.

## Getting Help

- **Sphinx**: https://www.sphinx-doc.org/
- **RST**: https://docutils.sourceforge.io/rst.html
- **PyTorch Theme**: https://github.com/pytorch/pytorch_sphinx_theme
- **ReadTheDocs**: https://docs.readthedocs.io/

## Links

- **Live Docs**: https://connectomics.readthedocs.io
- **GitHub**: https://github.com/zudi-lin/pytorch_connectomics
- **Paper**: https://arxiv.org/abs/2112.05754
