# Guide to Updating PyTorch Connectomics Documentation

This guide explains how to update and build the Sphinx-based documentation for PyTorch Connectomics v2.0.

## Documentation Structure

```
docs/
├── README.md                    # Basic build instructions
├── UPDATING_DOCS.md            # This file - comprehensive update guide
├── Makefile                    # Build commands
├── requirements.txt            # Documentation dependencies
├── environment_docs.yml        # Conda environment for docs
└── source/                     # Documentation source files
    ├── conf.py                 # Sphinx configuration
    ├── index.rst              # Main documentation page
    ├── notes/                  # Getting started guides
    │   ├── installation.rst   # Installation instructions
    │   ├── config.rst         # Configuration guide
    │   ├── dataloading.rst    # Data loading guide
    │   └── faq.rst            # FAQ
    ├── tutorials/              # Tutorial pages
    │   ├── neuron.rst         # Neuron segmentation
    │   ├── mito.rst           # Mitochondria segmentation
    │   ├── synapse.rst        # Synapse detection
    │   └── artifact.rst       # Artifact detection
    ├── modules/                # API reference (auto-generated)
    │   ├── data.rst           # Data module
    │   ├── engine.rst         # Engine module (legacy)
    │   ├── model.rst          # Model module
    │   └── utils.rst          # Utilities module
    ├── external/               # External tools
    │   └── neuroglancer.rst   # Neuroglancer integration
    ├── about/                  # About pages
    │   └── team.rst           # Team information
    ├── _static/                # Static files (CSS, images)
    └── _templates/             # HTML templates
```

## Prerequisites

### 1. Install Documentation Dependencies

```bash
# Activate your environment
conda activate pytc  # or your environment name

# Install documentation requirements
cd docs
pip install -r requirements.txt

# Install PyTorch Sphinx theme
git clone https://github.com/pytorch/pytorch_sphinx_theme.git
cd pytorch_sphinx_theme
pip install -e .
cd ..
```

### 2. Install PyTorch Connectomics

Make sure the package is installed in development mode:

```bash
cd ..  # Back to repo root
pip install -e .[full]
```

## Building Documentation

### Quick Build

```bash
cd docs
make html
```

The generated HTML files will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser.

### Clean Build

```bash
cd docs
make clean
make html
```

### Watch Mode (Auto-rebuild)

Install `sphinx-autobuild`:

```bash
pip install sphinx-autobuild
sphinx-autobuild source build/html
```

Then open http://127.0.0.1:8000 in your browser.

## What Needs Updating for v2.0

### Critical Updates Needed

1. **Installation Guide** (`source/notes/installation.rst`)
   - Update PyTorch version requirements (1.8+)
   - Add PyTorch Lightning and MONAI as core dependencies
   - Add OmegaConf/Hydra for configuration
   - Update installation command to use extras: `pip install -e .[full]`
   - Add MedNeXt optional installation

2. **Configuration Guide** (`source/notes/config.rst`)
   - Update from YACS to Hydra/OmegaConf examples
   - Show new YAML config structure
   - Explain dataclass-based configs
   - CLI override examples

3. **Main Index** (`source/index.rst`)
   - Add v2.0 announcement
   - Update feature list (Lightning, MONAI)
   - Add links to new architecture docs

4. **Tutorials** (`source/tutorials/*.rst`)
   - Update to use `scripts/main.py` instead of `scripts/build.py`
   - Update config file references
   - Add MedNeXt examples
   - Update training commands

5. **API Reference** (`source/modules/*.rst`)
   - Add new modules:
     - `connectomics.training.lightning` (LightningModule, LightningDataModule, Trainer)
     - `connectomics.config` (Hydra configs)
     - `connectomics.models.architectures` (Architecture registry)
   - Mark legacy modules (engine) as deprecated

### New Documentation Needed

1. **Lightning Integration Guide**
   - LightningModule usage
   - LightningDataModule usage
   - Trainer configuration
   - Callbacks and logging

2. **MONAI Integration Guide**
   - MONAI models overview
   - MONAI transforms
   - MONAI losses
   - MONAI metrics

3. **MedNeXt Guide**
   - Installation
   - Model sizes (S/B/M/L)
   - Configuration examples
   - Best practices

4. **Migration Guide**
   - v1.0 → v2.0 migration
   - YACS → Hydra conversion
   - Custom trainer → Lightning migration
   - Config file updates

## Updating Specific Files

### Updating RST Files

ReStructuredText (`.rst`) is the markup language used. Basic syntax:

```rst
Section Title
=============

Subsection
----------

Subsubsection
^^^^^^^^^^^^^

**Bold text**
*Italic text*
``code``

.. code-block:: python

    # Python code block
    import connectomics

.. code-block:: yaml

    # YAML code block
    model:
      architecture: mednext

.. note::
   This is a note box

.. warning::
   This is a warning box

.. tip::
   This is a tip box

Links:
`Link text <https://example.com>`_

Internal references:
:ref:`section-label`

API references:
:class:`connectomics.models.build.build_model`
:func:`connectomics.config.load_config`
:mod:`connectomics.training.lightning`
```

### Adding New Pages

1. Create the `.rst` file in appropriate directory:
   ```bash
   touch source/notes/lightning.rst
   ```

2. Add content to the file

3. Add to `index.rst` table of contents:
   ```rst
   .. toctree::
      :glob:
      :maxdepth: 1
      :caption: Get Started

      notes/installation
      notes/lightning      # New page
      notes/config
   ```

4. Rebuild docs:
   ```bash
   make html
   ```

### Auto-generating API Documentation

The API documentation is generated from docstrings. To update:

1. Ensure your code has proper docstrings:
   ```python
   def load_config(config_path: str) -> DictConfig:
       """Load configuration from YAML file.

       Args:
           config_path: Path to YAML configuration file.

       Returns:
           Loaded configuration as OmegaConf DictConfig.

       Example:
           >>> cfg = load_config("tutorials/lucchi.yaml")
           >>> print(cfg.model.architecture)
           'monai_basic_unet3d'
       """
       pass
   ```

2. Update module RST files to include new classes/functions:
   ```rst
   .. currentmodule:: connectomics.config

   .. autosummary::
       :toctree: generated
       :nosignatures:

       load_config
       save_config
       print_config
   ```

3. Rebuild:
   ```bash
   make clean
   make html
   ```

## Common Issues and Solutions

### Issue: Module not found during build

**Solution**: Make sure `connectomics` package is installed:
```bash
pip install -e .
```

### Issue: Theme not found

**Solution**: Install pytorch_sphinx_theme:
```bash
git clone https://github.com/pytorch/pytorch_sphinx_theme.git
cd pytorch_sphinx_theme
pip install -e .
```

### Issue: Broken cross-references

**Solution**: Check `nitpick_ignore` in `conf.py` and add missing references.

### Issue: Outdated auto-generated docs

**Solution**: Clean and rebuild:
```bash
make clean
rm -rf source/generated/
make html
```

## Testing Documentation

### Local Testing

1. Build docs:
   ```bash
   make html
   ```

2. Open in browser:
   ```bash
   # Linux
   xdg-open build/html/index.html

   # macOS
   open build/html/index.html

   # Windows
   start build/html/index.html
   ```

3. Check for:
   - Broken links
   - Missing images
   - Incorrect code examples
   - Formatting issues

### Checking for Warnings

```bash
make html 2>&1 | grep -i warning
```

Fix all warnings before deploying.

## Deployment to ReadTheDocs

The documentation is automatically built and deployed to ReadTheDocs when pushed to GitHub.

### Configuration

- `.readthedocs.yml` (if exists) - ReadTheDocs config
- `docs/requirements.txt` - Dependencies for ReadTheDocs build
- `docs/environment_docs.yml` - Conda environment

### Manual Trigger

1. Go to https://readthedocs.org/projects/connectomics/
2. Click "Build version"
3. Select "latest" or specific branch
4. Monitor build logs for errors

## Best Practices

### Writing Documentation

1. **Be concise**: Users scan documentation, they don't read every word
2. **Use examples**: Show code examples for every major feature
3. **Add cross-links**: Link to related sections
4. **Keep it up-to-date**: Update docs when code changes
5. **Test code examples**: Ensure all code examples actually work

### Code Examples

- Always include imports
- Use realistic, complete examples
- Add comments explaining non-obvious parts
- Show expected output when helpful

### Structure

- Put getting-started content in `notes/`
- Put task-specific guides in `tutorials/`
- Keep API reference in `modules/` (auto-generated)
- Use consistent heading levels

## Updating Checklist

Before committing documentation changes:

- [ ] Build docs locally without errors: `make html`
- [ ] Check no broken links
- [ ] Verify all code examples work
- [ ] Check images display correctly
- [ ] Review for typos and grammar
- [ ] Update relevant cross-references
- [ ] Clean build to verify: `make clean && make html`
- [ ] Test in multiple browsers (Chrome, Firefox, Safari)
- [ ] Check mobile responsiveness
- [ ] Update CHANGELOG or release notes if major changes

## Quick Reference Commands

```bash
# Build documentation
cd docs && make html

# Clean build
cd docs && make clean && make html

# Watch mode (auto-rebuild)
pip install sphinx-autobuild
cd docs && sphinx-autobuild source build/html

# Check for warnings
cd docs && make html 2>&1 | grep -i warning

# Find broken links
cd docs && make linkcheck

# Serve locally
cd docs/build/html && python -m http.server 8000
```

## Getting Help

- **Sphinx Documentation**: https://www.sphinx-doc.org/
- **RST Reference**: https://docutils.sourceforge.io/rst.html
- **PyTorch Sphinx Theme**: https://github.com/pytorch/pytorch_sphinx_theme
- **ReadTheDocs**: https://docs.readthedocs.io/

## Contributing

When contributing documentation:

1. Follow existing structure and style
2. Test builds locally before pushing
3. Update this guide if adding new documentation workflows
4. Ask for review from maintainers

---

**Last Updated**: 2024 (v2.0 release)

For questions or issues with documentation, please open an issue on GitHub or ask on Slack.
