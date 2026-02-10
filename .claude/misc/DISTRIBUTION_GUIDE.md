# Distribution Guide for PyTorch Connectomics

This guide covers how to distribute PyTorch Connectomics via PyPI and conda-forge.

---

## Table of Contents

1. [PyPI Distribution](#pypi-distribution)
2. [Conda-Forge Distribution](#conda-forge-distribution)
3. [GitHub Releases](#github-releases)
4. [CI/CD Pipelines](#cicd-pipelines)
5. [Version Management](#version-management)
6. [Testing Distributions](#testing-distributions)

---

## PyPI Distribution

### Prerequisites

1. **PyPI Account**: Create account at https://pypi.org/
2. **API Token**: Generate at https://pypi.org/manage/account/token/
3. **TestPyPI Account** (optional): For testing at https://test.pypi.org/

### One-Time Setup

```bash
# Install tools
pip install --upgrade pip build twine

# Configure PyPI credentials
# Create ~/.pypirc
cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...YOUR_TOKEN...

[testpypi]
username = __token__
password = pypi-...YOUR_TOKEN...
EOF

chmod 600 ~/.pypirc
```

### Building the Package

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build wheel and sdist
python -m build

# Check the distributions
twine check dist/*
```

**Output:**
```
dist/
├── pytorch_connectomics-2.0.0-py3-none-any.whl
└── pytorch_connectomics-2.0.0.tar.gz
```

### Testing Locally

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from wheel
pip install dist/pytorch_connectomics-2.0.0-py3-none-any.whl

# Test import
python -c "import connectomics; print(connectomics.__version__)"

# Run demo
python -c "from connectomics.utils.demo import run_demo; run_demo()"
```

### Uploading to TestPyPI (Recommended First)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    pytorch-connectomics

# Verify
python -c "import connectomics; print(connectomics.__version__)"
```

### Uploading to PyPI (Production)

```bash
# Upload to PyPI
twine upload dist/*

# Verify on PyPI
open https://pypi.org/project/pytorch-connectomics/

# Test installation
pip install pytorch-connectomics

# Verify
python -c "import connectomics; print(connectomics.__version__)"
```

---

## Conda-Forge Distribution

### Prerequisites

1. **GitHub Account**: For conda-forge feedstock
2. **Fork conda-forge/staged-recipes**: https://github.com/conda-forge/staged-recipes

### Creating the Recipe

The recipe is already created at `conda-recipe/meta.yaml`. To submit to conda-forge:

1. **Fork staged-recipes:**
   ```bash
   # Visit https://github.com/conda-forge/staged-recipes
   # Click "Fork"
   ```

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/staged-recipes.git
   cd staged-recipes
   ```

3. **Create recipe branch:**
   ```bash
   git checkout -b pytorch-connectomics
   ```

4. **Copy recipe:**
   ```bash
   mkdir recipes/pytorch-connectomics
   cp /path/to/pytorch_connectomics/conda-recipe/meta.yaml recipes/pytorch-connectomics/
   ```

5. **Update SHA256:**
   ```bash
   # Download source tarball
   wget https://github.com/zudi-lin/pytorch_connectomics/archive/v2.0.0.tar.gz

   # Calculate SHA256
   sha256sum v2.0.0.tar.gz
   # Copy the hash to meta.yaml
   ```

6. **Commit and push:**
   ```bash
   git add recipes/pytorch-connectomics/
   git commit -m "Add pytorch-connectomics recipe"
   git push origin pytorch-connectomics
   ```

7. **Create Pull Request:**
   - Go to https://github.com/conda-forge/staged-recipes
   - Click "Pull requests" → "New pull request"
   - Select your fork and branch
   - Fill in PR description

8. **Wait for review:**
   - conda-forge team will review
   - Fix any CI failures
   - Once merged, feedstock repo is created

### Using the Conda Package (After Approval)

```bash
# Install from conda-forge
conda install -c conda-forge pytorch-connectomics

# Or add to environment.yml
cat > environment.yml << EOF
name: pytc
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pytorch-connectomics
EOF

conda env create -f environment.yml
```

---

## GitHub Releases

### Creating a Release

1. **Update version:**
   ```bash
   # Update version in:
   # - setup.py
   # - pyproject.toml
   # - connectomics/__init__.py
   ```

2. **Update changelog:**
   ```bash
   # Edit RELEASE_NOTES.md
   vim RELEASE_NOTES.md
   ```

3. **Commit changes:**
   ```bash
   git add .
   git commit -m "Release v2.0.0"
   git push origin v2.0
   ```

4. **Create tag:**
   ```bash
   git tag -a v2.0.0 -m "Release v2.0.0: Major accessibility improvements"
   git push origin v2.0.0
   ```

5. **Create GitHub Release:**
   - Go to https://github.com/zudi-lin/pytorch_connectomics/releases
   - Click "Draft a new release"
   - Choose tag: v2.0.0
   - Title: "PyTorch Connectomics v2.0.0"
   - Description: Copy from RELEASE_NOTES.md
   - Attach binaries (optional): wheels from dist/
   - Publish release

**GitHub Actions will automatically:**
- Build wheels for all platforms
- Upload to PyPI (if configured)
- Update documentation

---

## CI/CD Pipelines

### GitHub Actions Workflows

We have 3 workflows:

#### 1. Tests (`tests.yml`)

**Triggers:**
- Push to main/master/v2.0
- Pull requests

**Jobs:**
- Run tests on Linux, macOS, Windows
- Python 3.8-3.11
- Upload coverage to Codecov
- Linting (black, flake8, isort, mypy)

#### 2. Build Wheels (`build-wheels.yml`)

**Triggers:**
- Push tags (v*)
- Manual workflow dispatch

**Jobs:**
- Build wheels for all platforms
- Build source distribution (sdist)
- Upload to PyPI (on tag push)

#### 3. Documentation (`docs.yml`)

**Triggers:**
- Push to main/master/v2.0
- Pull requests

**Jobs:**
- Build Sphinx documentation
- Deploy to GitHub Pages (main branch only)

### Setting Up Secrets

**Required secrets in GitHub:**

1. **PYPI_API_TOKEN:**
   - Get from https://pypi.org/manage/account/token/
   - Settings → Secrets → New repository secret
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-...`

2. **CODECOV_TOKEN** (optional):
   - Get from https://codecov.io/
   - Add to GitHub secrets

### Manual Workflow Dispatch

```bash
# Trigger build-wheels manually
gh workflow run build-wheels.yml
```

Or via GitHub UI:
- Actions → Build Wheels → Run workflow

---

## Version Management

### Version Scheme

We use [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

MAJOR: Incompatible API changes
MINOR: New features (backward-compatible)
PATCH: Bug fixes (backward-compatible)
```

**Example:**
- 2.0.0 → Major rewrite (Lightning + MONAI)
- 2.1.0 → Add new model architecture
- 2.1.1 → Fix bug in data loading

### Files to Update

When bumping version:

1. **setup.py**
   ```python
   __version__ = "2.1.0"
   ```

2. **pyproject.toml**
   ```toml
   version = "2.1.0"
   ```

3. **connectomics/__init__.py**
   ```python
   __version__ = "2.1.0"
   ```

4. **conda-recipe/meta.yaml**
   ```yaml
   {% set version = "2.1.0" %}
   ```

5. **RELEASE_NOTES.md**
   ```markdown
   ## Version 2.1.0 (2025-02-01)
   ### Added
   - New feature X
   ### Fixed
   - Bug Y
   ```

### Automated Version Bump

Use `bump2version` for consistency:

```bash
# Install
pip install bump2version

# Create .bumpversion.cfg
cat > .bumpversion.cfg << EOF
[bumpversion]
current_version = 2.0.0
commit = True
tag = True

[bumpversion:file:setup.py]
[bumpversion:file:pyproject.toml]
[bumpversion:file:connectomics/__init__.py]
[bumpversion:file:conda-recipe/meta.yaml]
EOF

# Bump version
bump2version patch  # 2.0.0 → 2.0.1
bump2version minor  # 2.0.1 → 2.1.0
bump2version major  # 2.1.0 → 3.0.0
```

---

## Testing Distributions

### Test PyPI Package

```bash
# Create test environment
python -m venv test_pypi
source test_pypi/bin/activate

# Install from PyPI
pip install pytorch-connectomics

# Run tests
python -c "import connectomics; print(connectomics.__version__)"
pytc-demo
pytc-download --list

# Try importing all modules
python << EOF
from connectomics.config import load_config
from connectomics.models import build_model
from connectomics.lightning import ConnectomicsModule
from connectomics.data.dataset import create_data_dicts_from_paths
from connectomics.utils.demo import run_demo
from connectomics.utils.download import download_dataset
from connectomics.utils.errors import ConnectomicsError
print("All imports successful!")
EOF

# Run full demo
pytc-demo
```

### Test Conda Package

```bash
# Create test environment
conda create -n test_conda python=3.10
conda activate test_conda

# Install from conda-forge
conda install -c conda-forge pytorch-connectomics

# Run same tests as above
python -c "import connectomics; print(connectomics.__version__)"
pytc-demo
```

### Test Wheels for Different Platforms

**Using Docker (for Linux):**

```bash
# Test on Ubuntu 20.04
docker run -it --rm -v $(pwd):/work ubuntu:20.04 bash
cd /work
apt-get update && apt-get install -y python3 python3-pip
pip3 install dist/pytorch_connectomics-2.0.0-py3-none-any.whl
python3 -c "import connectomics; print(connectomics.__version__)"
```

**Using GitHub Actions:**
- Wheels are automatically tested on Linux, macOS, Windows
- Python 3.8-3.12
- See `.github/workflows/build-wheels.yml`

---

## Release Checklist

### Pre-Release

- [ ] Update version in all files
- [ ] Update RELEASE_NOTES.md
- [ ] Run full test suite locally
- [ ] Update documentation
- [ ] Test installation from source
- [ ] Review pending issues/PRs

### Release

- [ ] Create release commit
- [ ] Tag version (v2.0.0)
- [ ] Push tag to GitHub
- [ ] Wait for CI to build wheels
- [ ] Create GitHub Release
- [ ] Upload wheels to PyPI (automatic)
- [ ] Submit conda-forge recipe (if new package)

### Post-Release

- [ ] Verify PyPI package works
- [ ] Verify conda-forge package works (if available)
- [ ] Update documentation site
- [ ] Announce on:
  - [ ] Slack community
  - [ ] Twitter/social media
  - [ ] Mailing lists
  - [ ] Reddit (r/neuroscience, r/MachineLearning)
- [ ] Monitor for issues
- [ ] Respond to user feedback

---

## Troubleshooting

### PyPI Upload Fails

**Error: "File already exists"**

```bash
# Cannot re-upload same version
# Solution: Bump version or use TestPyPI first
```

**Error: "Invalid distribution"**

```bash
# Check with twine
twine check dist/*

# Common issues:
# - Missing README.md
# - Invalid classifiers
# - Broken long_description
```

### Conda Build Fails

**Error: "Missing dependencies"**

```bash
# Check meta.yaml
# All dependencies must be available on conda-forge or defaults
```

**Error: "Tests failed"**

```bash
# Check test section in meta.yaml
# Ensure imports work
# Ensure entry points work
```

### CI/CD Failures

**Error: "Tests failed on Windows"**

```bash
# Common issues:
# - Path separators (use pathlib)
# - Line endings (use .gitattributes)
# - Case-sensitive imports
```

**Error: "Out of disk space"**

```bash
# Clean up artifacts
# Use smaller test data
# Exclude unnecessary files in MANIFEST.in
```

---

## Useful Commands

### PyPI

```bash
# Check package on PyPI
pip index versions pytorch-connectomics

# Download source from PyPI
pip download --no-deps --no-binary :all: pytorch-connectomics

# View package metadata
pip show pytorch-connectomics
```

### Conda

```bash
# Search for package
conda search pytorch-connectomics

# Show package info
conda info pytorch-connectomics

# List files in package
conda list --show-channel-urls pytorch-connectomics
```

### GitHub

```bash
# List releases
gh release list

# Download release assets
gh release download v2.0.0

# Create release
gh release create v2.0.0 dist/*.whl --title "v2.0.0" --notes-file RELEASE_NOTES.md
```

---

## Resources

### PyPI
- **PyPI:** https://pypi.org/
- **TestPyPI:** https://test.pypi.org/
- **Packaging Guide:** https://packaging.python.org/
- **Twine Docs:** https://twine.readthedocs.io/

### Conda-Forge
- **Conda-Forge:** https://conda-forge.org/
- **Staged Recipes:** https://github.com/conda-forge/staged-recipes
- **Docs:** https://conda-forge.org/docs/

### CI/CD
- **GitHub Actions:** https://docs.github.com/en/actions
- **Codecov:** https://codecov.io/
- **Build:** https://pypa-build.readthedocs.io/

---

## Support

**Questions?**
- 💬 Slack: https://join.slack.com/t/pytorchconnectomics/...
- 🐛 Issues: https://github.com/zudi-lin/pytorch_connectomics/issues
- 📧 Email: See maintainers list

---

**Document prepared by:** Claude
**Last updated:** 2025-01-23
