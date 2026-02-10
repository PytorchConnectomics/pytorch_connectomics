# Codebase Cleanup Plan

This document outlines redundant/legacy files and folders to remove for a cleaner repository.

---

## Files/Folders to Remove

### 1. Runtime/Build Artifacts (Should NOT be in repo)

**Folders:**
- `lightning_logs/` - PyTorch Lightning training logs (gitignored)
- `outputs/` - Training outputs (gitignored)
- `datasets/` - Downloaded data (gitignored)
- `slurm_outputs/` - SLURM job outputs (gitignored)
- `__pycache__/` - Python cache (gitignored)
- `.pytest_cache/` - Pytest cache (gitignored)
- `connectomics.egg-info/` - Build artifact (gitignored)
- `.vscode/` - Editor config (should be in .gitignore)

**Action:** These are already gitignored but exist locally. Leave them (local only).

---

### 2. Legacy/Outdated Files

#### A. Old Notebooks (notebooks/)
**Keep:**
- `PyTorch_Connectomics_Tutorial.ipynb` - NEW Colab tutorial ✅

**Remove (legacy):**
- `voldataset_cond.ipynb` - Old dataset testing
- `seg2diffgrads.ipynb` - Old preprocessing
- `seg2affinity.ipynb` - Old preprocessing
- `augmentation.ipynb` - Old augmentation testing
- `seg2distance.ipynb` - Old preprocessing
- `blending.ipynb` - Old blending notebook

**Reason:** These are outdated notebooks from v1.0 that use old API. Not maintained.

---

#### B. Examples Folder (examples/)
**Contains:**
- `monai_transforms_usage.py` - MONAI transform examples

**Decision:** Move to notebooks or remove entirely

**Reason:** Single file in entire folder. Better as notebook or docs example.

---

#### C. Duplicate/Legacy Documentation
**Keep:**
- `README.md` - Main README (update with README_NEW.md)
- `QUICKSTART.md` - Phase 1 ✅
- `TROUBLESHOOTING.md` - Phase 1 ✅
- `CONTRIBUTING.md` - Contributing guide ✅
- `LICENSE` - License file ✅
- `RELEASE_NOTES.md` - Release history ✅

**Remove:**
- `README_NEW.md` - Temporary file (merge into README.md)
- `INSTALL_GUIDE.md` - Superseded by QUICKSTART.md
- `BENCHMARK.md` - Outdated benchmarks

**Reason:** Consolidate documentation, remove duplicates.

---

### 3. Scripts Cleanup (scripts/)

**Keep:**
- `main.py` - Primary training script ✅
- `build_package.sh` - Phase 2 packaging ✅
- `visualize_neuroglancer.py` - Visualization tool ✅
- `slurm_launcher.py` - HPC launcher ✅
- `slurm_template.sh` - SLURM template ✅

**Consider removing:**
- `profile_dataloader.py` - Development profiling script
- `tools/` folder - Check contents

**Action:** Check tools/ folder first

---

### 4. Docker (docker/)

**Current status:** Outdated Dockerfile (CUDA 11.3, Ubuntu 20.04)

**Options:**
1. Update Dockerfile to modern version
2. Remove entirely (use PyPI/conda instead)

**Recommendation:** Keep but update (see Phase 2 notes about skipping Docker)

**Decision:** Keep for now, mark for Phase 4 improvement

---

## Cleanup Actions

### Phase 1: Remove Legacy Notebooks

```bash
cd notebooks/
# Keep only the new tutorial
rm voldataset_cond.ipynb
rm seg2diffgrads.ipynb
rm seg2affinity.ipynb
rm augmentation.ipynb
rm seg2distance.ipynb
rm blending.ipynb
```

### Phase 2: Remove Examples Folder

```bash
# Option 1: Remove entirely
rm -rf examples/

# Option 2: Convert to notebook
# (Decide based on usefulness)
```

### Phase 3: Consolidate Documentation

```bash
# Merge README_NEW.md into README.md
mv README.md README_OLD_BACKUP.md
mv README_NEW.md README.md

# Remove duplicate docs
rm INSTALL_GUIDE.md
rm BENCHMARK.md
rm README_OLD_BACKUP.md  # After verifying
```

### Phase 4: Clean Scripts

```bash
cd scripts/
# Check tools/ folder
ls -la tools/

# If empty or unused, remove
rm -rf tools/

# Consider removing profiling script (keep for development)
# rm profile_dataloader.py
```

### Phase 5: Update .gitignore

Add entries to ensure build artifacts are ignored:
```
# Runtime directories
lightning_logs/
outputs/
slurm_outputs/
datasets/

# Build artifacts
*.egg-info/
__pycache__/
.pytest_cache/
dist/
build/

# Editor configs
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
```

---

## Impact Assessment

### Before Cleanup

```
Repository size: ~XXX MB (with datasets/outputs)
Number of notebooks: 7
Number of documentation files: 7
Legacy files: 10+
```

### After Cleanup

```
Repository size: ~XX MB (reduced)
Number of notebooks: 1 (focused)
Number of documentation files: 5 (consolidated)
Legacy files: 0
```

### Benefits

1. **Clearer structure**: Less confusing for new users
2. **Faster cloning**: Smaller repository
3. **Better maintenance**: Fewer files to update
4. **Professional appearance**: Clean, focused codebase
5. **Reduced confusion**: No outdated examples

---

## Files to Keep (Essential)

### Root Level
- `README.md` (updated)
- `QUICKSTART.md`
- `TROUBLESHOOTING.md`
- `CONTRIBUTING.md`
- `LICENSE`
- `RELEASE_NOTES.md`
- `setup.py`
- `pyproject.toml`
- `MANIFEST.in`
- `quickstart.sh`
- `install.py`
- `justfile`
- `.gitignore`
- `.gitattributes`
- `.readthedocs.yaml`

### Folders
- `.github/` - CI/CD workflows
- `.claude/` - Internal docs (optional, could be in docs/)
- `connectomics/` - Main package
- `scripts/` - Utility scripts
- `tutorials/` - Config examples
- `notebooks/` - Colab tutorial (1 file)
- `conda-recipe/` - Conda packaging
- `docker/` - Docker config (optional)
- `docs/` - Sphinx documentation
- `tests/` - Test suite

---

## Migration Path

### Option 1: Aggressive Cleanup (Recommended)

Remove all legacy files immediately. Archive them in a `legacy` branch if needed.

**Commands:**
```bash
# Create legacy branch for backup
git checkout -b legacy/v1.0-notebooks
git push origin legacy/v1.0-notebooks

# Return to main branch
git checkout v2.0

# Remove legacy files
git rm notebooks/voldataset_cond.ipynb
git rm notebooks/seg2diffgrads.ipynb
git rm notebooks/seg2affinity.ipynb
git rm notebooks/augmentation.ipynb
git rm notebooks/seg2distance.ipynb
git rm notebooks/blending.ipynb
git rm -r examples/
git rm README_NEW.md INSTALL_GUIDE.md BENCHMARK.md

# Commit
git commit -m "Remove legacy notebooks and duplicate documentation

- Remove 6 outdated notebooks from v1.0
- Remove examples/ folder (single unused file)
- Remove duplicate documentation (INSTALL_GUIDE.md, BENCHMARK.md)
- Keep only new Colab tutorial and consolidated docs

Archived in legacy/v1.0-notebooks branch for reference"
```

### Option 2: Gradual Cleanup

Move files to `legacy/` folder first, remove in next release.

**Commands:**
```bash
mkdir legacy/
mv notebooks/voldataset_cond.ipynb legacy/
# ... etc
```

---

## Recommendation

**Use Option 1 (Aggressive Cleanup)** because:
1. V2.0 is a major release (good time for breaking changes)
2. Old notebooks are incompatible with new API anyway
3. Reduces confusion for new users
4. Files are preserved in git history
5. Can create legacy branch for reference

---

## Post-Cleanup Checklist

- [ ] Remove legacy notebooks
- [ ] Remove examples/ folder
- [ ] Consolidate documentation (README_NEW → README)
- [ ] Remove duplicate docs (INSTALL_GUIDE, BENCHMARK)
- [ ] Update .gitignore
- [ ] Test that package still builds
- [ ] Test that new tutorial still works
- [ ] Update documentation references
- [ ] Commit changes with clear message
- [ ] Update RELEASE_NOTES.md

---

## Next Steps

1. Review this plan
2. Create backup/legacy branch
3. Execute cleanup commands
4. Test everything still works
5. Commit and push

