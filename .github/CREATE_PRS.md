# How to Create Pull Requests for Phases 6-10

This guide helps you create GitHub Pull Requests for the BANIS integration phases.

---

## Quick Start

### Option 1: Using GitHub CLI (Recommended)

```bash
# Install gh CLI if needed
# macOS: brew install gh
# Ubuntu: sudo apt install gh
# Or download from: https://cli.github.com/

# Authenticate (one-time)
gh auth login

# Create PRs for each phase
cd /projects/weilab/weidf/lib/pytorch_connectomics

# Phase 6
gh pr create \
  --title "Phase 6: EM-Specific Augmentations" \
  --body-file .github/PR_TEMPLATES/phase6.md \
  --label "enhancement,BANIS,augmentation"

# Phase 7
gh pr create \
  --title "Phase 7: Numba-Accelerated Connected Components" \
  --body-file .github/PR_TEMPLATES/phase7.md \
  --label "enhancement,BANIS,performance"

# Phase 8
gh pr create \
  --title "Phase 8: Weighted Dataset Mixing" \
  --body-file .github/PR_TEMPLATES/phase8.md \
  --label "enhancement,BANIS,data"

# Phase 9
gh pr create \
  --title "Phase 9: Optuna-Based Threshold Tuning" \
  --body-file .github/PR_TEMPLATES/phase9.md \
  --label "enhancement,BANIS,optimization"

# Phase 10
gh pr create \
  --title "Phase 10: Auto-Configuration System" \
  --body-file .github/PR_TEMPLATES/phase10.md \
  --label "enhancement,BANIS,config,documentation"
```

### Option 2: GitHub Web UI

1. **Commit and push your changes:**
   ```bash
   git add .
   git commit -m "feat: implement BANIS phases 6-10"
   git push origin your-branch-name
   ```

2. **Go to GitHub repository**
   - Navigate to your repo on github.com
   - Click "Pull requests" → "New pull request"

3. **Create PR for each phase:**
   - **Title:** Copy from PR template (e.g., "Phase 6: EM-Specific Augmentations")
   - **Description:** Copy entire content from corresponding `.github/PR_TEMPLATES/phase*.md`
   - **Labels:** Add `enhancement`, `BANIS`, and phase-specific labels
   - **Assignees:** Assign to yourself
   - **Reviewers:** Request reviews if needed

4. **Repeat for all phases** (6, 7, 8, 9, 10)

---

## Creating Issues First (Recommended)

Track each phase with a GitHub issue before creating PRs:

```bash
# Create issues
gh issue create \
  --title "Phase 6: EM-Specific Augmentations" \
  --body "Implement EM-specific augmentations (DropSliced, ShiftSliced)" \
  --label "enhancement,BANIS"

gh issue create \
  --title "Phase 7: Numba-Accelerated Connected Components" \
  --body "Implement fast connected components for affinity segmentation" \
  --label "enhancement,BANIS"

gh issue create \
  --title "Phase 8: Weighted Dataset Mixing" \
  --body "Implement multi-dataset utilities for mixing synthetic and real data" \
  --label "enhancement,BANIS"

gh issue create \
  --title "Phase 9: Optuna-Based Threshold Tuning" \
  --body "Implement hyperparameter optimization using Optuna and skeleton metrics" \
  --label "enhancement,BANIS"

gh issue create \
  --title "Phase 10: Auto-Configuration System" \
  --body "Document and test auto-configuration system" \
  --label "enhancement,BANIS,documentation"

# Then reference issues in PRs
gh pr create \
  --title "Phase 6: EM-Specific Augmentations" \
  --body-file .github/PR_TEMPLATES/phase6.md \
  --body "Closes #<issue-number>" \
  --label "enhancement,BANIS"
```

---

## Single Combined PR (Alternative)

If you prefer one large PR instead of separate ones:

```bash
# Create comprehensive PR
gh pr create \
  --title "BANIS Integration: Phases 6-10" \
  --body "$(cat <<EOF
# BANIS Integration: Phases 6-10

Implements features from BANIS baseline to improve PyTorch Connectomics:

## Phase 6: EM-Specific Augmentations
- Added DropSliced and ShiftSliced transforms
- 4 preset configurations in tutorials/presets/

## Phase 7: Numba-Accelerated Connected Components
- Fast affinity-based segmentation (10-100x speedup)
- Added affinity_cc3d() function

## Phase 8: Weighted Dataset Mixing
- 3 dataset mixing strategies (weighted, stratified, uniform)
- 18 comprehensive tests

## Phase 9: Optuna-Based Threshold Tuning
- Skeleton-based metrics (NERL, VOI)
- Bayesian optimization with Optuna
- 18 comprehensive tests

## Phase 10: Auto-Configuration System
- Documented and tested existing auto-config
- 30+ comprehensive tests
- nnU-Net-inspired planning

## Files Added
See individual phase templates in .github/PR_TEMPLATES/ for details.

## Testing
\`\`\`bash
pytest tests/test_dataset_multi.py -v
pytest tests/test_auto_tuning.py -v
pytest tests/test_auto_config.py -v
\`\`\`

## Documentation
- Updated .claude/CLAUDE.md
- Created .claude/IMPLEMENTATION_HISTORY.md
- Example configs in tutorials/
EOF
)" \
  --label "enhancement,BANIS,major"
```

---

## After Creating PRs

### 1. Update IMPLEMENTATION_HISTORY.md

Replace TODO placeholders with actual PR numbers:

```markdown
### Phase 6: EM-Specific Augmentations ✅
**PR:** #125

### Phase 7: Numba-Accelerated Connected Components ✅
**PR:** #126

### Phase 8: Weighted Dataset Mixing ✅
**PR:** #127

### Phase 9: Optuna-Based Threshold Tuning ✅
**PR:** #128

### Phase 10: Auto-Configuration System ✅
**PR:** #129
```

### 2. Clean Up .claude/ Directory

Once PRs are created, you can archive or delete detailed summaries:

```bash
# Option 1: Delete summaries (info now in PRs)
rm .claude/PHASE6_SUMMARY.md
rm .claude/PHASE7_SUMMARY.md
rm .claude/PHASE8_SUMMARY.md
rm .claude/PHASE9_SUMMARY.md
rm .claude/PHASE10_SUMMARY.md

# Option 2: Archive summaries
mkdir -p .claude/archive
mv .claude/PHASE*_SUMMARY.md .claude/archive/

# Keep these files:
# - .claude/CLAUDE.md (living documentation)
# - .claude/DESIGN.md (architecture principles)
# - .claude/BANIS_PLAN.md (implementation plan)
# - .claude/IMPLEMENTATION_HISTORY.md (links to PRs)
# - .claude/MEDNEXT*.md (MedNeXt documentation)
```

### 3. Update README.md

Add link to implementation tracking:

```markdown
## Recent Updates

See [Implementation History](.claude/IMPLEMENTATION_HISTORY.md) for detailed changelog.

Major features:
- ✅ Phase 6-10: BANIS Integration (#125-#129)
- ✅ Phase 1-5: MedNeXt Integration
- ✅ I/O Refactoring
```

---

## PR Review Checklist

When reviewing PRs, check:

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Example configs provided
- [ ] CLAUDE.md reflects changes
- [ ] No breaking changes (or clearly documented)
- [ ] Code follows project style
- [ ] Type hints present
- [ ] Docstrings complete

---

## Merging Strategy

### Option 1: Merge Separately (Recommended)
- Merge Phase 6 → wait for CI
- Merge Phase 7 → wait for CI
- ...
- Allows incremental integration
- Easier to bisect if issues arise

### Option 2: Merge Together
- Merge all phases at once
- Faster but riskier
- Requires comprehensive testing

### Option 3: Feature Branch
- Create `feature/banis-integration` branch
- Merge all phases into feature branch
- Merge feature branch to main when stable

---

## Troubleshooting

### "gh command not found"
```bash
# Install GitHub CLI
# macOS
brew install gh

# Ubuntu/Debian
type -p curl >/dev/null || sudo apt install curl -y
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# Or download from https://cli.github.com/
```

### "Authentication required"
```bash
gh auth login
# Follow prompts to authenticate with GitHub
```

### "Cannot create PR from current branch"
```bash
# Make sure you're on a feature branch, not main
git checkout -b feature/banis-phases-6-10
git push -u origin feature/banis-phases-6-10

# Then create PR
gh pr create ...
```

---

## Questions?

- Check `.github/PR_TEMPLATES/` for detailed PR descriptions
- See `.claude/IMPLEMENTATION_HISTORY.md` for phase summaries
- Refer to `.claude/CLAUDE.md` for usage examples
