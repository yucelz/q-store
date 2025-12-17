# Quick Reference Guide

## Common Scenarios

### 🚀 Making a Release

```bash
# 1. Update version
vim pyproject.toml  # or setup.py

# 2. Commit changes
git add .
git commit -m "chore: bump version to 1.0.0"
git push

# 3. Create and push tag
git tag v1.0.0
git push origin v1.0.0

# 4. Monitor at: github.com/YOUR_ORG/YOUR_REPO/actions
```

### 🧪 Testing Before Release

```bash
# Option 1: Create a PR (automatic)
git checkout -b test-build
git push origin test-build
# Open PR on GitHub

# Option 2: Manual trigger
# Go to: Actions → Test Build Wheels → Run workflow
# Select platforms: linux (fastest)
```

### 🐛 Build Failed - How to Fix

```bash
# 1. Check which platform failed in GitHub Actions UI

# 2. Delete the tag
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0

# 3. Fix the issue, commit

# 4. Recreate tag
git tag v1.0.0
git push origin v1.0.0
```

### 📦 Testing Locally

```bash
# Install build tools
pip install build cibuildwheel twine

# Test metadata
python -m build --sdist
python -m twine check dist/*

# Test wheel build (Linux example)
python -m cibuildwheel --platform linux --output-dir test-wheels

# Test import
pip install test-wheels/*.whl
python -c "from q_store import QuantumDatabase; print('OK')"
```

### 🔄 Update After Failed Publish

```bash
# If PyPI publish failed but wheels are good:

# 1. Get the artifacts from GitHub Actions
# - Go to failed workflow run
# - Download artifacts

# 2. Publish manually
pip install twine
twine upload --skip-existing artifacts/*.whl
```

## Common Pre-check Failures

### ❌ "Neither setup.py nor pyproject.toml found"

**Fix:**
```bash
# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "Cython>=0.29.0"]
build-backend = "setuptools.build_meta"

[project]
name = "q-store"
version = "1.0.0"
description = "Quantum database"
# ... more metadata
EOF
```

### ❌ "PYPI_API_TOKEN secret is not set"

**Fix:**
1. Go to https://pypi.org/manage/account/token/
2. Create new token (limit to project)
3. GitHub: Settings → Secrets → Actions → New secret
4. Name: `PYPI_API_TOKEN`
5. Paste token

### ⚠️ "Version X already exists on PyPI"

**Fix Option 1 (Recommended):**
```bash
# Use a new version
vim pyproject.toml  # Change 1.0.0 → 1.0.1
git add pyproject.toml
git commit -m "chore: bump to 1.0.1"
git tag v1.0.1
git push origin v1.0.1
```

**Fix Option 2 (Not Recommended):**
- Delete version from PyPI (can't be undone)
- Reuse same version

## Platform-Specific Issues

### Linux: Missing library

```yaml
# In build-wheels-improved.yml, update:
CIBW_BEFORE_BUILD: |
  pip install Cython>=0.29.0 &&
  yum install -y libfoo-devel
```

### macOS: Wrong architecture

```yaml
# For Intel (macos-12):
CIBW_ARCHS_MACOS: "x86_64"

# For ARM (macos-14):
CIBW_ARCHS_MACOS: "arm64"
```

### Windows: Compilation errors

```yaml
# Ensure MSVC flags are set:
CIBW_ENVIRONMENT: "CL=/O2 /GL /MP"
```

## Workflow Triggers

### When workflows run:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `build-wheels-improved.yml` | Tag push `v*` | Production release |
| `build-wheels-test.yml` | PR to main/develop | Test before merge |
| `build-wheels-test.yml` | Manual | Test specific platforms |

### Manual trigger:

```
GitHub → Actions → [Workflow Name] → Run workflow
```

## Version Naming

### ✅ Good versions:

- `v1.0.0` - Production
- `v1.2.3` - Production update
- `v2.0.0-beta.1` - Beta (marked as prerelease)
- `v1.0.0-rc.1` - Release candidate (marked as prerelease)
- `v1.0.0-alpha.1` - Alpha (marked as prerelease)

### ❌ Bad versions:

- `1.0.0` - Missing 'v' prefix, won't trigger
- `v1.0` - Not specific enough
- `release-1.0.0` - Wrong format

## Expected Build Times

| Phase | Duration | Cost Factor |
|-------|----------|-------------|
| Pre-checks | 2-3 min | 1x (Linux) |
| Linux build | 15-20 min | 1x |
| macOS Intel | 20-25 min | 10x |
| macOS ARM | 20-25 min | 10x |
| Windows | 25-35 min | 2x |
| Publish | 2-3 min | 1x (Linux) |
| **Total** | ~30-35 min | ~170x equiv |

**Tip:** Test on Linux first (cheapest)

## Checking Build Status

### From command line:

```bash
# Using GitHub CLI
gh run list --workflow=build-wheels-improved.yml

# Watch specific run
gh run watch [RUN_ID]

# View logs
gh run view [RUN_ID] --log
```

### From web:

```
https://github.com/YOUR_ORG/YOUR_REPO/actions
```

## Environment Variables

### Customizing builds:

```yaml
# Python versions
CIBW_BUILD: cp311-* cp312-* cp313-*

# Architectures
CIBW_ARCHS_LINUX: "x86_64"
CIBW_ARCHS_MACOS: "x86_64 arm64"
CIBW_ARCHS_WINDOWS: "AMD64"

# Dependencies
CIBW_BEFORE_BUILD: pip install Cython numpy

# Testing
CIBW_TEST_REQUIRES: pytest
CIBW_TEST_COMMAND: pytest {project}/tests
```

## Artifacts

### Production (30 days):
- `wheels-linux-x86_64`
- `wheels-macos-intel`
- `wheels-macos-arm64`
- `wheels-windows-amd64`

### Test (7 days):
- `test-wheels-linux`
- `test-wheels-macos-intel`
- `test-wheels-macos-arm64`
- `test-wheels-windows`

### Downloading:

```bash
# Using GitHub CLI
gh run download [RUN_ID]

# Or from web UI:
# Workflow run → Artifacts section → Download
```

## Security Checklist

- [ ] `PYPI_API_TOKEN` stored as secret (not in code)
- [ ] Token scoped to single project
- [ ] Token has minimum required permissions
- [ ] Secrets never logged or printed
- [ ] Using `__token__` as username (not personal account)

## Before Each Release

- [ ] Update version number in code
- [ ] Update CHANGELOG.md
- [ ] Test build locally or via PR
- [ ] Commit all changes
- [ ] Create annotated tag: `git tag -a v1.0.0 -m "Release 1.0.0"`
- [ ] Push tag: `git push origin v1.0.0`
- [ ] Monitor GitHub Actions
- [ ] Verify GitHub release created
- [ ] Verify on PyPI: `pip index versions q-store`
- [ ] Test install: `pip install q-store==1.0.0`

## Emergency: Cancel Release

```bash
# 1. Cancel GitHub Actions workflow immediately
# GitHub → Actions → Running workflow → Cancel

# 2. Delete the tag
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0

# 3. Delete GitHub release (if created)
# GitHub → Releases → [Your release] → Delete

# 4. PyPI: Cannot delete once uploaded!
# Use a new version number instead
```

## Getting Help

1. Check workflow logs in GitHub Actions
2. Review WORKFLOW_IMPROVEMENTS.md (full docs)
3. Check this quick reference
4. Look at workflow summary page
5. Review pre-check error messages

## Useful Commands

```bash
# Check current version
python setup.py --version
# or
python -c "import tomli; print(tomli.load(open('pyproject.toml', 'rb'))['project']['version'])"

# List all tags
git tag -l

# Check tag details
git show v1.0.0

# Delete local tag
git tag -d v1.0.0

# Delete remote tag
git push origin :refs/tags/v1.0.0

# Fetch tags
git fetch --tags

# Check PyPI versions
pip index versions q-store

# Test wheel
pip install /path/to/wheel.whl
python -c "import q_store; print(q_store.__version__)"
```
