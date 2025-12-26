# Q-Store Scripts Guide

This document describes all the bash scripts available in the `scripts/` directory and how to use them.

## Table of Contents

- [Build Scripts](#build-scripts)
  - [build_binary_distribution.sh](#build_binary_distributionsh)
  - [test_wheel_metadata.sh](#test_wheel_metadatash)
- [Testing Scripts](#testing-scripts)
  - [test_before_build.sh](#test_before_buildsh)
  - [run_coverage.sh](#run_coveragesh)
- [Release Scripts](#release-scripts)
  - [release.sh](#releasesh)
- [Workflow Scripts](#workflow-scripts)
  - [test_workflow_locally.sh](#test_workflow_locallysh)
  - [migrate-workflows.sh](#migrate-workflowssh)

---

## Build Scripts

### build_binary_distribution.sh

**Purpose:** Builds binary wheel distributions for Q-Store with Cython optimizations.

**What it does:**
- Cleans previous builds and artifacts
- Installs Cython if not present
- Builds binary wheel distribution using `setup.py bdist_wheel`
- Validates the wheel file
- Displays installation instructions

**Usage:**

```bash
# Run from project root
./scripts/build_binary_distribution.sh
```

**Prerequisites:**
- Must be run from project root directory
- Python environment activated
- `setup.py` present in project root

**Output:**
- Binary wheel in `dist/` directory
- Build artifacts in `build/` directory

**When to use:**
- Creating optimized binary distributions
- Before publishing to PyPI
- Testing installation across different platforms

---

### test_wheel_metadata.sh

**Purpose:** Validates wheel build and metadata before publishing to PyPI.

**What it does:**
- Cleans previous builds
- Installs build dependencies (pip, setuptools, wheel, build, twine, Cython)
- Builds source distribution (.tar.gz)
- Validates source distribution metadata with twine
- Builds wheel for current Python version
- Validates wheel metadata with twine
- Tests installation in clean environment
- Shows package information (author, version, license)

**Usage:**

```bash
# Run from project root
./scripts/test_wheel_metadata.sh
```

**Prerequisites:**
- Project root directory
- Python environment activated

**Output:**
- Source distribution (.tar.gz) in `dist/`
- Wheel file (.whl) in `dist/`
- Metadata validation report
- Installation test results

**When to use:**
- Before publishing to PyPI
- Validating package metadata
- Testing installation locally

---

## Testing Scripts

### test_before_build.sh

**Purpose:** Comprehensive pre-build test suite to ensure code quality before publication.

**What it does:**
- **Syntax Check:** Validates Python syntax for all `.py` files
- **Import Tests:** Tests imports for all Q-Store modules
- **Core Tests:** Tests core functionality (QuantumDatabase, backends)
- **ML Tests:** Tests ML integration (PyTorch, TensorFlow layers)
- **Algorithm Tests:** Tests quantum algorithms and embeddings
- **Utility Tests:** Tests analysis, monitoring, and profiling tools
- Generates detailed test report with pass/fail counts

**Usage:**

```bash
# Run from project root
./scripts/test_before_build.sh
```

**Prerequisites:**
- Project root directory
- Python environment with Q-Store dependencies installed
- `src/q_store/` directory present

**Output:**
- Console output with colored status indicators (✓ PASSED / ✗ FAILED)
- Summary of test results
- List of failed tests (if any)

**When to use:**
- Before creating a release
- Before pushing to main/dev branch
- After major code changes
- As part of CI/CD pipeline

---

### run_coverage.sh

**Purpose:** Runs comprehensive test suite with coverage reporting.

**What it does:**
- Executes test suite using pytest
- Generates coverage reports in multiple formats:
  - Terminal output (text summary)
  - HTML report (browsable)
  - JSON report (machine-readable)
- Shows list of test files included

**Usage:**

```bash
# Run from project root
./scripts/run_coverage.sh
```

**Prerequisites:**
- pytest and pytest-cov installed
- Test files in `tests/` directory

**Output:**
- `htmlcov/` directory with HTML coverage report
- `coverage.json` with detailed coverage data
- Terminal output with coverage summary

**Viewing Reports:**

```bash
# Open HTML report in browser
firefox htmlcov/index.html
# or
google-chrome htmlcov/index.html
```

**When to use:**
- Checking test coverage percentage
- Identifying untested code
- Before releases to ensure adequate testing
- Continuous integration

---

## Release Scripts

### release.sh

**Purpose:** Creates a release tag, pushes to remote, and triggers the build-wheels workflow.

**What it does:**
1. **Version Verification:**
   - Extracts version from `pyproject.toml`
   - Checks version consistency across:
     - `pyproject.toml`
     - `src/q_store/__init__.py`
     - `setup.py`

2. **Pre-Release Checks:**
   - Validates version format (X.Y.Z)
   - Checks if tag already exists (with option to recreate)
   - Warns about uncommitted changes (with option to continue)
   - Shows current git branch

3. **Release Process:**
   - Creates annotated git tag (v{VERSION})
   - Pushes current branch code to origin
   - Pushes tag to origin to trigger GitHub Actions
   - Includes rollback on failure

4. **Post-Release:**
   - Provides release instructions
   - Shows monitoring URLs for GitHub Actions

**Usage:**

```bash
# Use version from pyproject.toml (recommended)
./scripts/release.sh

# Override with specific version
./scripts/release.sh 4.1.0
```

**Prerequisites:**
- Consistent version numbers across project files (pyproject.toml and __init__.py)
- Git configured with remote repository (origin)
- GitHub Actions workflow configured
- (Optional) Clean working directory - script warns but allows proceeding

**Version Update Workflow:**

Before running `release.sh`, update version in:

1. `pyproject.toml`:
   ```toml
   version = "4.0.0"
   ```

2. `src/q_store/__init__.py`:
   ```python
   __version__ = "4.0.0"
   ```

**Example Session:**

```bash
$ ./scripts/release.sh

==========================================
Q-Store Release Script
==========================================

Version found in pyproject.toml: 4.0.0

Checking version consistency...
  ✓ __init__.py: 4.0.0

Using version from pyproject.toml: 4.0.0
Preparing release: v4.0.0

Current branch: dev

You are about to:
  1. Create tag: v4.0.0
  2. Push tag to origin
  3. Trigger build-wheels workflow on GitHub Actions

Continue with release? (y/N): y

Step 1: Creating release tag...
  ✓ Tag v4.0.0 created

Step 2: Pushing code to origin...
  ✓ Code pushed to origin/dev

Step 3: Pushing release tag...
  ✓ Tag pushed to origin

==========================================
Release v4.0.0 Complete!
==========================================

What happens next:
  1. GitHub Actions will automatically start the build-wheels workflow
  2. Wheels will be built for:
     - Linux (x86_64)
     - macOS (x86_64 and ARM64)
     - Windows (AMD64)
  3. Check workflow progress at:
     https://github.com/YOUR_USERNAME/q-store/actions

Usage:
  ./scripts/release.sh              # Use version from pyproject.toml (4.0.0)
  ./scripts/release.sh <version>    # Override with specific version

To monitor the build:
  gh run list --workflow=build-wheels.yml
  gh run watch

To download artifacts after build completes:
  gh run download
```

**Handling Conflicts:**

If tag already exists:
```bash
Warning: Tag v4.0.0 already exists locally
Do you want to delete and recreate it? (y/N):
```

If uncommitted changes exist:
```bash
Warning: You have uncommitted changes
Do you want to continue anyway? (y/N):
```

**When to use:**
- Creating new releases
- Publishing to PyPI
- Triggering automated builds
- Version tagging

---

## Workflow Scripts

### test_workflow_locally.sh

**Purpose:** Test GitHub Actions workflows locally using `act` before pushing to GitHub.

**What it does:**
- Checks if `act` is installed
- Lists available workflows
- Runs workflows locally in Docker containers
- Provides dry-run option to validate without execution

**Usage:**

```bash
# List available workflows
./scripts/test_workflow_locally.sh list

# Test build job
./scripts/test_workflow_locally.sh build

# Dry run (validate only)
./scripts/test_workflow_locally.sh dry-run

# Test specific workflow job
./scripts/test_workflow_locally.sh <workflow-file> <job-name>
```

**Prerequisites:**
- `act` installed (GitHub Actions local runner)
- Docker installed and running
- Workflow files in `.github/workflows/`

**Installing act:**

```bash
# Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf \
  https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# macOS with Homebrew
brew install act

# Windows with Chocolatey
choco install act-cli
```

**Available Workflows:**
- `build-wheels.yml` - Build wheels for all platforms
- `build-windows.yml` - Build Windows wheels only
- `build-linux.yml` - Build Linux wheels only
- `build-macos.yml` - Build macOS wheels only

**When to use:**
- Testing workflow changes before pushing
- Debugging CI/CD issues locally
- Validating workflow syntax
- Reducing GitHub Actions usage costs

---

### migrate-workflows.sh

**Purpose:** Helps migrate from old separate workflows to new unified workflow structure.

**What it does:**
1. **Backup:** Creates backup of existing workflows
2. **Analysis:** Identifies workflow files to migrate
3. **Migration:** Provides guidance for consolidating workflows
4. **Validation:** Checks for common migration issues
5. **Cleanup:** Optionally removes old workflow files

**Usage:**

```bash
# Run migration
./scripts/migrate-workflows.sh

# Dry run (no changes)
./scripts/migrate-workflows.sh --dry-run

# Skip backup
./scripts/migrate-workflows.sh --no-backup
```

**Prerequisites:**
- Git repository
- `.github/workflows/` directory

**What gets backed up:**
- All `.yml` and `.yaml` files in `.github/workflows/`
- Backup location: `.github/workflows/old-workflows-backup/`

**Migration Process:**

1. **Backup Phase:**
   - Counts existing workflow files
   - Creates backup directory
   - Copies all workflows to backup location

2. **Analysis Phase:**
   - Lists all workflow files
   - Identifies duplicate jobs
   - Suggests consolidation strategies

3. **Migration Phase:**
   - Provides step-by-step instructions
   - Shows example unified workflow
   - Validates new workflow syntax

4. **Cleanup Phase:**
   - Optionally removes old workflows
   - Updates repository documentation
   - Tests new workflows locally

**When to use:**
- Consolidating multiple workflow files
- Upgrading to unified workflow pattern
- Reducing workflow maintenance overhead
- Improving CI/CD efficiency

---

## General Usage Notes

### Running Scripts

All scripts should be run from the **project root directory**:

```bash
# Correct (from project root)
cd /home/yucelz/yz_code/q-store
./scripts/build_binary_distribution.sh

# Incorrect (from scripts directory)
cd scripts/
./build_binary_distribution.sh  # Will fail
```

### Script Permissions

Ensure scripts have execute permissions:

```bash
# Make all scripts executable
chmod +x scripts/*.sh

# Or individually
chmod +x scripts/build_binary_distribution.sh
```

### Exit Codes

All scripts follow standard exit code conventions:
- `0` - Success
- `1` - General error
- `2` - Misuse of shell command

### Color Output

Scripts use colored output for clarity:
- 🟢 **GREEN** - Success/Passed
- 🔴 **RED** - Error/Failed  
- 🟡 **YELLOW** - Warning/Info
- 🔵 **BLUE** - Informational

### Environment Requirements

Most scripts expect:
- Python 3.8 or higher
- Virtual environment activated (recommended)
- Dependencies installed via `pip install -e .`

---

## Typical Workflow Sequences

### Development Cycle

```bash
# 1. Make code changes
# 2. Run tests with coverage
./scripts/run_coverage.sh

# 3. Test before committing
./scripts/test_before_build.sh

# 4. Commit and push changes
git add .
git commit -m "Your changes"
git push
```

### Release Cycle

```bash
# 1. Update version numbers (pyproject.toml, __init__.py, setup.py)

# 2. Run comprehensive tests
./scripts/test_before_build.sh

# 3. Test wheel building
./scripts/test_wheel_metadata.sh

# 4. Create release
./scripts/release.sh

# 5. Monitor GitHub Actions for wheel builds
# 6. Publish to PyPI (automated via GitHub Actions)
```

### Testing Workflow Changes

```bash
# 1. Modify workflow files in .github/workflows/

# 2. Test locally with act
./scripts/test_workflow_locally.sh dry-run

# 3. Run specific workflow
./scripts/test_workflow_locally.sh build-wheels.yml build

# 4. If successful, push to GitHub
git add .github/workflows/
git commit -m "Update workflows"
git push
```

---

## Troubleshooting

### Script Fails with "Must run from project root"

**Solution:** Change to project root directory:
```bash
cd /home/yucelz/yz_code/q-store
```

### "Permission denied" Error

**Solution:** Add execute permission:
```bash
chmod +x scripts/script_name.sh
```

### Import Errors During Tests

**Solution:** Install package in development mode:
```bash
pip install -e .
```

### act Not Found (test_workflow_locally.sh)

**Solution:** Install act:
```bash
curl --proto '=https' --tlsv1.2 -sSf \
  https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

### Build Failures

**Solution:** Clean and rebuild:
```bash
rm -rf build/ dist/ *.egg-info
./scripts/build_binary_distribution.sh
```

---

## Additional Resources

- **GitHub Actions Docs:** https://docs.github.com/en/actions
- **act Documentation:** https://github.com/nektos/act
- **PyPI Publishing:** https://packaging.python.org/tutorials/packaging-projects/
- **Python Packaging:** https://packaging.python.org/

---

## Contributing

When adding new scripts:

1. Use bash shebang: `#!/bin/bash`
2. Include help text and usage examples
3. Use colored output for clarity
4. Add error checking (`set -e`)
5. Document in this README
6. Make executable: `chmod +x scripts/your_script.sh`

---

**Last Updated:** December 26, 2025  
**Q-Store Version:** 4.0.0
