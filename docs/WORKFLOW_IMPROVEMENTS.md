# GitHub Actions Workflow Improvements

## Overview

This document explains the improvements made to the GitHub Actions workflows for building and publishing Python wheels to PyPI.

## Problems Solved

### Previous Issues

1. **Race Conditions**: Multiple workflows tried to publish to PyPI simultaneously
2. **No Pre-checks**: Builds started without validating package metadata or dependencies
3. **Tag Reuse Problem**: If one platform failed, the tag was consumed and couldn't be reused
4. **Duplicate Releases**: Each workflow tried to create GitHub releases independently
5. **Poor Error Visibility**: Hard to see which platform failed and why
6. **No Build Testing**: No way to test builds without publishing

### New Solutions

1. **Single Publishing Job**: Only one job publishes to PyPI after ALL platforms succeed
2. **Comprehensive Pre-checks**: Validates everything before starting expensive builds
3. **Independent Build Stages**: Each platform builds separately; if one fails, others continue
4. **Unified Release**: GitHub release created only once with all artifacts
5. **Better Visibility**: Clear summary of what succeeded/failed
6. **Test Workflow**: Separate workflow for testing builds without publishing

---

## Workflow Architecture

### Main Production Workflow: `build-wheels-improved.yml`

```
┌─────────────────────────────────────────────────────────────┐
│                      pre_checks                              │
│  ✓ Version validation                                       │
│  ✓ Package metadata check                                   │
│  ✓ PyPI credentials verification                            │
│  ✓ Duplicate version check                                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
      ┌───────────┼───────────┬───────────┬───────────┐
      ▼           ▼           ▼           ▼           ▼
 ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
 │ Linux  │ │  macOS   │ │  macOS   │ │ Windows  │
 │  x64   │ │  Intel   │ │  ARM64   │ │  AMD64   │
 └────┬───┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
      └──────────┼────────────┼────────────┘
                 ▼
         ┌───────────────┐
         │ create_release │
         │  (GitHub)      │
         └───────┬────────┘
                 ▼
         ┌───────────────┐
         │ publish_pypi   │
         │  (Single Job)  │
         └───────┬────────┘
                 ▼
         ┌───────────────┐
         │publish_summary │
         └────────────────┘
```

### Test Workflow: `build-wheels-test.yml`

```
┌─────────────────────────────────────────────────┐
│           test_pre_checks                       │
│  ✓ Metadata validation                         │
│  ✓ Syntax checking                             │
│  ✓ Platform selection                          │
└─────────────┬───────────────────────────────────┘
              │
      ┌───────┼────────┬────────┬─────────┐
      ▼       ▼        ▼        ▼         ▼
   Linux   macOS   macOS   Windows
           Intel   ARM64
              │
              ▼
        (Artifacts saved, NO publishing)
```

---

## Workflow Files

### 1. `build-wheels-improved.yml` - Production Workflow

**Triggers:**
- Push of version tags (e.g., `v1.0.0`, `v1.2.3-beta.1`)
- Manual workflow dispatch

**Key Features:**

#### Pre-check Job
- Validates version format
- Checks package metadata
- Verifies PyPI credentials exist
- Checks for version conflicts on PyPI
- Extracts version info for downstream jobs

#### Build Jobs (Parallel)
- **build_linux**: manylinux wheels for x86_64
- **build_macos_intel**: macOS Intel wheels
- **build_macos_arm**: macOS ARM64 (Apple Silicon) wheels
- **build_windows**: Windows AMD64 wheels

Each build job:
- Runs independently
- Uses cibuildwheel for consistent builds
- Tests imports after building
- Uploads artifacts with unique names
- Fails loudly if no wheels produced

#### Publishing Jobs (Sequential)

**create_release**: 
- Runs ONLY if all builds succeed
- Downloads all wheel artifacts
- Verifies expected wheel count
- Creates single GitHub release
- Marks as prerelease if version contains alpha/beta/rc/dev

**publish_pypi**:
- Runs ONLY after successful release creation
- Downloads all wheels
- Validates with twine
- Publishes to PyPI in single operation
- Uses `--skip-existing` for idempotency

**publish_summary**:
- Creates markdown summary
- Shows status of all jobs
- Visible in GitHub Actions UI

### 2. `build-wheels-test.yml` - Test Workflow

**Triggers:**
- Pull requests to main/master/develop
- Manual workflow dispatch with platform selection

**Key Features:**
- Builds wheels but does NOT publish
- Allows selecting specific platforms to test
- Faster feedback during development
- Shorter artifact retention (7 days vs 30)
- No PyPI credentials required

---

## Usage Guide

### Production Release

1. **Prepare your release:**
   ```bash
   # Update version in pyproject.toml or setup.py
   # Commit changes
   git add .
   git commit -m "Bump version to 1.0.0"
   git push
   ```

2. **Create and push tag:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

3. **Monitor the workflow:**
   - Go to GitHub Actions tab
   - Watch the "Build and Publish Wheels" workflow
   - Pre-checks run first (fast)
   - Then all platform builds run in parallel
   - Finally, release and PyPI publish run

4. **If a build fails:**
   - Check the failed job logs
   - Fix the issue
   - Delete the tag locally and remotely:
     ```bash
     git tag -d v1.0.0
     git push origin :refs/tags/v1.0.0
     ```
   - Fix the code, commit, and recreate the tag

### Test Builds (Before Release)

1. **Test all platforms:**
   - Create a pull request, or
   - Go to Actions → "Test Build Wheels" → "Run workflow"
   - Select platforms: `linux,macos-intel,macos-arm,windows`

2. **Test specific platform:**
   - Go to Actions → "Test Build Wheels" → "Run workflow"
   - Select platform: `linux` (fastest)

3. **Download test wheels:**
   - Go to the completed workflow run
   - Scroll to "Artifacts" section
   - Download test wheels for local testing

---

## Configuration Requirements

### GitHub Secrets

**Required:**
- `PYPI_API_TOKEN`: Your PyPI API token
  - Get from: https://pypi.org/manage/account/token/
  - Set in: Repository Settings → Secrets → Actions

### Repository Settings

**Actions Permissions:**
- Settings → Actions → General → Workflow permissions
- Select: "Read and write permissions"
- Check: "Allow GitHub Actions to create and approve pull requests"

---

## Pre-check Validations

The workflow performs these validations before building:

### 1. Version Format Check
- ✓ Validates semantic versioning (e.g., `1.0.0`, `2.1.3-beta.1`)
- ✓ Detects prerelease markers (alpha, beta, rc, dev)
- ⚠️ Warns on non-standard formats

### 2. Package Metadata Check
- ✓ Verifies setup.py or pyproject.toml exists
- ✓ Builds source distribution to validate metadata
- ✓ Runs `twine check` to validate PyPI requirements

### 3. PyPI Credentials Check
- ✓ Verifies `PYPI_API_TOKEN` secret is set
- ✗ Fails early if missing (don't waste build time)

### 4. Version Conflict Check
- ✓ Checks if version already exists on PyPI
- ⚠️ Warns but allows (--skip-existing handles it)

### 5. File Structure Check
- ✓ Looks for README.md, LICENSE
- ⚠️ Warns if missing (recommended but not required)

---

## Build Verification

Each platform build includes:

### 1. Wheel Building
- Uses cibuildwheel for reproducible builds
- Builds for Python 3.11, 3.12, 3.13
- Installs Cython before building (for compiled extensions)

### 2. Import Testing
- Tests: `from q_store import QuantumDatabase`
- Runs in clean environment
- Fails if import fails

### 3. Artifact Verification
- Checks that wheels were actually created
- Fails if wheelhouse is empty
- Lists built wheels for visibility

### 4. Artifact Upload
- Unique artifact names per platform
- `if-no-files-found: error` ensures nothing is missed
- 30-day retention for production builds

---

## Publishing Process

### GitHub Release

**Happens when:**
- All platform builds succeed
- Running on a version tag

**Creates:**
- GitHub release with all wheels attached
- Auto-generated release notes
- Marked as prerelease if version indicates it
- Custom body with installation instructions

### PyPI Publishing

**Happens when:**
- GitHub release succeeds
- Running on a version tag

**Process:**
1. Downloads all wheels (12 expected: 3 Python versions × 4 platforms)
2. Validates with `twine check`
3. Publishes all wheels in single operation
4. Uses `--skip-existing` flag (prevents errors if file exists)
5. Verifies upload by checking PyPI

**Safety features:**
- Separate `environment` for PyPI (can add approval rules)
- Verbose output for debugging
- Post-upload verification

---

## Troubleshooting

### Build Failures

#### Pre-check fails
```
Error: Neither setup.py nor pyproject.toml found!
```
**Solution:** Ensure your package has a proper build configuration file.

#### Version already exists on PyPI
```
Warning: Version 1.0.0 already exists on PyPI
```
**Solution:** This is usually fine - `--skip-existing` handles it. If you need to replace it:
1. Use a new version number (preferred), or
2. Delete the release from PyPI (not recommended)

#### Missing PyPI token
```
Error: PYPI_API_TOKEN secret is not set!
```
**Solution:** Add your PyPI token to GitHub secrets.

### Platform-Specific Failures

#### Linux: Import test fails
```
ImportError: libsomething.so.1: cannot open shared object file
```
**Solution:** Check `CIBW_BEFORE_BUILD` includes all required dependencies.

#### macOS: Architecture mismatch
```
Error: Wrong architecture
```
**Solution:** Check `CIBW_ARCHS_MACOS` matches the runner (macos-12 = x86_64, macos-14 = arm64).

#### Windows: Compilation errors
```
Error: MSVC compiler not found
```
**Solution:** Ensure Cython is installed in `CIBW_BEFORE_BUILD`.

### Publishing Failures

#### Race condition (shouldn't happen anymore)
```
Error: File already exists
```
**Solution:** This shouldn't happen with the new workflow. The `--skip-existing` flag handles it.

#### Rate limiting
```
Error: Too many requests
```
**Solution:** Wait a few minutes and re-run the failed job.

---

## Best Practices

### Version Numbering

**Good:**
- `v1.0.0` - Major release
- `v1.2.3` - Minor update
- `v2.0.0-beta.1` - Beta release
- `v1.0.0-rc.1` - Release candidate

**Avoid:**
- `v1.0` - Not specific enough
- `release-1.0.0` - Won't trigger workflow
- `1.0.0` - Missing 'v' prefix

### Release Process

1. **Always test first:**
   - Use test workflow or PR to verify builds
   - Check that imports work
   - Verify on at least one platform

2. **Use prerelease versions:**
   - Test with `v1.0.0-beta.1` first
   - This marks GitHub release as prerelease
   - Easier to fix issues before final release

3. **Monitor the workflow:**
   - Don't walk away after pushing tag
   - Watch for failures in real-time
   - Fix issues quickly if they occur

4. **Verify the release:**
   - Check GitHub release was created
   - Verify on PyPI: `https://pypi.org/project/q-store/VERSION/`
   - Test installation: `pip install q-store==VERSION`

### Security

**DO:**
- ✓ Use `secrets.PYPI_API_TOKEN` for credentials
- ✓ Limit token scope to single project
- ✓ Use environment protection rules for production
- ✓ Rotate tokens periodically

**DON'T:**
- ✗ Hardcode tokens in workflow files
- ✗ Use personal access tokens
- ✗ Share tokens across multiple projects
- ✗ Commit tokens to repository

---

## Migration from Old Workflows

### Step 1: Backup
```bash
mkdir .github/workflows/old
mv .github/workflows/build-*.yml .github/workflows/old/
```

### Step 2: Add New Workflows
```bash
# Copy the new workflows
cp build-wheels-improved.yml .github/workflows/
cp build-wheels-test.yml .github/workflows/
```

### Step 3: Test
1. Create a test branch
2. Make a trivial change
3. Open PR to trigger test workflow
4. Verify builds succeed

### Step 4: Production Test
1. Create a prerelease version: `v1.0.0-test.1`
2. Monitor workflow
3. Verify GitHub release and PyPI upload
4. Delete test release and version if desired

### Step 5: Cleanup
```bash
# If everything works
rm -rf .github/workflows/old/
```

---

## Advanced Configuration

### Adding More Python Versions

Edit `CIBW_BUILD` in each build job:

```yaml
env:
  CIBW_BUILD: cp311-* cp312-* cp313-* cp314-*  # Add Python 3.14
```

### Adding More Architectures

For Linux ARM64:

```yaml
build_linux_arm:
  runs-on: ubuntu-22.04
  steps:
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3
    - name: Build wheels
      env:
        CIBW_ARCHS_LINUX: "aarch64"
```

### Custom Test Commands

Modify `CIBW_TEST_COMMAND`:

```yaml
env:
  CIBW_TEST_COMMAND: >-
    python -c "from q_store import QuantumDatabase; db = QuantumDatabase(); print('OK')"
```

### Environment-Specific Publishing

Add environment protection:

```yaml
publish_pypi:
  environment:
    name: pypi-production
    url: https://pypi.org/project/q-store/${{ needs.pre_checks.outputs.version }}
  # Now requires manual approval in GitHub
```

---

## Metrics and Monitoring

### Build Time Estimates

- **Pre-checks:** 2-3 minutes
- **Linux build:** 15-20 minutes
- **macOS Intel:** 20-25 minutes
- **macOS ARM64:** 20-25 minutes
- **Windows:** 25-35 minutes
- **Publishing:** 2-3 minutes

**Total time:** ~25-35 minutes (parallel builds)

### Cost Estimates (GitHub Actions)

- Linux: Included in free tier
- Windows: 2x minutes
- macOS: 10x minutes

For a full release (all platforms):
- ~150 macOS minutes (most expensive)
- ~35 Windows minutes
- ~20 Linux minutes

**Tip:** Test on Linux first (cheapest, fastest)

---

## Summary of Improvements

| Issue | Old Behavior | New Behavior |
|-------|-------------|--------------|
| **Multiple PyPI publishes** | Each workflow published independently | Single publish job after all succeed |
| **No validation** | Builds started immediately | Pre-checks validate everything first |
| **Tag reuse** | Can't reuse tag if one platform fails | Builds are independent; tag safe |
| **Release conflicts** | Multiple workflows created releases | Single release with all artifacts |
| **Error visibility** | Unclear which platform failed | Clear summary with status of each |
| **Testing** | Had to publish to test | Separate test workflow |
| **Race conditions** | Files could conflict on PyPI | Atomic publish with --skip-existing |
| **Wasted resources** | Built all platforms even if metadata invalid | Pre-checks catch issues early |

---

## Support and Feedback

For issues or questions:
1. Check this documentation first
2. Review GitHub Actions logs
3. Check the workflow summary page
4. Create an issue in the repository

---

## Changelog

### Version 2.0 (Current)
- ✨ Added comprehensive pre-check validations
- ✨ Unified publishing (single PyPI upload job)
- ✨ Independent platform builds
- ✨ Test workflow for development
- ✨ Better error messages and summaries
- 🐛 Fixed race conditions in PyPI publishing
- 🐛 Fixed duplicate GitHub releases
- 📝 Added comprehensive documentation

### Version 1.0 (Old)
- Basic cibuildwheel integration
- Separate workflows per platform
- Independent publishing (caused issues)
