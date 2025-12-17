# GitHub Actions CI/CD for Q-Store

## Automated Wheel Building

The `.github/workflows/build-wheels.yml` workflow automatically builds binary wheels for multiple platforms.

### Supported Platforms

- **Linux:** x86_64 (manylinux)
- **Windows:** AMD64
- **macOS:** x86_64 (Intel) and arm64 (Apple Silicon)
- **Python Versions:** 3.11, 3.12, 3.13

### Triggers

1. **Version Tags:** Pushing a tag like `v3.4.0` triggers the full release workflow
2. **Manual:** Use "Run workflow" in GitHub Actions tab

### Release Process

1. **Build Wheels:** Builds wheels for all platform/Python combinations
2. **Create Release:** Automatically creates a GitHub Release with all wheels attached
3. **Publish to PyPI:** (Optional) Publishes wheels to PyPI

### Usage

#### Automated Release (Recommended)

```bash
# Update version in pyproject.toml to 3.4.1
# Commit changes
git add pyproject.toml
git commit -m "Bump version to 3.4.1"

# Create and push tag
git tag v3.4.1
git push origin v3.4.1
```

This will:
- Build wheels for all platforms
- Create a GitHub Release
- Attach all wheels to the release
- (Optional) Publish to PyPI

#### Manual Trigger

1. Go to GitHub Actions tab
2. Select "Build Wheels" workflow
3. Click "Run workflow"
4. Select branch and run

### PyPI Publishing

To enable PyPI publishing:

1. Create a PyPI API token at https://pypi.org/manage/account/token/
2. Add the token as a GitHub secret named `PYPI_API_TOKEN`:
   - Go to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI token (starts with `pypi-`)

To disable PyPI publishing, comment out or remove the `publish_pypi` job.

### Private Distribution

For closed-source distribution without PyPI:

1. Remove or disable the `publish_pypi` job
2. Wheels will be available in:
   - GitHub Release artifacts
   - Workflow run artifacts (for 30 days)

Customers can install directly from GitHub releases:
```bash
pip install https://github.com/YOUR_USERNAME/q-store/releases/download/v3.4.0/q_store-3.4.0-cp313-cp313-linux_x86_64.whl
```

### Testing

Each wheel is automatically tested after building with:
```python
from q_store import QuantumDatabase
print('Import successful!')
```

### Customization

Edit `.github/workflows/build-wheels.yml` to:
- Add/remove Python versions: Modify `matrix.python-version`
- Change platforms: Modify `matrix.os`
- Skip certain builds: Modify `CIBW_SKIP`
- Add more tests: Modify `CIBW_TEST_COMMAND`

### Troubleshooting

**Build failures:**
- Check the Actions tab for detailed logs
- Ensure all dependencies are in `pyproject.toml`
- Verify Cython can compile on all platforms

**Import errors during testing:**
- Check that all runtime dependencies are listed
- Verify compiled extensions work on target platform


**ARM64 macOS builds:**
- Requires macos-14 runner (GitHub-hosted ARM64)
- Cannot test on x86_64 runners
