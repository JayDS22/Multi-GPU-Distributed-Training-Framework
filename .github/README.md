# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for automated testing and deployment.

## Workflows

### 1. `ci-cd.yml` - Main CI/CD Pipeline
**Triggers:** Push to main/develop, Pull Requests, Releases

**Jobs:**
- **Code Quality** - Linting, formatting checks (non-blocking)
- **Unit Tests** - Python 3.8, 3.9, 3.10 compatibility
- **Integration Tests** - End-to-end testing (CPU only in CI)
- **Quick Functional Test** - Smoke test with dummy data
- **Docker Build** - Container image verification
- **Deploy** - Staging/Production deployment (simulated in CI)

### 2. `tests.yml` - Comprehensive Testing
**Triggers:** Push, Pull Requests

**Jobs:**
- Unit tests across Python versions
- Integration tests (CPU-based)
- Performance tests (requires GPU, self-hosted)
- PyTorch compatibility tests
- Docker build verification

### 3. `security.yml` - Security Scanning
**Triggers:** Push, Pull Requests, Daily schedule

**Jobs:**
- Dependency vulnerability scanning
- Code security analysis (Bandit)
- Secret scanning (TruffleHog)
- Docker image security (Trivy)
- CodeQL analysis
- License compliance

## CI/CD Behavior

### Tests Run on CPU
Most tests run on GitHub-hosted runners (Ubuntu, CPU only):
- ✅ Import tests
- ✅ Configuration tests
- ✅ Data loading tests
- ✅ Single-process unit tests
- ⚠️ GPU tests are skipped (marked with `@pytest.mark.gpu`)
- ⚠️ Multi-GPU tests are skipped (requires self-hosted runners)

### What Gets Tested
1. **Code compiles** - All imports work
2. **Functionality** - Core features work on CPU
3. **Data pipeline** - Synthetic data generation
4. **Configuration** - Config loading and validation
5. **Docker** - Image builds successfully
6. **Security** - No vulnerabilities or secrets

### What Doesn't Get Tested in CI
- GPU-specific features (FSDP, mixed precision)
- Multi-GPU distributed training
- Performance benchmarks
- Kubernetes deployment (no cluster in CI)

## Local Testing

### Run All Tests
```bash
# Run everything
pytest

# Skip GPU tests
pytest -m "not gpu"

# Skip slow tests
pytest -m "not slow"

# Specific test file
pytest tests/test_distributed.py -v
```

### Run CI Locally
```bash
# Install act (GitHub Actions locally)
brew install act  # or: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run workflows
act push  # Simulate push event
act pull_request  # Simulate PR
```

## Configuration

### Required Secrets (for actual deployment)
- `DOCKERHUB_USERNAME` - Docker Hub username
- `DOCKERHUB_TOKEN` - Docker Hub access token
- `SLACK_WEBHOOK` - Slack webhook URL (optional)

### Self-Hosted Runner Setup (for GPU tests)
```bash
# Install runner on GPU machine
# Settings → Actions → Runners → New self-hosted runner

# Add labels
./config.sh --labels gpu
```

## Troubleshooting

### Tests Failing in CI
**Common issues:**

1. **Import errors**
   - Ensure all dependencies in `requirements.txt`
   - Run `pip install -e .` before tests

2. **GPU tests failing**
   - Mark GPU tests: `@pytest.mark.gpu`
   - CI runs with: `pytest -m "not gpu"`

3. **Timeout errors**
   - Increase timeout in `pytest.ini`
   - Or add `@pytest.mark.timeout(600)` to slow tests

4. **Data not found**
   - Generate dummy data first: `python scripts/generate_dummy_data.py`
   - Or use synthetic data (automatic fallback)

### Fixing CI/CD

The updated `ci-cd.yml` now:
- ✅ Generates dummy data before tests
- ✅ Skips GPU-only tests
- ✅ Uses `continue-on-error: true` for optional checks
- ✅ Runs quick functional test
- ✅ Has proper timeouts

## Status Badges

Add to your README.md:

```markdown
![CI/CD](https://github.com/USERNAME/REPO/workflows/CI/CD%20Pipeline/badge.svg)
![Tests](https://github.com/USERNAME/REPO/workflows/Tests/badge.svg)
![Security](https://github.com/USERNAME/REPO/workflows/Security%20Scanning/badge.svg)
```

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Workflows run automatically
3. ✅ Check Actions tab for results
4. ⚠️ Some tests may be skipped (GPU tests)
5. ✅ All CPU tests should pass

The CI/CD is designed to verify the code works, even without GPU access!
