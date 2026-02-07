# Contributing to pyBLM

## Development Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

```bash
black iblm/ tests/ examples/
flake8 iblm/ tests/ examples/
```

## Building and Publishing

```bash
# Install build tools
pip install build twine

# Build
python -m build

# Upload
twine upload dist/*
```
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-<your-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-test-token>
```

#### Step 3: Upload to TestPyPI first

```bash
# Upload to test repository
twine upload -r testpypi dist/*

# Test installation from TestPyPI
pip install -i https://test.pypi.org/simple/ pyBLM
```

#### Step 4: Upload to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Test installation
pip install pyBLM
```

### Version Management

Update version in:
1. `setup.py` - `version` parameter
2. `iblm/__init__.py` - `__version__` variable

Version format: `MAJOR.MINOR.PATCH` (semantic versioning)

## Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run all tests
- [ ] Run code quality checks (black, flake8)
- [ ] Create git tag: `git tag -a v0.1.0 -m "Release 0.1.0"`
- [ ] Push tag: `git push origin v0.1.0`
- [ ] Build distribution: `python -m build`
- [ ] Test build locally
- [ ] Upload to TestPyPI
- [ ] Test installation from TestPyPI
- [ ] Upload to PyPI
- [ ] Verify on PyPI.org
- [ ] Create GitHub Release

## Additional Resources

- [PyPI Help](https://pypi.org/help/)
- [setuptools Documentation](https://setuptools.pypa.io/)
- [twine Documentation](https://twine.readthedocs.io/)
- [PEP 427 - Wheel Binary Package Format](https://www.python.org/dev/peps/pep-0427/)
- [PEP 517 - A build-backend interface for Python source trees](https://www.python.org/dev/peps/pep-0517/)
