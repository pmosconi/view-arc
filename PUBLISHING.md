# Publishing to PyPI

This document describes how to publish `actvalue.view-arc` to PyPI.

## Prerequisites

1. Install dev dependencies (includes `twine`):
   ```bash
   make install
   ```

2. Set up PyPI credentials. Choose one method:

   **Option A: API Token (Recommended)**
   - Create a token at https://pypi.org/manage/account/token/
   - Add to `~/.pypirc`:
     ```ini
     [pypi]
     username = __token__
     password = pypi-YourTokenHere
     ```

   **Option B: Environment Variables**
   ```bash
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=pypi-YourTokenHere
   ```

## Pre-Release Checklist

1. **Update version** in [pyproject.toml](pyproject.toml):
   ```toml
   version = "0.1.0"
   ```

2. **Run quality checks**:
   ```bash
   make check  # Runs mypy + tests + performance baseline
   ```

3. **Clean previous builds**:
   ```bash
   make clean-build
   ```

4. **Verify dependencies** are minimal (only numpy for production):
   ```bash
   grep -A 3 'dependencies =' pyproject.toml
   ```

## Publishing Process

### 1. Build Distribution Packages

```bash
make build
```

This creates:
- `dist/view_arc-X.Y.Z-py3-none-any.whl` (wheel)
- `dist/view-arc-X.Y.Z.tar.gz` (source distribution)

### 2. Test on TestPyPI (Optional but Recommended)

```bash
make publish-test
```

Then test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ actvalue.view-arc
```

### 3. Publish to PyPI

```bash
make publish
```

### 4. Verify Installation

```bash
pip install actvalue.view-arc

# Test basic import
python -c "from view_arc import compute_attention_seconds; print('Success!')"
```

## Package Extras

Users can install optional dependencies:

```bash
# For visualization functions
pip install actvalue.view-arc[viz]

# For running examples
pip install actvalue.view-arc[examples]

# For development
pip install actvalue.view-arc[dev]
```

## Notes

- The core library only requires `numpy` - it's lightweight!
- `opencv-python` and `scikit-image` are optional (used only for visualization/examples)
- Visualization functions will raise helpful errors if cv2 is not installed
- Examples require `[examples]` extra to run
