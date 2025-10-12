---
title: Contributing Guide
description: How to contribute to PySuricata development
---

# Contributing to PySuricata

Thank you for considering contributing to PySuricata! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.9+ 
- `uv` package manager (recommended) or `pip`
- Git

### Clone Repository

```bash
git clone https://github.com/alvarodiez20/pysuricata.git
cd pysuricata
```

### Install Dependencies

=== "Using uv (recommended)"
    ```bash
    uv sync --dev
    uv run python -c "import pysuricata; print('Success!')"
    ```

=== "Using pip"
    ```bash
    pip install -e ".[dev]"
    python -c "import pysuricata; print('Success!')"
    ```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=pysuricata --cov-report=html

# Run specific test file
uv run pytest tests/test_numeric.py

# Run tests in parallel
uv run pytest -n auto
```

## Code Style

PySuricata uses **Ruff** for linting and formatting.

```bash
# Format code
uv run ruff format pysuricata/

# Check linting
uv run ruff check pysuricata/

# Auto-fix issues
uv run ruff check --fix pysuricata/
```

### Style Guidelines

- Follow PEP 8
- Line length: 88 characters (Black-style)
- Use type hints for function signatures
- Docstrings: Google style

Example:

```python
def compute_mean(values: np.ndarray) -> float:
    """Compute arithmetic mean of values.
    
    Args:
        values: Array of numeric values
        
    Returns:
        Mean value
        
    Raises:
        ValueError: If array is empty
    """
    if len(values) == 0:
        raise ValueError("Cannot compute mean of empty array")
    return float(np.mean(values))
```

## Documentation

### Build Documentation Locally

```bash
# Install docs dependencies
uv sync --dev

# Build docs
uv run mkdocs serve

# Open http://localhost:8000 in browser
```

### Documentation Style

- Use clear, concise language
- Include code examples
- Add mathematical formulas for algorithms
- Link to related pages
- Update relevant sections when changing code

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 2. Make Changes

- Write tests for new functionality
- Update documentation
- Follow code style guidelines
- Keep commits atomic and well-described

### 3. Run Checks

```bash
# Format
uv run ruff format pysuricata/

# Lint
uv run ruff check pysuricata/

# Test
uv run pytest

# Type check (if using mypy)
uv run mypy pysuricata/

# Build docs
uv run mkdocs build --strict
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add support for XYZ"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `refactor:` - Code refactoring
- `test:` - Test updates
- `chore:` - Build/tooling changes

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots for UI changes
- Checklist of completed items

## Testing Guidelines

### Unit Tests

Test individual functions/classes in isolation.

```python
def test_welford_mean():
    """Test Welford mean computation"""
    from pysuricata.accumulators.algorithms import StreamingMoments
    
    moments = StreamingMoments()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    for v in values:
        moments.update(np.array([v]))
    
    result = moments.finalize()
    assert abs(result["mean"] - 3.0) < 1e-10
```

### Integration Tests

Test components working together.

```python
def test_full_profile():
    """Test end-to-end profiling"""
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    report = profile(df)
    
    assert report.html is not None
    assert len(report.stats["columns"]) == 2
```

### Property-Based Tests

Use hypothesis for randomized testing.

```python
from hypothesis import given
from hypothesis.strategies import lists, floats

@given(lists(floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_welford_matches_numpy(values):
    """Welford should match NumPy"""
    moments = StreamingMoments()
    for v in values:
        moments.update(np.array([v]))
    
    result = moments.finalize()
    expected = np.mean(values)
    
    assert abs(result["mean"] - expected) < 1e-6
```

## Architecture Overview

```
pysuricata/
├── api.py              # Public API (profile, summarize)
├── report.py           # Report generation orchestration
├── config.py           # Configuration classes
├── accumulators/       # Streaming accumulators
│   ├── numeric.py      # Numeric statistics
│   ├── categorical.py  # Categorical analysis
│   ├── datetime.py     # Temporal analysis
│   └── boolean.py      # Boolean analysis
├── compute/            # Data processing
│   ├── adapters/       # pandas/polars adapters
│   ├── analysis/       # Correlations, metrics
│   └── processing/     # Chunking, inference
├── render/             # HTML generation
│   ├── *_card.py       # Variable type cards
│   ├── html.py         # Main template
│   └── svg_utils.py    # SVG charts
└── templates/          # HTML templates
```

## Adding New Features

### Add New Statistic to Numeric Analysis

1. **Update accumulator** (`pysuricata/accumulators/numeric.py`):
```python
class NumericAccumulator:
    def __init__(self, ...):
        self._new_stat = 0  # Add state
    
    def update(self, values):
        # Update new statistic
        self._new_stat += some_computation(values)
    
    def finalize(self):
        return NumericSummary(
            ...
            new_stat=self._new_stat  # Include in summary
        )
```

2. **Update summary dataclass** (`pysuricata/accumulators/numeric.py`):
```python
@dataclass
class NumericSummary:
    ...
    new_stat: float = 0.0
```

3. **Update renderer** (`pysuricata/render/numeric_card.py`):
```python
def render_card(self, stats):
    # Add new_stat to HTML
    html += f"<div>New Stat: {stats.new_stat:.2f}</div>"
```

4. **Add tests** (`tests/test_numeric.py`):
```python
def test_new_stat():
    acc = NumericAccumulator("test")
    acc.update(np.array([1, 2, 3]))
    summary = acc.finalize()
    assert summary.new_stat == expected_value
```

5. **Update documentation** (`docs/stats/numeric.md`):
```markdown
### New Statistic

Mathematical definition:
\[
\text{NewStat} = \sum_{i=1}^{n} f(x_i)
\]

Interpretation: ...
```

## Release Process

(For maintainers only)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.x.y`
4. Push tag: `git push origin v0.x.y`
5. CI/CD automatically builds and publishes to PyPI

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Assume good intentions

## Getting Help

- 💬 [GitHub Discussions](https://github.com/alvarodiez20/pysuricata/discussions)
- 🐛 [GitHub Issues](https://github.com/alvarodiez20/pysuricata/issues)
- 📧 Email: alvarodiez20@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to PySuricata! 🎉




