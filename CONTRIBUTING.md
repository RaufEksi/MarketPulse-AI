# Contributing to MarketPulse AI

## Development Workflow

### 1. Branch Naming

Use conventional branch names:

```
feat/<issue-number>-<short-description>   # Feature
bugfix/<issue-number>-<short-description> # Bug fix
hotfix/<short-description>                # Critical production fix
refactor/<short-description>              # Code refactoring
docs/<short-description>                  # Documentation
```

Example:
```
git checkout -b feat/2-1-alpaca-api-connector
git checkout -b bugfix/3-2-atr-calculation-bug
```

### 2. Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Example:
```
feat(data-engine): implement Alpaca API connector with retry logic

- Add AlpacaClient wrapper class
- Implement exponential backoff retry (max 3 attempts)
- Add rate limiting (200 req/min)
- Include comprehensive error handling

Fixes #2-1
```

**Types**: feat, fix, docs, style, refactor, perf, test, ci, chore

### 3. Pull Request Process

1. **Create branch** from `main`
2. **Make changes** with atomic commits
3. **Write tests** (min 80% coverage increase)
4. **Run linting & tests locally**:
   ```bash
   make lint
   make test
   ```
5. **Push** to GitHub
6. **Create PR** with template (auto-generated)
7. **Link issue**: Add `Fixes #<issue-number>` in PR description
8. **Request review** from team members
9. **Address feedback** with new commits
10. **Merge** after approval & CI passes

### 4. Code Standards

#### Python Style
- **Formatter**: Black (line length: 100)
- **Linter**: Flake8, mypy (strict type hints)
- **Docstrings**: Google style

```bash
make format   # auto-format code
make lint     # check style & types
```

#### Example Function

```python
def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR) for OHLCV data.
    
    Args:
        highs: Array of high prices [N]
        lows: Array of low prices [N]
        closes: Array of close prices [N]
        period: EMA period (default: 14)
    
    Returns:
        Array of ATR values [N]
    
    Raises:
        ValueError: If arrays have mismatched lengths or period > len(data)
    
    Example:
        >>> highs = np.array([100, 101, 102])
        >>> lows = np.array([99, 100, 101])
        >>> closes = np.array([99.5, 100.5, 101.5])
        >>> atr = compute_atr(highs, lows, closes, period=14)
    """
    # Implementation
    pass
```

### 5. Testing Requirements

**Unit Tests**:
- Test individual functions with mocked dependencies
- Location: `tests/test_<module>/`
- Naming: `test_<function>_<scenario>.py`

**Integration Tests**:
- Test end-to-end workflows
- Location: `tests/integration/`
- Mock external APIs (Alpaca, Reddit, GDELT)

**Example**:

```python
import pytest
from src.feature_engineering.time_series_features import compute_atr

def test_atr_calculation_basic():
    """Test ATR calculation with known values."""
    highs = np.array([100, 101, 102, 101, 100])
    lows = np.array([99, 100, 101, 100, 99])
    closes = np.array([99.5, 100.5, 101.5, 100.5, 99.5])
    
    atr = compute_atr(highs, lows, closes, period=2)
    
    assert atr.shape == highs.shape
    assert not np.isnan(atr[2:]).any()  # First 2 values may be NaN
    assert atr.min() >= 0

def test_atr_insufficient_data():
    """Test ATR with insufficient data."""
    with pytest.raises(ValueError, match="period"):
        compute_atr(np.array([100]), np.array([99]), np.array([99.5]), period=14)
```

### 6. Documentation

**For new features**:
- Add docstring (Google style)
- Update relevant `.md` file in `docs/`
- Add code examples
- Update README if user-facing

**For breaking changes**:
- Add migration guide
- Document deprecation timeline
- Provide upgrade examples

### 7. Local Development Setup

```bash
# Clone
git clone https://github.com/RaufEksi/MarketPulse-AI.git
cd MarketPulse-AI

# Setup environment
make setup

# Activate venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dev dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run local services
make serve
```

### 8. Pre-Commit Hooks

Automatically format & lint before commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Issue Labels & Triage

| Label | Meaning |
|-------|----------|
| `bug` | Something is broken |
| `enhancement` | Feature request |
| `documentation` | Docs need update |
| `data-science` | ML/data work |
| `backend` | API/infrastructure |
| `devops` | Deployment/monitoring |
| `high-priority` | Critical for milestone |
| `good-first-issue` | For newcomers |

## Code Review Checklist

**Reviewer**: Verify before approving:

- [ ] Code follows project style guide
- [ ] Tests added/updated
- [ ] Coverage maintained/increased
- [ ] Documentation updated
- [ ] No hardcoded values (use config)
- [ ] Error handling implemented
- [ ] Performance acceptable
- [ ] No security issues
- [ ] Commit history clean

## Reporting Issues

Use GitHub issue template. Include:

1. **Clear title** (what's broken or needed)
2. **Description** (context & impact)
3. **Steps to reproduce** (for bugs)
4. **Expected vs actual behavior**
5. **Environment** (OS, Python version, etc.)
6. **Screenshots/logs** (if applicable)

## Questions?

Open a [GitHub Discussion](https://github.com/RaufEksi/MarketPulse-AI/discussions) or ask in PR reviews.

---

**Happy Contributing! 🚀**
