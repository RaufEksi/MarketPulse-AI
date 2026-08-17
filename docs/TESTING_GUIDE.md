# MarketPulse AI — Testing Guide 🧪

This guide covers the testing strategy, conventions, tools, and patterns used in MarketPulse AI to maintain ≥80% code coverage.

---

## 1. Testing Stack

| Tool | Version | Purpose |
|------|---------|---------|
| **pytest** | ≥7.4 | Test runner & framework |
| **pytest-cov** | ≥4.1 | Coverage reporting |
| **pytest-asyncio** | ≥0.21 | Async test support (FastAPI) |
| **pytest-benchmark** | ≥4.0 | Performance benchmarking |
| **responses** | ≥0.23 | HTTP request mocking |
| **unittest.mock** | stdlib | General purpose mocking |

---

## 2. Running Tests

### Quick Commands

```bash
# Run all tests
make test

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run specific test class/method
pytest tests/test_models.py::TestMarketPulseNet::test_forward_pass -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

# Run only unit tests
pytest tests/ -m "unit" -v

# Run only integration tests
pytest tests/ -m "integration" -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Skip GPU tests (on CPU machines)
pytest tests/ -m "not gpu" -v
```

### Coverage Report

After running `make test`, view the HTML coverage report:

```bash
# Open in browser
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux
```

**Target**: ≥80% line coverage across `src/`.

---

## 3. Test File Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures (settings, mock data, etc.)
├── test_config.py                 # Tests for src/config/
├── test_feature_engineering.py    # Tests for src/feature_engineering/
├── test_alignment.py              # Tests for src/data_alignment/
├── test_models.py                 # Tests for src/models/
├── test_api.py                    # Tests for src/api/
├── test_data_engine.py            # Tests for src/data_engine/ (future)
├── test_xai.py                    # Tests for src/xai_explainer/ (future)
└── integration/                   # End-to-end integration tests (future)
    └── test_pipeline.py
```

### Naming Convention

| Element | Pattern | Example |
|---------|---------|---------|
| Test file | `test_<module>.py` | `test_models.py` |
| Test class | `Test<ClassName>` | `TestMarketPulseNet` |
| Test method | `test_<function>_<scenario>` | `test_forward_pass_correct_shape` |

---

## 4. Test Markers

Use pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_atr_calculation():
    """Fast, isolated unit test."""
    ...

@pytest.mark.integration
def test_full_pipeline():
    """Tests multiple modules together."""
    ...

@pytest.mark.slow
def test_finbert_embedding_large_batch():
    """Takes >10 seconds to run."""
    ...

@pytest.mark.gpu
def test_model_training_cuda():
    """Requires CUDA GPU."""
    ...
```

Markers are configured in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow running tests",
    "gpu: GPU-dependent tests",
]
```

---

## 5. Fixtures (`conftest.py`)

Shared test fixtures are defined in `tests/conftest.py`. Use these instead of creating your own test data in each test file.

### Available Fixtures

```python
# Sample OHLCV DataFrame
@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Returns a small OHLCV DataFrame for testing."""
    ...

# Sample settings (no .env required)
@pytest.fixture
def mock_settings() -> Settings:
    """Returns Settings with test defaults, no API keys needed."""
    ...

# Small PyTorch tensors for model tests
@pytest.fixture
def sample_price_tensor() -> torch.Tensor:
    """Returns [2, 78, 16] tensor simulating a batch of price sequences."""
    return torch.randn(2, 78, 16)

@pytest.fixture
def sample_text_embedding() -> torch.Tensor:
    """Returns [2, 768] tensor simulating FinBERT embeddings."""
    return torch.randn(2, 768)
```

### Adding New Fixtures

When adding fixtures, follow this pattern:

```python
@pytest.fixture
def fixture_name() -> ReturnType:
    """Clear description of what this fixture provides."""
    # Setup
    data = create_test_data()
    yield data
    # Teardown (optional)
    cleanup()
```

---

## 6. Test Patterns

### Pattern 1: Arrange-Act-Assert (AAA)

```python
def test_atr_basic_computation(self) -> None:
    """Test ATR produces correct shape and non-negative values."""
    # Arrange — setup test data and dependencies
    engine = TechnicalFeatureEngine()
    df = create_sample_ohlcv(rows=100)

    # Act — execute the function under test
    result = engine.compute_all_features(df)

    # Assert — verify expected outcomes
    assert "atr_14" in result.columns
    assert result["atr_14"].notna().sum() > 0
    assert (result["atr_14"].dropna() >= 0).all()
```

### Pattern 2: Exception Testing

```python
def test_atr_insufficient_data_raises(self) -> None:
    """Test ATR raises ValueError when period > data length."""
    engine = TechnicalFeatureEngine()
    short_df = create_sample_ohlcv(rows=5)

    with pytest.raises(ValueError, match="period"):
        engine.compute_atr(short_df, period=14)
```

### Pattern 3: Mocking External APIs

```python
from unittest.mock import MagicMock, patch

@patch("src.data_engine.alpaca_connector.REST")
def test_alpaca_fetch_bars(mock_rest_class) -> None:
    """Test AlpacaDataCollector with mocked API response."""
    # Arrange — configure the mock
    mock_client = MagicMock()
    mock_rest_class.return_value = mock_client
    mock_client.get_bars.return_value = create_mock_bars_response()

    collector = AlpacaDataCollector(api_key="test", secret_key="test")

    # Act
    df = collector.fetch_bars("SPY", start="2026-08-01", end="2026-08-07")

    # Assert
    assert len(df) > 0
    assert "close" in df.columns
    mock_client.get_bars.assert_called_once()
```

### Pattern 4: PyTorch Model Testing

```python
def test_marketpulsenet_forward_pass(
    sample_price_tensor: torch.Tensor,
    sample_text_embedding: torch.Tensor,
) -> None:
    """Test MarketPulseNet produces correct output shape."""
    # Arrange
    model = MarketPulseNet(
        ts_input_dim=16,
        text_input_dim=768,
        hidden_dim=128,
        num_heads=4,
    )
    model.eval()

    # Act
    with torch.no_grad():
        output = model(sample_price_tensor, sample_text_embedding)

    # Assert
    assert output.shape == (2, 1)  # [batch_size, 1] for binary classification
    assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output
```

### Pattern 5: FastAPI Endpoint Testing

```python
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint() -> None:
    """Test /health returns 200 with status info."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_predict_endpoint_validation() -> None:
    """Test /predict rejects invalid request body."""
    response = client.post("/predict", json={"invalid": "data"})

    assert response.status_code == 422  # Pydantic validation error
```

### Pattern 6: Parameterized Testing

```python
@pytest.mark.parametrize("period,expected_nans", [
    (14, 13),   # ATR(14) has 13 leading NaN values
    (7, 6),     # ATR(7) has 6 leading NaN values
    (1, 0),     # ATR(1) has no leading NaN values
])
def test_atr_nan_count(period: int, expected_nans: int) -> None:
    """Test ATR produces expected number of leading NaN values."""
    engine = TechnicalFeatureEngine()
    df = create_sample_ohlcv(rows=100)
    atr = engine.compute_atr(df, period=period)

    actual_nans = atr.isna().sum()
    assert actual_nans == expected_nans
```

---

## 7. What to Test per Module

| Module | What to Test | Mock What? |
|--------|-------------|------------|
| `src/config/` | Settings loading, YAML parsing, env override | File system (optional) |
| `src/data_engine/` | Connector fetch, data validation, error handling | External APIs (Alpaca, Reddit, NewsAPI) |
| `src/feature_engineering/` | Indicator computation, NaN handling, shape correctness | FinBERT model (use dummy embeddings) |
| `src/data_alignment/` | Decay calculation, dataset construction, split correctness | Nothing (pure computation) |
| `src/models/` | Forward pass shapes, loss computation, training step | Dataset (use synthetic tensors) |
| `src/xai_explainer/` | Attribution output format, feature ranking | Trained model (use simple mock model) |
| `src/api/` | Endpoint responses, validation errors, status codes | Model inference (return dummy prediction) |
| `src/utils/` | Logger format, exception hierarchy, metric calculations | Nothing (pure utilities) |

---

## 8. Test Coverage Policy

- **Minimum threshold**: 80% line coverage (enforced in `pyproject.toml`)
- **New code**: Every new public function must have at least one test
- **Bug fixes**: Each bug fix must include a regression test
- **Exclude from coverage**: `__repr__`, `if __name__ == "__main__"`, `TYPE_CHECKING` blocks

Coverage configuration in `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__pycache__/*", "*/.venv/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
min_percentage = 80
```

---

## 9. CI Integration

Tests run automatically on every push and PR via GitHub Actions:

```yaml
# .github/workflows/ci.yml
- name: Run Pytest Suite with Coverage
  run: |
    pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing -v
```

**CI will fail if**:
- Any test fails
- Coverage drops below 80%
- Black/isort formatting is incorrect
- Flake8 reports errors
- Bandit finds security issues
