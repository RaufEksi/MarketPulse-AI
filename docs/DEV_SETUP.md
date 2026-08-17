# MarketPulse AI — Developer Setup Guide 🛠️

This guide provides step-by-step instructions for setting up the MarketPulse AI development environment, including Python version management, dependency installation, and common troubleshooting.

---

## Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10 or 3.11 | Runtime (3.12+ has compatibility issues) |
| **Git** | Latest | Version control |
| **Docker** | Latest (optional) | Containerized deployment |
| **pyenv** | Latest (recommended) | Python version management |
| **make** | GNU Make | Task automation |

---

## 1. Python Version Setup (pyenv)

> [!WARNING]
> This project requires **Python 3.10 or 3.11**. Python 3.12+ has known compatibility issues with pinned dependencies (`numpy==1.24.3`, `torch==2.0.1`). If your system Python is 3.12+, use **pyenv** to install a compatible version.

### Install pyenv

```bash
# Linux (Ubuntu/Debian)
curl https://pyenv.run | bash

# Add to shell profile (~/.bashrc, ~/.zshrc, or ~/.config/fish/config.fish)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Restart shell
exec $SHELL
```

### Install Python 3.11

```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# Install Python 3.11
pyenv install 3.11.9

# Set as local version for this project
cd /path/to/MarketPulse-AI
pyenv local 3.11.9

# Verify
python --version  # Should output: Python 3.11.9
```

---

## 2. Project Setup

### Option A: Using Make (Recommended)

```bash
# Clone repository
git clone https://github.com/RaufEksi/MarketPulse-AI.git
cd MarketPulse-AI

# Ensure correct Python version
python --version  # Must be 3.10.x or 3.11.x

# Full setup (creates venv, installs all dependencies)
make setup

# Activate virtual environment
source venv/bin/activate     # bash/zsh
source venv/bin/activate.fish  # fish
```

### Option B: Manual Setup

```bash
# Clone
git clone https://github.com/RaufEksi/MarketPulse-AI.git
cd MarketPulse-AI

# Create virtual environment
python3.11 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install all dependencies
pip install -r requirements.txt

# Install project in editable mode with dev extras
pip install -e ".[dev]"
```

---

## 3. Environment Configuration

```bash
# Copy template
cp .env.example .env

# Edit with your API keys
nano .env  # or vim, code, etc.
```

### Required API Keys

| Variable | Source | Required? |
|----------|--------|-----------|
| `ALPACA_API_KEY` | [Alpaca Markets](https://alpaca.markets/) | ✅ Yes (for real data) |
| `ALPACA_SECRET_KEY` | Alpaca Markets | ✅ Yes (for real data) |
| `REDDIT_CLIENT_ID` | [Reddit Apps](https://www.reddit.com/prefs/apps) | ⬜ Optional (for sentiment) |
| `REDDIT_CLIENT_SECRET` | Reddit Apps | ⬜ Optional (for sentiment) |

> [!TIP]
> For initial development and testing, API keys are **optional**. The data connectors have offline synthetic fallback generators, and tests use mocked data.

---

## 4. Verify Installation

```bash
# Run tests
make test
# or
pytest tests/ -v

# Check code quality
make lint

# Format code
make format

# Start API (Terminal 1)
make api

# Start Dashboard (Terminal 2)
make dashboard
```

---

## 5. IDE Configuration

### VS Code

Recommended extensions:
- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Black Formatter** (ms-python.black-formatter)

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=100"],
    "editor.formatOnSave": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}
```

### PyCharm

1. Set Project Interpreter → Select `.venv/bin/python`
2. Settings → Tools → Black → Line length: 100
3. Settings → Tools → External Tools → Add isort
4. Enable "Optimize imports on save"

---

## 6. Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

This automatically runs Black, isort, and Flake8 before each commit.

---

## 7. GPU Setup (Optional)

For FinBERT embedding generation and model training with GPU acceleration:

```bash
# Install CUDA-enabled PyTorch (check https://pytorch.org for your CUDA version)
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Update .env
DEVICE=cuda
```

> [!NOTE]
> GPU is **not required** for development. All modules fall back to CPU automatically. GPU primarily accelerates FinBERT batch embedding (Sprint 3) and model training (Sprint 4).

---

## 8. Common Issues & Solutions

### Issue: `numpy==1.24.3` fails to install with Python 3.14

**Error**: `BackendUnavailable: Cannot import 'setuptools.build_meta'`

**Cause**: `numpy 1.24.3` does not support Python 3.14. The `requirements.txt` uses pinned versions tested against Python 3.10/3.11.

**Solution**:
```bash
# Option 1 (Recommended): Use pyenv to install Python 3.11
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option 2: Use pyproject.toml minimum versions (less tested)
pip install -e "."
```

---

### Issue: `make setup` creates `venv/` but I created `.venv/`

**Cause**: The Makefile creates `venv/` by default, but you may have created `.venv/` manually.

**Solution**: Use one consistently. If you prefer `.venv/`:
```bash
rm -rf venv/  # Remove Makefile-created one if it exists
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

---

### Issue: `ta-lib` installation fails

**Cause**: `ta-lib` Python package requires the TA-Lib C library to be installed first.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install -y ta-lib

# macOS
brew install ta-lib

# Then retry
pip install ta-lib==0.4.28
```

If TA-Lib C library is unavailable, comment out `ta-lib==0.4.28` from `requirements.txt`. The project's `src/feature_engineering/technical_indicators.py` uses the `ta` library as primary implementation.

---

### Issue: `torch` download is very large (~2GB)

**Solution**: For CPU-only development, install the CPU-only wheel:
```bash
pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

---

## 9. Project Structure Quick Reference

```
MarketPulse-AI/
├── src/                  # Source code (importable package)
│   ├── config/           # Pydantic Settings + YAML
│   ├── data_engine/      # API connectors (Alpaca, Reddit, News)
│   ├── feature_engineering/  # Technical indicators, FinBERT, labeler
│   ├── data_alignment/   # Temporal alignment, PyTorch Dataset
│   ├── models/           # Bi-LSTM, TCN, MarketPulseNet, Trainer
│   ├── xai_explainer/    # SHAP, Integrated Gradients
│   ├── api/              # FastAPI backend
│   ├── dashboard/        # Streamlit frontend
│   └── utils/            # Logger, exceptions, metrics
├── tests/                # pytest test suite
├── config/               # YAML config files
├── data/                 # Raw & processed data (not versioned)
├── docs/                 # Documentation
├── scripts/              # Standalone utility scripts
├── deploy/               # Docker, K8s configs
├── .agents/              # AI IDE rules & configuration
├── .env.example          # Environment variables template
├── requirements.txt      # Pinned Python dependencies
├── pyproject.toml        # Project metadata & tool config
├── Makefile              # Development task automation
└── docker-compose.yml    # Local multi-container stack
```
