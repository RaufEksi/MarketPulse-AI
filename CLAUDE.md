# MarketPulse AI — Claude Code Guide

This file is auto-loaded by Claude Code at the start of every session in this repo.

## What this project is

Multi-modal hybrid deep learning system that predicts short-term **volatility spikes**
(not price direction) by fusing 5-min OHLCV price data with financial sentiment
(FinBERT embeddings of Reddit/news text). See [README.md](README.md) for the full pitch.

## Project rules (read these — don't duplicate them here)

This repo already has a full AI-agent rulebook written for a previous AI IDE
(Antigravity/Gemini). It applies to Claude Code too. Read it before writing code:

@.agents/AGENTS.md
@.agents/rules/coding_standards.md
@.agents/rules/architecture_guard.md

Key points from those files, in one line each:
- Layer dependency direction is one-way: `data_engine → feature_engineering → data_alignment → models → xai_explainer → api → dashboard`. Never import upward (e.g. dashboard must call the API over HTTP, never import `src.models` directly).
- All config via `get_settings()` (`src/config/settings.py`), never `os.getenv()` or hardcoded values.
- All logging via `src/utils/logger.py` (`get_logger(__name__)`), never `print()`.
- All exceptions derive from `MarketPulseException` (`src/utils/exceptions.py`).
- Google-style docstrings + full type hints on every public function.
- Tests live in `tests/test_<module>.py`, follow Arrange-Act-Assert, mock external APIs.

## ⚠️ Status reality check (read this before trusting docs/SPRINT_STATUS.md)

`docs/SPRINT_STATUS.md` and `docs/ROADMAP.md` claim "33/33 issues done, v1.0.0 release
ready, 87% coverage, 30/30 tests passing" with an 8-week sprint calendar (2026-08-01 →
2026-09-28). Git history shows the entire repo was actually generated in a single day
(2026-08-17) — the "8-week roadmap" is a planning artifact, not a real timeline.

**Verified independently on 2026-09-16** on this machine (fresh clone, fresh Python
3.11 venv, `pip install -r requirements.txt`, no prior state carried over):
- `pytest tests/ --cov=src` → **30 passed, 86.64% coverage** — matches the documented
  claim almost exactly, despite `requirements.txt` resolving to much newer major
  versions than originally pinned (numpy 2.x, pandas 3.x, torch 2.14 vs. the
  numpy==1.24.3/torch==2.0.1 the docs assume — see "Dependency version drift" below).
- `uvicorn src.api.main:app` boots cleanly and `/health` returns `200` with real
  structured JSON logs.

So the completion claims hold up under independent verification — this is a solid,
working scaffold, not just aspirational docs. Still, "verified once on 2026-09-16"
isn't a standing guarantee: re-run the tests after any dependency bump or before
trusting a doc's checkmark on something you're about to build on top of.

## Environment (this machine, Windows)

- Python 3.11.9 installed via winget at `C:\Users\Pc\AppData\Local\Programs\Python\Python311\python.exe`
- Git installed via winget (`C:\Program Files\Git\cmd\git.exe`)
- Virtualenv at `.venv\` (not `venv\` — the Makefile assumes `venv/bin/activate`, which
  is Unix-only and won't work here. The Makefile targets are not directly usable on
  this Windows/PowerShell setup unless you have `make` installed separately; run the
  underlying commands directly instead — see below)
- No `make` installed — don't suggest `make test` etc. as-is; use the PowerShell
  equivalents.

### Common commands (PowerShell, from repo root)

```powershell
# Run tests with coverage
.venv\Scripts\python.exe -m pytest tests/ --cov=src --cov-report=term-missing -v

# Lint
.venv\Scripts\python.exe -m flake8 src/ tests/ --max-line-length=100
.venv\Scripts\python.exe -m mypy src/ --ignore-missing-imports
.venv\Scripts\python.exe -m bandit -r src/ -ll

# Format
.venv\Scripts\python.exe -m black src/ tests/ --line-length=100
.venv\Scripts\python.exe -m isort src/ tests/

# Run API
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload

# Run dashboard
.venv\Scripts\python.exe -m streamlit run src/dashboard/app.py
```

## API keys

No `.env` file exists yet — API keys (Alpaca, Reddit, NewsAPI) are **not** currently
available. Per `docs/DEV_SETUP.md`, all data connectors have synthetic/mock fallbacks
and tests mock external APIs, so this doesn't block development or testing. Don't
assume real API keys are configured; don't write code that hard-fails without them.

## Dependency version drift

`requirements.txt` pins minimum versions only (`>=`), and `docs/DEV_SETUP.md` warns the
project was tested against `numpy==1.24.3` / `torch==2.0.1`. A fresh install on this
machine pulled much newer majors (numpy 2.x, torch 2.14, pandas 3.x, etc.). If tests
fail in ways that look like a numpy 2.x / pandas 3.x breaking change rather than a
logic bug, suspect version drift first before assuming the original code was wrong.
