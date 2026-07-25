.PHONY: help setup install test lint format type-check train serve clean docker-build docker-up docker-down

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help:
	@echo "$(GREEN)MarketPulse AI - Development Tasks$(NC)"
	@echo ""
	@echo "$(YELLOW)Environment Setup:$(NC)"
	@echo "  make setup              Create venv and install dependencies"
	@echo "  make install            Install dependencies only"
	@echo ""
	@echo "$(YELLOW)Quality Assurance:$(NC)"
	@echo "  make test               Run pytest with coverage"
	@echo "  make lint               Run flake8, mypy, bandit"
	@echo "  make format             Auto-format code with black, isort"
	@echo "  make type-check         Type checking with mypy"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make train              Train model"
	@echo "  make serve              Start API + Dashboard locally"
	@echo "  make api                Start FastAPI server only"
	@echo "  make dashboard          Start Streamlit dashboard only"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make docker-build       Build Docker images"
	@echo "  make docker-up          Start services with docker-compose"
	@echo "  make docker-down        Stop docker-compose services"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make clean              Remove artifacts and caches"
	@echo "  make help               Show this message"

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

setup:
	@echo "$(GREEN)[1/4] Creating virtual environment...$(NC)"
	python3.10 -m venv venv
	@echo "$(GREEN)[2/4] Activating venv and upgrading pip...$(NC)"
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	@echo "$(GREEN)[3/4] Installing dependencies...$(NC)"
	. venv/bin/activate && pip install -r requirements.txt
	@echo "$(GREEN)[4/4] Installing development dependencies...$(NC)"
	. venv/bin/activate && pip install -e ".[dev]"
	@echo "$(GREEN)✓ Setup complete! Activate venv: source venv/bin/activate$(NC)"

install:
	@echo "$(GREEN)Installing dependencies...$(NC)"
	. venv/bin/activate && pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

# ============================================================================
# QUALITY ASSURANCE
# ============================================================================

test:
	@echo "$(GREEN)Running pytest with coverage...$(NC)"
	. venv/bin/activate && pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)✓ Tests complete. Coverage report: htmlcov/index.html$(NC)"

lint:
	@echo "$(GREEN)Running flake8...$(NC)"
	. venv/bin/activate && flake8 src/ tests/ --max-line-length=100 --count
	@echo "$(GREEN)Running mypy (type checking)...$(NC)"
	. venv/bin/activate && mypy src/ --ignore-missing-imports
	@echo "$(GREEN)Running bandit (security)...$(NC)"
	. venv/bin/activate && bandit -r src/ -ll
	@echo "$(GREEN)✓ Linting complete$(NC)"

format:
	@echo "$(GREEN)Running black (formatter)...$(NC)"
	. venv/bin/activate && black src/ tests/ --line-length=100
	@echo "$(GREEN)Running isort (import sorter)...$(NC)"
	. venv/bin/activate && isort src/ tests/
	@echo "$(GREEN)✓ Formatting complete$(NC)"

type-check:
	@echo "$(GREEN)Running mypy (strict mode)...$(NC)"
	. venv/bin/activate && mypy src/ --strict --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# ============================================================================
# DEVELOPMENT
# ============================================================================

train:
	@echo "$(GREEN)Starting model training...$(NC)"
	. venv/bin/activate && python scripts/train_model.py

serve:
	@echo "$(GREEN)Starting API and Dashboard...$(NC)"
	. venv/bin/activate && echo "API and Dashboard starting on http://localhost:8000 and http://localhost:8501"
	@echo "$(YELLOW)Note: Run in separate terminals:$(NC)"
	@echo "  Terminal 1: make api"
	@echo "  Terminal 2: make dashboard"

api:
	@echo "$(GREEN)Starting FastAPI server (http://localhost:8000)...$(NC)"
	. venv/bin/activate && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	@echo "$(GREEN)Starting Streamlit dashboard (http://localhost:8501)...$(NC)"
	. venv/bin/activate && streamlit run src/dashboard/app.py

# ============================================================================
# DOCKER
# ============================================================================

docker-build:
	@echo "$(GREEN)Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Build complete$(NC)"

docker-up:
	@echo "$(GREEN)Starting docker-compose services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started:$(NC)"
	@echo "  API: http://localhost:8000"
	@echo "  Dashboard: http://localhost:8501"
	@echo "  Docs: http://localhost:8000/docs"

docker-down:
	@echo "$(RED)Stopping docker-compose services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

# ============================================================================
# MAINTENANCE
# ============================================================================

clean:
	@echo "$(RED)Cleaning up artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info .eggs/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .dmypy.json dmypy.json
	rm -rf __pycache__ **/__pycache__
	rm -rf .tox/
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"
