"""
FastAPI application instance and middleware configuration for MarketPulse AI.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import backtest, explain, health, predict
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("MarketPulseAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown procedures."""
    logger.info("Initializing MarketPulse AI REST API service...")
    yield
    logger.info("Shutting down MarketPulse AI REST API service...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="MarketPulse AI API",
        description="Multi-Modal Financial Volatility & Market Regime Shift Prediction Service",
        version=settings.app.version,
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root welcome & redirection endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Welcome to MarketPulse AI REST API",
            "version": settings.app.version,
            "documentation": "/docs",
            "status": "healthy",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "explain": "/explain",
                "backtest": "/backtest",
                "metrics": "/metrics",
            },
        }

    # Include route modules
    app.include_router(predict.router, tags=["Prediction"])
    app.include_router(explain.router, tags=["Explainability"])
    app.include_router(backtest.router, tags=["Backtesting"])
    app.include_router(health.router, tags=["System & Monitoring"])

    return app


app = create_app()
