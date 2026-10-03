"""
Standalone training script for the MarketPulse AI multi-modal hybrid model.

Builds the training set from 5-minute OHLCV bars and real financial text
(Reddit / news collectors and/or a text file), embeds the text with FinBERT,
decay-aligns it to the bars, trains MarketPulseNet and writes a self-describing
checkpoint that the API's /predict and /explain endpoints load.

Usage:
    python scripts/train_model.py --symbol SPY --epochs 20 --text-file data/raw/texts.parquet
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

# Allow `python scripts/train_model.py` from the repo root without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.data_alignment.dataset_builder import (  # noqa: E402
    build_sliding_windows,
    create_walk_forward_dataloaders,
)
from src.data_alignment.exponential_decay import TemporalAligner  # noqa: E402
from src.data_engine.alpaca_connector import AlpacaDataCollector  # noqa: E402
from src.data_engine.news_collector import NewsCollector  # noqa: E402
from src.data_engine.reddit_collector import RedditCollector  # noqa: E402
from src.data_engine.yfinance_connector import YFinanceDataCollector  # noqa: E402
from src.feature_engineering.labeler import VolatilityLabeler  # noqa: E402
from src.feature_engineering.sentiment_embedder import (  # noqa: E402
    FALLBACK_BACKEND,
    FinBERTEmbedder,
)
from src.feature_engineering.technical_indicators import (  # noqa: E402
    MODEL_FEATURE_COLUMNS,
    TechnicalFeatureEngine,
)
from src.models.checkpoint import fit_feature_normalizer, save_checkpoint  # noqa: E402
from src.models.hybrid_network import MarketPulseNet  # noqa: E402
from src.models.trainer import ModelTrainer  # noqa: E402
from src.utils.exceptions import (  # noqa: E402
    ConfigurationError,
    DataIngestionError,
    MarketPulseException,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("TrainScript")

TEXT_COLUMNS = ["id", "timestamp", "symbol", "source", "text", "score", "num_comments"]
BAR_SOURCES = ["auto", "alpaca", "yfinance", "synthetic"]
YFINANCE_MAX_5MIN_DAYS = 59  # Yahoo only serves ~60 days of 5-minute history
TRAINER_CHECKPOINT_NAME = "best_marketpulse_net.pt"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train MarketPulseNet multi-modal model.")
    parser.add_argument("--symbol", type=str, default="SPY", help="Asset ticker")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=settings.training.batch_size, help="Mini-batch size"
    )
    parser.add_argument("--days", type=int, default=30, help="Days of 5-minute bars to train on")
    parser.add_argument(
        "--bars-source",
        choices=BAR_SOURCES,
        default="auto",
        help="auto = Alpaca if keys are configured, else Yahoo Finance",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help=f"Parquet/CSV of texts with columns {', '.join(TEXT_COLUMNS)}",
    )
    parser.add_argument(
        "--no-live-text",
        action="store_true",
        help="Do not call the Reddit/News collectors; use only --text-file",
    )
    parser.add_argument(
        "--allow-fallback-embeddings",
        action="store_true",
        help="Train even if FinBERT cannot be loaded (hash embeddings; for offline smoke tests)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.model.checkpoint_path),
        help="Where to write the final checkpoint loaded by the API",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Directory for per-epoch best weights",
    )
    return parser.parse_args(argv)


def load_bars(source: str, symbol: str, days: int) -> pd.DataFrame:
    """
    Load 5-minute OHLCV bars from the requested source.

    Args:
        source: One of ``BAR_SOURCES``.
        symbol: Asset ticker.
        days: Number of calendar days of history.

    Returns:
        Bars DataFrame sorted by timestamp with a UTC 'timestamp' column.

    Raises:
        DataIngestionError: If the source returns no bars.
    """
    settings = get_settings()
    if source == "auto":
        has_alpaca = bool(settings.alpaca_api_key and settings.alpaca_secret_key)
        source = "alpaca" if has_alpaca else "yfinance"

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    if source == "yfinance":
        period_days = min(days, YFINANCE_MAX_5MIN_DAYS)
        bars = YFinanceDataCollector().fetch_5min_bars(symbol, period=f"{period_days}d")
    elif source == "alpaca":
        bars = AlpacaDataCollector().fetch_5min_bars(symbol, start_time, end_time, limit=10000)
    else:
        bars = AlpacaDataCollector()._generate_synthetic_bars(symbol, start_time, end_time)

    if bars is None or bars.empty:
        raise DataIngestionError(f"No bars returned from {source} for {symbol}")

    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"Loaded {len(bars)} bars for {symbol} from {source}",
        extra={"start": str(bars["timestamp"].iloc[0]), "end": str(bars["timestamp"].iloc[-1])},
    )
    return bars


def load_texts(symbol: str, text_file: Optional[Path], use_collectors: bool) -> pd.DataFrame:
    """
    Gather financial text events for ``symbol`` from a file and/or the collectors.

    Args:
        symbol: Asset ticker; rows for other symbols are dropped.
        text_file: Optional Parquet/CSV file following the collector column contract.
        use_collectors: Whether to call the Reddit and News collectors.

    Returns:
        De-duplicated DataFrame with ``TEXT_COLUMNS`` and a UTC 'timestamp' column.

    Raises:
        DataIngestionError: If the text file is missing required columns.
    """
    frames = []
    if text_file is not None:
        if text_file.suffix == ".parquet":
            file_df = pd.read_parquet(text_file)
        else:
            file_df = pd.read_csv(text_file)
        missing = {"timestamp", "symbol", "text"} - set(file_df.columns)
        if missing:
            raise DataIngestionError(f"Text file {text_file} missing columns: {missing}")
        frames.append(file_df)

    if use_collectors:
        for name, fetch in (
            ("reddit", lambda: RedditCollector().fetch_posts(symbols=[symbol])),
            ("news", lambda: NewsCollector().fetch_headlines(symbols=[symbol])),
        ):
            try:
                frames.append(fetch())
            except MarketPulseException as e:
                logger.warning(f"Skipping {name} texts: {e.message}")

    if not frames:
        return pd.DataFrame(columns=TEXT_COLUMNS)

    texts = pd.concat(frames, ignore_index=True)
    for col in TEXT_COLUMNS:
        if col not in texts.columns:
            texts[col] = None
    texts = texts[TEXT_COLUMNS]
    texts = texts[texts["symbol"].astype(str).str.upper() == symbol.upper()]
    texts = texts.dropna(subset=["timestamp", "text"])
    texts = texts[texts["text"].astype(str).str.strip() != ""]
    texts["timestamp"] = pd.to_datetime(texts["timestamp"], utc=True)
    texts = texts.drop_duplicates(subset=["timestamp", "text"]).sort_values("timestamp")
    return texts.reset_index(drop=True)


def build_text_matrix(
    bars: pd.DataFrame,
    texts: pd.DataFrame,
    embedder: FinBERTEmbedder,
) -> np.ndarray:
    """
    Embed texts and decay-align them onto every bar.

    Args:
        bars: Bars DataFrame with a 'timestamp' column.
        texts: Text DataFrame with 'timestamp' and 'text' columns.
        embedder: FinBERT embedder.

    Returns:
        Float32 array [num_bars, embedding_dim].
    """
    embedding_dim = get_settings().nlp.embedding_dim
    if texts.empty:
        return np.zeros((len(bars), embedding_dim), dtype=np.float32)
    embeddings = embedder.embed_texts(texts["text"].astype(str).tolist())
    return TemporalAligner().align_sentiment_to_bars(
        bars, texts, embeddings, embedding_dim=embedding_dim
    )


def main(argv: Optional[List[str]] = None) -> Path:
    """
    Run the full training pipeline and write the API checkpoint.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Path of the written checkpoint.

    Raises:
        ConfigurationError: If FinBERT is unavailable and fallback embeddings are not allowed.
        DataIngestionError: If no bars could be loaded.
    """
    args = parse_args(argv)
    settings = get_settings()
    torch.manual_seed(settings.app.seed)
    np.random.seed(settings.app.seed)

    # 1. Text embedder: insist on real FinBERT unless explicitly told otherwise
    embedder = FinBERTEmbedder()
    backend = embedder.backend
    if backend == FALLBACK_BACKEND and not args.allow_fallback_embeddings:
        raise ConfigurationError(
            f"FinBERT '{embedder.model_name}' could not be loaded, so training would use "
            "hash embeddings. Check network access to huggingface.co or pass "
            "--allow-fallback-embeddings for an offline smoke test."
        )

    # 2. Bars -> technical features -> labels
    bars = load_bars(args.bars_source, args.symbol, args.days)
    features_df = TechnicalFeatureEngine().transform(bars)
    labeled_df = VolatilityLabeler(
        horizon_bars=settings.data.prediction_horizon_bars,
        threshold_pct=settings.data.volatility_threshold_pct,
    ).create_labels(features_df)

    # 3. Texts -> FinBERT embeddings -> decay-aligned per-bar sentiment
    texts = load_texts(args.symbol, args.text_file, use_collectors=not args.no_live_text)
    text_mat = build_text_matrix(labeled_df, texts, embedder)
    bars_with_text = int(np.any(text_mat != 0.0, axis=1).sum())
    logger.info(
        f"Embedded {len(texts)} texts with {backend}; "
        f"{bars_with_text}/{len(text_mat)} bars carry sentiment"
    )
    if bars_with_text == 0:
        logger.warning("No bar has any text in its lookback; the text branch will learn nothing.")

    # 4. Normalize price features with statistics from the training split only
    feat_mat = labeled_df[MODEL_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    targets = labeled_df["atr_spike_target"].to_numpy(dtype=np.float64)
    seq_len = settings.data.lookback_bars
    num_windows = len(feat_mat) - seq_len
    train_rows = int(
        num_windows * (1.0 - settings.training.val_split - settings.training.test_split)
    )
    feature_mean, feature_std = fit_feature_normalizer(feat_mat[: train_rows + seq_len])
    feat_norm = ((feat_mat - feature_mean) / feature_std).astype(np.float32)

    ts_win, text_win, y_win = build_sliding_windows(
        feat_norm, text_mat, targets, sequence_length=seq_len
    )
    train_loader, val_loader, test_loader = create_walk_forward_dataloaders(
        ts_win,
        text_win,
        y_win,
        val_split=settings.training.val_split,
        test_split=settings.training.test_split,
        batch_size=args.batch_size,
    )

    # 5. Train, then restore the best-validation weights
    model_config = {
        "ts_input_dim": len(MODEL_FEATURE_COLUMNS),
        "text_input_dim": settings.nlp.embedding_dim,
        "hidden_dim": settings.model.time_series.hidden_dim,
        "num_heads": settings.model.fusion.num_heads,
        "dropout": settings.model.fusion.dropout,
        "ts_model_type": settings.model.time_series.model_type,
    }
    model = MarketPulseNet(**model_config)
    trainer = ModelTrainer(
        model=model,
        learning_rate=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
        device="cpu",
    )
    results = trainer.fit(
        train_loader, val_loader, epochs=args.epochs, checkpoint_dir=str(args.checkpoint_dir)
    )
    best_weights = args.checkpoint_dir / TRAINER_CHECKPOINT_NAME
    model.load_state_dict(torch.load(best_weights, map_location="cpu", weights_only=True))
    test_metrics = trainer.evaluate(test_loader)
    logger.info(
        f"Best val PR-AUC {results['best_pr_auc']:.4f} (epoch {results['best_epoch']}); "
        f"test PR-AUC {test_metrics['pr_auc']:.4f}, ROC-AUC {test_metrics['roc_auc']:.4f}"
    )

    # 6. Persist the checkpoint the API serves
    out_path = save_checkpoint(
        model,
        args.output,
        model_config=model_config,
        feature_columns=MODEL_FEATURE_COLUMNS,
        feature_mean=feature_mean,
        feature_std=feature_std,
        metadata={
            "symbol": args.symbol,
            "text_embedder": backend,
            "num_texts": int(len(texts)),
            "bars_with_text": bars_with_text,
            "num_bars": int(len(bars)),
            "data_start": str(bars["timestamp"].iloc[0]),
            "data_end": str(bars["timestamp"].iloc[-1]),
            "epochs_run": int(args.epochs),
            "best_epoch": int(results["best_epoch"]),
            "val_pr_auc": float(results["best_pr_auc"]),
            "test_pr_auc": float(test_metrics["pr_auc"]),
            "test_roc_auc": float(test_metrics["roc_auc"]),
        },
    )
    return out_path


if __name__ == "__main__":
    main()
