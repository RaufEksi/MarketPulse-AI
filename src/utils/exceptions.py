"""
Custom domain exception hierarchy for MarketPulse AI.
"""


class MarketPulseException(Exception):
    """Base exception for all MarketPulse AI domain errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(MarketPulseException):
    """Raised when configuration parsing or validation fails."""

    pass


class DataIngestionError(MarketPulseException):
    """Raised when external data collection fails."""

    pass


class AlignmentError(MarketPulseException):
    """Raised when temporal alignment between text and time series fails."""

    pass


class ModelInferenceError(MarketPulseException):
    """Raised when model loading, forward pass, or prediction fails."""

    pass


class XAIError(MarketPulseException):
    """Raised when SHAP, Integrated Gradients, or attribution calculation fails."""

    pass
