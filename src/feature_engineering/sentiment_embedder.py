"""
FinBERT sentiment and contextual embedding generator.
Extracts 768-dimensional CLS token embeddings from financial texts.
"""

from typing import List, Optional
import numpy as np
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("FinBERTEmbedder")


class FinBERTEmbedder:
    """
    Extracts 768-D dense embeddings using Hugging Face FinBERT model.
    Falls back to deterministic hash projection if transformers/torch is unavailable.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.nlp.finbert_model
        self.device = device or settings.nlp.device
        self.max_length = max_length
        self.embedding_dim = 768
        self._tokenizer = None
        self._model = None

    def _lazy_load_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            return True
        except Exception as e:
            logger.warning(f"Could not load Hugging Face FinBERT ({str(e)}); using fallback embedder.")
            return False

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of text strings into an [N, 768] numpy array.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if not self._lazy_load_model():
            return self._fallback_embed(texts)

        import torch

        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**encoded)
                # Extract [CLS] token representation
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(cls_embeddings)

        return np.vstack(embeddings_list)

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Deterministic pseudo-embedding for testing or offline environments."""
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = abs(hash(text)) % (2**32)
            rng = np.random.RandomState(seed)
            embeddings[i] = rng.normal(0.0, 1.0, size=self.embedding_dim)
            # Normalize
            norm = np.linalg.norm(embeddings[i]) + 1e-9
            embeddings[i] = embeddings[i] / norm
        return embeddings
