"""
Text preprocessing and cleaning pipeline for financial headlines & social media posts.
"""

import re
from typing import List


class TextPreprocessor:
    """
    Cleans financial text, extracts cashtags, normalizes whitespace and punctuation.
    """

    def __init__(self, min_chars: int = 10, max_chars: int = 1000):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.html_pattern = re.compile(r"<.*?>")
        self.multiple_spaces = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        """
        Clean single text string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Remove URLs & HTML tags
        text = self.url_pattern.sub("", text)
        text = self.html_pattern.sub("", text)

        # Normalize ticker symbols ($AAPL -> AAPL)
        text = re.sub(r"\$([A-Za-z]+)", r"\1", text)

        # Collapse whitespace
        text = self.multiple_spaces.sub(" ", text).strip()

        # Enforce length constraints
        if len(text) < self.min_chars:
            return ""
        return text[: self.max_chars]

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of text strings.
        """
        return [self.clean(t) for t in texts]
