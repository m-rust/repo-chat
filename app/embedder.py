"""Local embeddings with sentence-transformers (downloads the model on first run)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config

# BGE models expect this prefix on queries (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL)


def embed_documents(texts: list[str], batch_size: int = 32) -> np.ndarray:
    return _model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=len(texts) > 200)


def embed_query(text: str) -> np.ndarray:
    if "bge" in config.EMBED_MODEL.lower():
        text = BGE_QUERY_PREFIX + text
    return _model().encode([text], normalize_embeddings=True)[0]
