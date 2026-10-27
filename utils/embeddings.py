"""Embedding utilities for QA Agent system.

Uses sentence-transformers when available. If the package or the model
weights cannot be loaded (e.g. offline), falls back to TF-IDF cosine
similarity so the system still runs, with a logged warning.
"""

from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.logger import log
from config import config


class EmbeddingManager:
    """Manages text embeddings for similarity calculations."""

    def __init__(self):
        """Initialize embedding model."""
        from pathlib import Path
        local_fine_tuned = Path(__file__).resolve().parents[1] / "models" / "fine_tuned_sentence_bert"
        if local_fine_tuned.exists() and any(local_fine_tuned.iterdir()):
            model_name = str(local_fine_tuned)
            log.info(f"Using local fine-tuned sentence transformer: {model_name}")
        else:
            model_name = config.get(
                "models.sentence_transformer",
                "all-MiniLM-L6-v2"
            )
            log.info(f"Loading sentence transformer model: {model_name}")
        
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:  # missing package or no network for model download
            log.warning(
                f"Could not load '{model_name}' ({e}). Falling back to "
                f"TF-IDF similarity - results will be less accurate."
            )
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def using_fallback(self) -> bool:
        return self.model is None

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Encode text to an embedding vector (requires the neural model)."""
        if use_cache and text in self._cache:
            return self._cache[text]

        embedding = self.model.encode(text, convert_to_numpy=True)

        if use_cache:
            self._cache[text] = embedding

        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts to embeddings (requires the neural model)."""
        return self.model.encode(texts, convert_to_numpy=True)

    def _similarities(self, query: str, corpus: List[str]) -> np.ndarray:
        """Cosine similarity of one query against every corpus text."""
        if not corpus:
            return np.zeros(0)

        if self.model is not None:
            q = self.encode(query)
            c = self.encode_batch(corpus)
        else:
            try:
                vec = TfidfVectorizer().fit([query] + list(corpus))
                q = vec.transform([query]).toarray()[0]
                c = vec.transform(corpus).toarray()
            except ValueError:  # empty vocabulary
                return np.zeros(len(corpus))

        q_norm = np.linalg.norm(q)
        c_norm = np.linalg.norm(c, axis=1)
        denom = q_norm * c_norm
        denom[denom == 0] = 1.0
        return (c @ q) / denom

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts (0.0 to 1.0 in practice)."""
        return float(self._similarities(text1, [text2])[0])

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """Find the top_k candidates most similar to query."""
        if not candidates:
            return []
        sims = self._similarities(query, candidates)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [
            {"text": candidates[i], "score": float(sims[i]), "index": int(i)}
            for i in top_indices
        ]

    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        threshold: float = 0.5
    ) -> List[Dict[str, any]]:
        """Return corpus texts with similarity >= threshold, best first."""
        sims = self._similarities(query, corpus)
        results = [
            {"text": corpus[i], "score": float(s), "index": i}
            for i, s in enumerate(sims) if s >= threshold
        ]
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        log.info("Embedding cache cleared")
