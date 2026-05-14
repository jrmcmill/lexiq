import os
import logging

# Suppress repeated Transformers advisory messages during lazy module imports.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)

from sentence_transformers import CrossEncoder
from src.config import get_device, Config
from src.observability.logger import get_logger

logger = get_logger(__name__)

class Reranker:
    def __init__(self):
        self.model_name = Config.RERANK_MODEL
        self.device = get_device()
        try:
            self.model = CrossEncoder(self.model_name, device=str(self.device))
        except Exception:
            self.model = CrossEncoder(self.model_name)
        logger.info(f"Reranker loaded on {self.device}")

    def rerank(self, query: str, results: list[dict], top_k: int = None) -> list[dict]:
        top_k = top_k or Config.RERANK_TOP_K
        pairs = [[query, r.get('text', '')] for r in results]
        scores = self.model.predict(pairs)
        for r, s in zip(results, scores):
            r['rerank_score'] = float(s)
        results_sorted = sorted(results, key=lambda r: r['rerank_score'], reverse=True)
        return results_sorted[:top_k]
