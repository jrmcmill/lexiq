import os
import logging

# Suppress repeated Transformers advisory messages during lazy module imports.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from src.config import get_device, Config
from src.observability.logger import get_logger

logger = get_logger(__name__)

class Embedder:
    def __init__(self):
        self.model_name = Config.EMBED_MODEL
        self.device = get_device()
        try:
            self.model = SentenceTransformer(self.model_name, device=str(self.device))
        except Exception:
            self.model = SentenceTransformer(self.model_name)
        logger.info(f"Embedder loaded on {self.device}")

    def _encode_batch(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        embeddings = []
        batch = []
        for t in tqdm(prefixed, desc="Embedding", total=len(prefixed), unit="text"):
            batch.append(t)
            if len(batch) >= 32:
                emb = self._encode_batch(batch)
                embeddings.extend(emb)
                batch = []
        if batch:
            embeddings.extend(self._encode_batch(batch))
        # normalize
        embs = [list(e / (np.linalg.norm(e) + 1e-12)) for e in embeddings]
        return embs

    def embed_query(self, query: str) -> list[float]:
        q = "Represent this question for searching relevant passages: " + query
        emb = self.model.encode([q])[0]
        import numpy as np
        return list((emb / (np.linalg.norm(emb) + 1e-12)).tolist())
