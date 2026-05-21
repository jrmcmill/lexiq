import os
import pickle
import math
from rank_bm25 import BM25Okapi
from typing import List, Tuple

def _simple_tokenize(text: str) -> List[str]:
    # lightweight tokenizer: lowercase, split on whitespace, remove punctuation
    import re
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens

class BM25Index:
    def __init__(self, persist_dir: str, name: str):
        self.name = name
        self.persist_dir = persist_dir
        self.path = os.path.join(persist_dir, f"bm25_{name}.pkl")
        self.ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def build(self, ids: List[str], texts: List[str]):
        self.ids = list(ids)
        self.corpus_tokens = [_simple_tokenize(t) for t in texts]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.save()

    def save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"ids": self.ids, "corpus_tokens": self.corpus_tokens}, f)

    def load(self):
        if not os.path.exists(self.path):
            return False
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.ids = data.get("ids", [])
        self.corpus_tokens = data.get("corpus_tokens", [])
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            return True
        return False

    def query(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.bm25:
            return []
        q_tok = _simple_tokenize(query)
        scores = self.bm25.get_scores(q_tok)
        # pair ids with scores, sort descending
        pairs = [(i, float(s)) for i, s in zip(self.ids, scores)]
        pairs_sorted = sorted(pairs, key=lambda p: p[1], reverse=True)
        return pairs_sorted[:top_k]

    def top_ids(self, query: str, top_k: int = 10) -> List[str]:
        return [pid for pid, _ in self.query(query, top_k=top_k)]
