import requests
from datetime import datetime
from src.config import Config
from src.observability.logger import get_logger

logger = get_logger(__name__)

class DocumentComparator:
    def __init__(self):
        self.base = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL

    def extract_full_text(self, chunks: list[dict]) -> str:
        sorted_chunks = sorted(chunks, key=lambda c: c.get('chunk_index', 0))
        return "\n\n".join([c['text'] for c in sorted_chunks])

    def _truncate_words(self, text, max_words=3000):
        words = text.split()
        return ' '.join(words[:max_words])

    def compare(self, text_a: str, text_b: str, label_a: str = "Document A", label_b: str = "Document B") -> dict:
        ta = self._truncate_words(text_a)
        tb = self._truncate_words(text_b)
        prompt = f"You are a legal document analyst. Compare the following two documents and produce a structured analysis with these sections:\n\n1. KEY SIMILARITIES: bullet points of substantive similarities\n2. KEY DIFFERENCES: bullet points of substantive differences\n3. MISSING FROM {label_a}: provisions/clauses in B but not A\4. MISSING FROM {label_b}: provisions/clauses in A but not B\n5. RISK FLAGS: any legal risks, inconsistencies, or concerning language\n6. OVERALL ASSESSMENT: 2-3 sentence summary\n\nDocument A ({label_a}):\n{ta}\n\nDocument B ({label_b}):\n{tb}\n\nRespond with the six numbered sections only."
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(f"{self.base}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            text = resp.text
        except requests.exceptions.RequestException as e:
            logger.error(str(e))
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")
        # naive parse: split sections by leading numbers
        parts = {}
        sections = ['similarities','differences','missing_from_a','missing_from_b','risk_flags','overall_assessment']
        # split by '\n1.' etc
        content = text
        for i, key in enumerate(sections, start=1):
            marker = f"{i}."
            next_marker = f"{i+1}." if i < len(sections) else None
            start = content.find(marker)
            if start == -1:
                parts[key] = ""
                continue
            start += len(marker)
            end = content.find(next_marker) if next_marker else len(content)
            parts[key] = content[start:end].strip()
        return {
            'label_a': label_a,
            'label_b': label_b,
            'similarities': parts.get('similarities',''),
            'differences': parts.get('differences',''),
            'missing_from_a': parts.get('missing_from_a',''),
            'missing_from_b': parts.get('missing_from_b',''),
            'risk_flags': parts.get('risk_flags',''),
            'overall_assessment': parts.get('overall_assessment',''),
            'model_used': self.model,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

if __name__ == '__main__':
    c = DocumentComparator()
    print('Comparator ready')
