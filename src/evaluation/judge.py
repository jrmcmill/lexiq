import requests
import json
from datetime import datetime
from src.config import Config
from src.observability.logger import get_logger

logger = get_logger(__name__)

class LLMJudge:
    def __init__(self):
        self.base = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL

    def _call_ollama(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            resp = requests.post(f"{self.base}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            logger.error(str(e))
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")

    def score(self, question: str, context: str, answer: str) -> dict:
        prompt = f"You are an impartial evaluator for a legal research AI assistant.\nScore the answer below on three dimensions (each 1-5):\n\nFAITHFULNESS: Does the answer only make claims supported by the context?\nRELEVANCE: Does the answer directly address the question?\nCOMPLETENESS: Does the answer cover all key aspects of the question?\n\nQuestion: {question}\nContext (retrieved sources): {context[:2000]}\nAnswer: {answer}\n\nReturn ONLY valid JSON in this format, with no preamble:\n{{\"faithfulness\": <1-5>, \"relevance\": <1-5>,\n \"completeness\": <1-5>, \"reasoning\": \"<one sentence>\"}}"
        try:
            resp = self._call_ollama(prompt)
        except Exception:
            return {"faithfulness": 0, "relevance": 0, "completeness": 0, "reasoning": "call_failed", "question_preview": question[:80], "answer_preview": answer[:80], "timestamp": datetime.utcnow().isoformat() + 'Z'}
        text = resp.strip()
        # strip fenced code
        if text.startswith('```'):
            # remove fence
            parts = text.split('\n')
            if parts[0].startswith('```'):
                text = '\n'.join(parts[1:])
                if text.endswith('```'):
                    text = '\n'.join(parts[1:-1])
        try:
            parsed = json.loads(text)
        except Exception:
            # try eval-like
            try:
                parsed = eval(text, {})
            except Exception:
                return {"faithfulness": 0, "relevance": 0, "completeness": 0, "reasoning": "parse_failed", "question_preview": question[:80], "answer_preview": answer[:80], "timestamp": datetime.utcnow().isoformat() + 'Z'}
        parsed.update({"question_preview": question[:80], "answer_preview": answer[:80], "timestamp": datetime.utcnow().isoformat() + 'Z'})
        return parsed

if __name__ == '__main__':
    j = LLMJudge()
    print(j.score('Q','ctx','A'))
