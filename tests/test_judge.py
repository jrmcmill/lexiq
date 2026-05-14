from src.evaluation.judge import LLMJudge
import pytest

class Dummy:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        return

def test_judge_parsing(monkeypatch):
    j = LLMJudge()
    # clean JSON
    monkeypatch.setattr(j, '_call_ollama', lambda p: '{"faithfulness": 5, "relevance": 4, "completeness": 3, "reasoning": "ok"}')
    out = j.score('q','ctx','ans')
    assert out['faithfulness'] == 5
    # fenced JSON
    monkeypatch.setattr(j, '_call_ollama', lambda p: '```json\n{"faithfulness": 4, "relevance": 4, "completeness": 4, "reasoning": "ok"}\n```')
    out = j.score('q','ctx','ans')
    assert out['faithfulness'] == 4
    # python dict literal
    monkeypatch.setattr(j, '_call_ollama', lambda p: "{'faithfulness': 3, 'relevance': 3, 'completeness': 3, 'reasoning': 'ok'}")
    out = j.score('q','ctx','ans')
    assert out['faithfulness'] == 3
    # plain text fallback
    monkeypatch.setattr(j, '_call_ollama', lambda p: 'I cannot respond in JSON')
    out = j.score('q','ctx','ans')
    assert out['faithfulness'] == 0
