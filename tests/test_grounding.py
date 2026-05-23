import json
from unittest.mock import patch
from src.agent import nodes as nd


def test_strict_grounding_repair_triggers_and_replaces(monkeypatch):
    calls = []

    # First call: model produces an unsupported attributed sentence
    def fake_call(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "Mirabelli v. Bonta held that the sale of fire extinguishers was unlawful. [Source: Mirabelli v. Bonta]"
        # Second call: strict grounding repair returns JSON mapping
        return json.dumps({
            "Mirabelli v. Bonta": {
                "summary": "Mirabelli v. Bonta discusses notice-and-comment requirements for administrative rulemaking.",
                "quote": "The agency failed to provide adequate notice and opportunity for comment."
            }
        })

    monkeypatch.setattr(nd, '_call_ollama', fake_call)

    st = {
        'messages': [],
        'query': 'give examples of due process cases',
        'retrieved_cases': [
            {
                'text': 'Mirabelli v. Bonta discusses notice-and-comment procedures in administrative rulemaking.',
                'metadata': {'bluebook_cite': 'Mirabelli v. Bonta', 'parent_opinion_id': 1, 'case_name': 'Mirabelli v. Bonta'},
                'distance': 0.2,
                'score': 0.9,
            }
        ],
        'retrieved_statutes': [],
        'retrieved_regs': [],
        'retrieved_session': [],
        'final_answer': '',
        'citations': [],
        'tool_calls': ['case_law'],
        'session_id': 's',
        'court_filter': None,
        'date_after': None,
        'date_before': None,
        'error': None,
    }

    out = nd.generate_answer(st)
    # Ensure strict grounding repair attempted and recorded
    assert out.get('grounding_repair') and out['grounding_repair'].get('attempted') is True
    # The final answer should include the grounded summary and the quote provided in the repair JSON
    assert 'notice-and-comment' in out['final_answer'] or 'SOURCE_QUOTE' in out['final_answer']
    # Ensure two model calls were made (initial draft + strict grounding repair)
    assert len(calls) >= 2
