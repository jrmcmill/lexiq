import pytest
from unittest.mock import patch
from src.agent.graph import run_query
from src.agent.nodes import route_query, generate_answer

def test_run_query_returns_final_answer(monkeypatch):
    state = run_query('What is X?','sess', [])
    assert isinstance(state, dict)
    assert 'final_answer' in state

def test_route_query_fallback_on_parse_failure(monkeypatch):
    # force route_query to parse invalid JSON by patching _call_ollama
    from src.agent import nodes as nd
    monkeypatch.setattr(nd, '_call_ollama', lambda p: 'not a json')
    st = {'messages': [], 'query': 'q', 'retrieved_cases': [], 'retrieved_statutes': [], 'retrieved_regs': [], 'retrieved_session': [], 'final_answer':'', 'citations':[], 'tool_calls':[], 'session_id':'s', 'court_filter':None, 'date_after':None, 'date_before':None, 'error':None}
    st2 = route_query(st)
    assert set(st2['tool_calls']) == set(['case_law','statute','regulation','session'])

def test_generate_answer_builds_prompt(monkeypatch):
    # patch _call_ollama to return a canned response
    from src.agent import nodes as nd
    captured = {}
    def fake_call(prompt):
        captured['prompt'] = prompt
        return 'Answer. 347 U.S. 483'
    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'q',
        'retrieved_cases': [{'text': 'Brown v. Board of Education', 'metadata': {'bluebook_cite': '347 U.S. 483', 'parent_opinion_id': 1}, 'distance': 0.2, 'score': 0.9}],
        'retrieved_statutes': [],
        'retrieved_regs': [],
        'retrieved_session': [],
        'final_answer':'',
        'citations':[],
        'tool_calls':['case_law'],
        'session_id':'s',
        'court_filter':None,
        'date_after':None,
        'date_before':None,
        'error':None,
    }
    st2 = generate_answer(st)
    assert 'final_answer' in st2
    assert isinstance(st2['citations'], list)
    assert st2.get('used_sources')
    assert st2['used_sources'][0]['type'] == 'Case Law'
    assert 'CASE SOURCES' in captured['prompt']
    # When no other sources exist, OTHER SOURCES may be omitted; ensure CASE SOURCES appears before the question block
    assert captured['prompt'].index('CASE SOURCES') < captured['prompt'].index('QUESTION:')


def test_generate_answer_repairs_when_case_missing_from_first_draft(monkeypatch):
    from src.agent import nodes as nd
    calls = []

    def fake_call(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "I cannot find a recent case in the provided sources."
        return "Louisiana v. Callais explains the 2026 gerrymandering dispute and its interaction with Section 2 of the Voting Rights Act.\n\nCitations:\nLouisiana v. Callais, 600 U.S. 1 (2026)."

    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'tell me about the recent 2026 Supreme Court case involving gerrymandering and the voting rights act',
        'retrieved_cases': [
            {
                'text': 'Louisiana v. Callais discusses map challenges under equal protection and Section 2.',
                'metadata': {'bluebook_cite': 'Louisiana v. Callais, 600 U.S. 1 (2026)', 'parent_opinion_id': 1, 'case_name': 'Louisiana v. Callais'},
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

    st2 = generate_answer(st)
    assert len(calls) == 2
    assert 'Louisiana v. Callais' in st2['final_answer']
    assert "cannot find a recent case" not in st2['final_answer'].lower()


def test_generate_answer_repairs_disallowed_citation(monkeypatch):
    from src.agent import nodes as nd
    calls = []

    def fake_call(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "Shelby County v. Holder controls this issue. Citations: 42 U.S.C. § 1973"
        return "Louisiana v. Callais, 600 U.S. 1 (2026), addresses the map dispute and explains how Section 2 analysis interacts with race-conscious districting concerns.\n\nCitations:\nLouisiana v. Callais, 600 U.S. 1 (2026)."

    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'tell me about the recent 2026 Supreme Court case involving gerrymandering and the voting rights act',
        'retrieved_cases': [
            {
                'text': 'Louisiana v. Callais discusses map challenges under equal protection and Section 2.',
                'metadata': {'bluebook_cite': 'Louisiana v. Callais, 600 U.S. 1 (2026)', 'parent_opinion_id': 1, 'case_name': 'Louisiana v. Callais'},
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

    st2 = generate_answer(st)
    assert len(calls) == 2
    assert 'Louisiana v. Callais' in st2['final_answer']
    assert 'Shelby County v. Holder' not in st2['final_answer']
