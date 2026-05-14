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
    monkeypatch.setattr(nd, '_call_ollama', lambda p: 'Answer. 347 U.S. 483')
    st = {'messages': [], 'query': 'q', 'retrieved_cases': [], 'retrieved_statutes': [], 'retrieved_regs': [], 'retrieved_session': [], 'final_answer':'', 'citations':[], 'tool_calls':['case_law'], 'session_id':'s', 'court_filter':None, 'date_after':None, 'date_before':None, 'error':None}
    st2 = generate_answer(st)
    assert 'final_answer' in st2
    assert isinstance(st2['citations'], list)
