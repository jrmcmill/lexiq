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
    assert set(st2['tool_calls']) == set(['case_law','statute','regulation','textbook','session'])

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
    assert 'SOURCE STRATEGY:' in captured['prompt']
    assert 'Case Law required: yes' in captured['prompt']
    assert 'CASE HIGHLIGHTS (read these first):' in captured['prompt']
    assert 'CASE SOURCES' in captured['prompt']
    # Ensure the case-focused context appears before the question block
    assert captured['prompt'].index('CASE HIGHLIGHTS (read these first):') < captured['prompt'].index('QUESTION:')


def test_generate_answer_includes_authority_labels_in_prompt(monkeypatch):
    from src.agent import nodes as nd
    captured = {}

    def fake_call(prompt):
        captured['prompt'] = prompt
        return 'Answer. 600 U.S. 1'

    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'q',
        'retrieved_cases': [
            {
                'text': 'A Supreme Court case directly controls the issue.',
                'metadata': {'bluebook_cite': '600 U.S. 1', 'parent_opinion_id': 1, 'case_name': 'Supreme Court Case', 'court': 'Supreme Court of the United States'},
                'distance': 0.1,
                'score': 0.95,
                'authority_score': 0.98,
                'authority_tier': 'high',
                'authority_notes': 'supreme court authority; citation metadata present',
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
    assert st2['final_answer']
    assert 'AUTHORITY: HIGH' in captured['prompt']
    assert 'AUTHORITY NOTES:' in captured['prompt']


def test_generate_answer_dedupes_duplicate_chunks_before_prompt(monkeypatch):
    from src.agent import nodes as nd
    captured = {}

    def fake_call(prompt):
        captured['prompt'] = prompt
        return 'Brown v. Board of Education controls this issue. Citations: 347 U.S. 483'

    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'q',
        'retrieved_cases': [
            {
                'text': 'Brown v. Board of Education held segregation unconstitutional.',
                'metadata': {'bluebook_cite': '347 U.S. 483', 'parent_opinion_id': 1, 'case_name': 'Brown v. Board of Education', 'source_id': 'chunk-1'},
                'source_id': 'chunk-1',
                'distance': 0.2,
                'score': 0.9,
            },
            {
                'text': '  Brown v. Board of Education held segregation unconstitutional.  ',
                'metadata': {'bluebook_cite': '347 U.S. 483', 'parent_opinion_id': 1, 'case_name': 'Brown v. Board of Education', 'source_id': 'chunk-1'},
                'source_id': 'chunk-1',
                'distance': 0.21,
                'score': 0.89,
            },
        ],
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
    assert st2['final_answer']
    assert captured['prompt'].count('SOURCE: Case Law') == 1
    assert captured['prompt'].count('CITATION: 347 U.S. 483') == 1


def test_generate_answer_validates_inline_source_references(monkeypatch):
    from src.agent import nodes as nd
    captured = {}

    def fake_call(prompt):
        captured['prompt'] = prompt
        return (
            'The source says this happened [Source: 347 U.S. 483]. '
            'An unsupported reference should be removed [Source: 999 U.S. 1].\n\n'
            'Citations:\n347 U.S. 483, 999 U.S. 1'
        )

    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'q',
        'retrieved_cases': [
            {
                'text': 'Brown v. Board of Education held segregation unconstitutional.',
                'metadata': {'bluebook_cite': '347 U.S. 483', 'parent_opinion_id': 1, 'case_name': 'Brown v. Board of Education', 'source_id': 'chunk-1'},
                'source_id': 'chunk-1',
                'distance': 0.2,
                'score': 0.9,
            },
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
    assert '[Source: 999 U.S. 1]' not in st2['final_answer']
    assert '999 U.S. 1' not in st2.get('citations', [])
    assert st2.get('citation_validation_warning')
    assert 'Add inline source references' in captured['prompt']


def test_generate_answer_allows_non_case_primary_sources(monkeypatch):
    from src.agent import nodes as nd
    captured = {}

    def fake_call(prompt):
        captured['prompt'] = prompt
        return 'The regulations control this issue. Citations: 44 C.F.R. § 6.87'

    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'what do the regulations say about the disclosure rule?',
        'retrieved_cases': [
            {
                'text': 'A marginal case mention that is not central to the question.',
                'metadata': {'bluebook_cite': '123 F.4th 456', 'parent_opinion_id': 99, 'case_name': 'Some Case'},
                'distance': 0.9,
                'score': 0.1,
            }
        ],
        'retrieved_statutes': [],
        'retrieved_regs': [
            {
                'text': '44 C.F.R. § 6.87 explains the disclosure rule directly.',
                'metadata': {'cfr_citation': '44 C.F.R. § 6.87'},
                'distance': 0.1,
                'score': 0.95,
            }
        ],
        'retrieved_session': [],
        'final_answer': '',
        'citations': [],
        'tool_calls': ['case_law', 'regulation'],
        'session_id': 's',
        'court_filter': None,
        'date_after': None,
        'date_before': None,
        'error': None,
    }

    st2 = generate_answer(st)
    assert 'SOURCE STRATEGY:' in captured['prompt']
    assert 'Case Law required: no' in captured['prompt']
    assert 'CASE HIGHLIGHTS (read these first):' not in captured['prompt']
    assert st2['final_answer']


def test_generate_answer_includes_textbook_sources_in_prompt(monkeypatch):
    from src.agent import nodes as nd
    captured = {}

    def fake_call(prompt):
        captured['prompt'] = prompt
        return 'The textbook explains the doctrine. Citations: '

    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'what is negligence in tort law?',
        'retrieved_cases': [],
        'retrieved_statutes': [],
        'retrieved_regs': [],
        'retrieved_textbooks': [
            {
                'text': 'Negligence is the failure to exercise reasonable care under the circumstances.',
                'metadata': {
                    'book_title': 'Everything You Need To Know About American Law',
                    'source_filename': 'law_101.pdf',
                    'section_heading': 'Negligence',
                    'chapter': 'Torts',
                    'page_number': 42,
                },
                'distance': 0.12,
                'score': 0.96,
            }
        ],
        'retrieved_session': [],
        'final_answer': '',
        'citations': [],
        'tool_calls': ['textbook'],
        'session_id': 's',
        'court_filter': None,
        'date_after': None,
        'date_before': None,
        'error': None,
    }

    st2 = generate_answer(st)
    assert st2['final_answer']
    assert 'SOURCE: Textbook' in captured['prompt']
    assert 'Everything You Need To Know About American Law' in captured['prompt']


def test_retrieve_node_retries_low_confidence_sources_and_sets_warning(monkeypatch):
    from src.agent import nodes as nd

    calls = []

    def fake_case_search(query, court_filter=None, date_after=None, date_before=None, debug=False, aggressive=False):
        calls.append({'query': query, 'aggressive': aggressive, 'debug': debug})
        if aggressive:
            return {
                'results': [
                    {'text': 'Better expanded case result', 'metadata': {'bluebook_cite': '111 U.S. 1'}, 'score': 0.72},
                ],
                'trace': {'average_score': 0.72, 'confidence_threshold': 0.40, 'result_count': 1},
            }
        return {
            'results': [
                {'text': 'Weak case result', 'metadata': {'bluebook_cite': '111 U.S. 1'}, 'score': 0.18},
            ],
            'trace': {'average_score': 0.18, 'confidence_threshold': 0.40, 'result_count': 1},
        }

    monkeypatch.setattr(nd.tools, 'case_law_search', fake_case_search)
    monkeypatch.setattr(nd.tools, 'statute_search', lambda query, debug=False, aggressive=False: {'results': [], 'trace': {}} if debug else [])
    monkeypatch.setattr(nd.tools, 'regulation_search', lambda query, debug=False, aggressive=False: {'results': [], 'trace': {}} if debug else [])

    state = {
        'messages': [],
        'query': 'some low confidence query',
        'retrieved_cases': [],
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
        'retrieval_warnings': [],
        'retrieval_confidence': {},
    }

    out = nd.retrieve_node(state)
    assert len(calls) == 2
    assert calls[0]['aggressive'] is False
    assert calls[1]['aggressive'] is True
    assert out['retrieved_cases'][0]['score'] == 0.72
    assert out['retrieval_warnings'] == []
    assert out['retrieval_confidence']['case_law']['average_score'] == 0.72


def test_generate_answer_surfaces_retrieval_warning(monkeypatch):
    from src.agent import nodes as nd

    def fake_call(prompt):
        return 'The answer is cautious and grounded. Citations: 347 U.S. 483'

    monkeypatch.setattr(nd, '_call_ollama', fake_call)
    st = {
        'messages': [],
        'query': 'q',
        'retrieved_cases': [{'text': 'Brown v. Board of Education', 'metadata': {'bluebook_cite': '347 U.S. 483', 'parent_opinion_id': 1}, 'distance': 0.2, 'score': 0.9}],
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
        'retrieval_warnings': ['Low-confidence retrieval for case law: average relevance 0.18 below threshold 0.40 after rewrite/expansion.'],
        'retrieval_confidence': {},
    }

    out = nd.generate_answer(st)
    assert 'RETRIEVAL WARNING' in out['final_answer']
    assert 'Low-confidence retrieval for case law' in out['final_answer']


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
