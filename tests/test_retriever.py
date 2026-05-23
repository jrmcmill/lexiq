import pytest
from unittest.mock import MagicMock
from src.rag.retriever import Retriever

class DummyCollection:
    def query(self, **kwargs):
        return {'documents': [['doc1','doc2']], 'metadatas': [[{'bluebook_cite':'a'},{'bluebook_cite':'b'}]], 'distances': [[0.1,0.2]]}

class DummyIndexer:
    def __init__(self):
        self.cases = DummyCollection()
        self.statutes = DummyCollection()
        self.regs = DummyCollection()
        self.textbooks = DummyCollection()
        self.titles = None

@pytest.fixture
def retriever(monkeypatch):
    r = Retriever()
    # monkeypatch the indexer to use dummy
    r.indexer = MagicMock()
    r.indexer.cases = DummyCollection()
    r.indexer.statutes = DummyCollection()
    r.indexer.regs = DummyCollection()
    r.indexer.textbooks = DummyCollection()
    r.indexer.titles = None
    r.indexer.client = MagicMock()
    r.bm25_cases = MagicMock(query=lambda q, top_k: [])
    r.bm25_statutes = MagicMock(query=lambda q, top_k: [])
    r.bm25_regs = MagicMock(query=lambda q, top_k: [])
    r.bm25_textbooks = MagicMock(query=lambda q, top_k: [])
    r.bm25_titles = None
    return r

def test_retrieve_cases_rerank(monkeypatch, retriever):
    # monkeypatch reranker to return as-is
    monkeypatch.setattr(retriever, 'reranker', MagicMock(rerank=lambda q, r: [dict(x, rerank_score=1.0) for x in r]))
    res = retriever.retrieve_cases('test')
    assert isinstance(res, list)
    assert len(res) == 2
    assert all('score' in item for item in res)


def test_hybrid_score_prefers_strong_title_signal():
    r = Retriever()
    case_with_title = r._combine_scores('cases', 'smith v jones', semantic_score=0.20, bm25_score=0.10, title_score=0.90)
    case_without_title = r._combine_scores('cases', 'smith v jones', semantic_score=0.70, bm25_score=0.10, title_score=0.0)
    assert case_with_title > case_without_title


def test_finalize_results_blends_hybrid_and_rerank(monkeypatch, retriever):
    candidates = [
        {'text': 'doc1', 'metadata': {'bluebook_cite': 'A', 'case_name': 'Doc One', 'parent_opinion_id': 1}, 'hybrid_score': 0.9, 'distance': 0.1, 'semantic_score': 0.8, 'bm25_score': 0.9, 'title_score': 0.2},
        {'text': 'doc2', 'metadata': {'bluebook_cite': 'B', 'case_name': 'Doc Two', 'parent_opinion_id': 2}, 'hybrid_score': 0.2, 'distance': 0.2, 'semantic_score': 0.2, 'bm25_score': 0.1, 'title_score': 0.0},
    ]
    monkeypatch.setattr(retriever, 'reranker', MagicMock(rerank=lambda q, r, top_k=None: [dict(r[1], rerank_score=0.95), dict(r[0], rerank_score=0.30)]))
    res = retriever._finalize_results('query', candidates, 'cases', 2)
    assert len(res) == 2
    assert res[0]['score'] >= res[1]['score']
    assert 'hybrid_score' in res[0]
    assert 'rerank_score' in res[0]


def test_finalize_results_prefers_higher_authority_case(monkeypatch, retriever):
    candidates = [
        {
            'text': 'lower authority but strong topical match',
            'metadata': {
                'bluebook_cite': '111 F. Supp. 2d 1',
                'case_name': 'Lower Court Case',
                'parent_opinion_id': 11,
                'court': 'District Court',
            },
            'hybrid_score': 0.95,
            'distance': 0.1,
            'semantic_score': 0.95,
            'bm25_score': 0.9,
            'title_score': 0.1,
        },
        {
            'text': 'higher authority with slightly weaker topical match',
            'metadata': {
                'bluebook_cite': '600 U.S. 1',
                'case_name': 'Supreme Court Case',
                'parent_opinion_id': 12,
                'court': 'Supreme Court of the United States',
            },
            'hybrid_score': 0.85,
            'distance': 0.1,
            'semantic_score': 0.85,
            'bm25_score': 0.8,
            'title_score': 0.1,
        },
    ]
    monkeypatch.setattr(retriever, 'reranker', MagicMock(rerank=lambda q, r, top_k=None: [dict(r[0], rerank_score=0.92), dict(r[1], rerank_score=0.92)]))

    res = retriever._finalize_results('query', candidates, 'cases', 2)

    assert res[0]['metadata']['bluebook_cite'] == '600 U.S. 1'
    assert res[0]['authority_tier'] == 'high'
    assert res[0]['authority_score'] > res[1]['authority_score']


def test_expand_query_variants():
    from src.rag.retriever import Retriever
    r = Retriever()
    variants = r._expand_query_variants('gerrymandering in louisiana')
    # should include original and at least one topical expansion term
    assert any('gerrymander' in v or 'vote dilution' in v or 'section 2' in v for v in variants)

def test_retrieve_cases_deduplicates(monkeypatch, retriever):
    class DupCollection:
        def query(self, **kwargs):
            return {
                'documents': [['doc1', 'doc1', 'doc2']],
                'metadatas': [[
                    {'bluebook_cite': 'A', 'parent_opinion_id': 1},
                    {'bluebook_cite': 'A', 'parent_opinion_id': 1},
                    {'bluebook_cite': 'B', 'parent_opinion_id': 2},
                ]],
                'distances': [[0.1, 0.2, 0.3]],
            }

    retriever.indexer.cases = DupCollection()
    retriever.bm25_cases = MagicMock(query=lambda q, top_k: [])
    monkeypatch.setattr(retriever, 'reranker', MagicMock(rerank=lambda q, r: [dict(x, rerank_score=1.0) for x in r]))

    res = retriever.retrieve_cases('test')

    assert len(res) == 2
    assert [item['metadata'].get('bluebook_cite') for item in res] == ['A', 'B']

def test_court_filter_where_clause(monkeypatch, retriever):
    # ensure no exceptions when court_filter provided
    res = retriever.retrieve_cases('q', court_filter='D.C. Cir.')
    assert isinstance(res, list)

def test_retrieve_session_docs_no_collection(monkeypatch, retriever):
    retriever.indexer.client.get_collection.side_effect = Exception('no collection')
    res = retriever.retrieve_session_docs('q','nosess')
    assert res == []


def test_expand_with_citation_graph_merges_neighbor_chunks(monkeypatch, retriever):
    retriever.indexer.cases = MagicMock()
    retriever.indexer.cases.get = MagicMock(return_value={
        'ids': ['case_2_0'],
        'documents': ['neighbor doc'],
        'metadatas': [{'bluebook_cite': 'B', 'parent_opinion_id': 2}],
    })

    class FakeGraph:
        def __init__(self):
            self.nodes = {'case:1': True}

        def node_for_result(self, result, source_kind):
            return 'case:1'

        def expand(self, seed_node_ids, max_hops=1, max_nodes=12):
            return [{
                'node_id': 'case:2',
                'kind': 'case',
                'label': 'B',
                'metadata': {'parent_opinion_id': 2},
                'chunk_ids': ['case_2_0'],
                'distance': 1,
                'score': 0.5,
                'relation': 'case_cites',
                'direction': 'out',
                'seed_node_id': 'case:1',
            }]

        def fetch_spec(self, node_id):
            return {
                'kind': 'case',
                'label': 'B',
                'metadata': {'parent_opinion_id': 2},
                'chunk_ids': ['case_2_0'],
                'node_id': node_id,
            }

    retriever.citation_graph = FakeGraph()
    base = [{
        'text': 'seed doc',
        'metadata': {'bluebook_cite': 'A', 'parent_opinion_id': 1},
        'distance': 0.1,
        'semantic_score': 0.9,
        'bm25_score': 0.8,
        'title_score': 0.0,
        'source_id': 'case_1_0',
        'provenance': {'cite': 'A'},
    }]

    merged, trace = retriever._expand_with_citation_graph(base, 'cases')

    assert trace['graph_expanded'] == 1
    assert any(item['metadata'].get('bluebook_cite') == 'B' for item in merged)
