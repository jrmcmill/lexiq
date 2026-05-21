import pytest
from unittest.mock import MagicMock
from src.rag.retriever import Retriever

class DummyCollection:
    def query(self, **kwargs):
        return {'documents': [['doc1','doc2']], 'metadatas': [[{'bluebook_cite':'a'},{'bluebook_cite':'b'}]], 'distances': [[0.1,0.2]]}

class DummyIndexer:
    def __init__(self):
        self.cases = DummyCollection()

@pytest.fixture
def retriever(monkeypatch):
    r = Retriever()
    # monkeypatch the indexer to use dummy
    r.indexer = MagicMock()
    r.indexer.cases = DummyCollection()
    r.indexer.client = MagicMock()
    return r

def test_retrieve_cases_rerank(monkeypatch, retriever):
    # monkeypatch reranker to return as-is
    monkeypatch.setattr(retriever, 'reranker', MagicMock(rerank=lambda q, r: [dict(x, rerank_score=1.0) for x in r]))
    res = retriever.retrieve_cases('test')
    assert isinstance(res, list)
    assert len(res) == 2


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
    retriever.bm25_cases = MagicMock(top_ids=lambda q, top_k: [])
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
