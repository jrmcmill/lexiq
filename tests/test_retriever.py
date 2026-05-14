import pytest
from unittest.mock import MagicMock
from src.rag.retriever import Retriever

class DummyCollection:
    def query(self, **kwargs):
        return {'documents': ['doc1','doc2'], 'metadatas': [{'bluebook_cite':'a'},{'bluebook_cite':'b'}], 'distances': [0.1,0.2]}

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
    monkeypatch.setattr(retriever, 'reranker', MagicMock(rerank=lambda q, r: r))
    res = retriever.retrieve_cases('test')
    assert isinstance(res, list)
    assert len(res) == 2

def test_court_filter_where_clause(monkeypatch, retriever):
    # ensure no exceptions when court_filter provided
    res = retriever.retrieve_cases('q', court_filter='D.C. Cir.')
    assert isinstance(res, list)

def test_retrieve_session_docs_no_collection(monkeypatch, retriever):
    retriever.indexer.client.get_collection.side_effect = Exception('no collection')
    res = retriever.retrieve_session_docs('q','nosess')
    assert res == []
