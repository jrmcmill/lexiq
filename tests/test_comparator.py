from src.documents.comparator import DocumentComparator

def test_extract_full_text_and_compare(monkeypatch):
    comp = DocumentComparator()
    chunks_a = [{'chunk_index':0,'text':'A1'},{'chunk_index':1,'text':'A2'}]
    chunks_b = [{'chunk_index':0,'text':'B1'},{'chunk_index':1,'text':'B2'}]
    full_a = comp.extract_full_text(chunks_a)
    assert 'A1' in full_a
    # patch requests.post
    class DummyResp:
        def __init__(self):
            self.text = '1. Similar\n2. Different\n3. Missing A\n4. Missing B\n5. Risks\n6. Assessment'
        def raise_for_status(self):
            return
    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **k: DummyResp())
    res = comp.compare('A', 'B')
    assert 'similarities' in res
