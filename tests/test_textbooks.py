from src.data.preprocessor import Preprocessor


def test_clean_textbooks_chunks_pdf_pages_with_metadata(monkeypatch, tmp_path):
    raw_dir = tmp_path / 'textbooks'
    raw_dir.mkdir()
    (raw_dir / 'sample.pdf').write_bytes(b'%PDF-1.4 fake')

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdf:
        metadata = {
            'Title': 'Sample Textbook',
            'Author': 'Ada Author',
            'Subject': 'Legal Doctrine',
            'Creator': 'pytest',
            'Producer': 'pytest',
        }

        pages = [
            FakePage('Chapter 1 Introduction\nNegligence is the failure to exercise reasonable care.'),
            FakePage('Negligence requires duty, breach, causation, and damages.'),
        ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr('src.data.preprocessor.pdfplumber.open', lambda path: FakePdf())

    pre = Preprocessor()
    df = pre.clean_textbooks(raw_dir=str(raw_dir))

    assert not df.empty
    assert set(['textbook_id', 'book_title', 'section_heading', 'page_number', 'chunk_index']).issubset(df.columns)
    assert df.iloc[0]['book_title'] == 'Sample Textbook'
    assert df.iloc[0]['section_heading']
    assert df.iloc[0]['page_number'] == 1
