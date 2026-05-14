import tempfile
import os
from src.rag.indexer import Indexer
from src.rag.embedder import Embedder
from src.documents.parser import DocumentParser
from src.observability.logger import get_logger
import time

logger = get_logger(__name__)

class SessionDocumentStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.indexer = Indexer()
        self.embedder = Embedder()
        self.parser = DocumentParser()
        self.collection = self.indexer.create_session_collection(session_id)
        self.uploaded_files = []

    def add_document(self, file_bytes: bytes, filename: str) -> dict:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        chunks = self.parser.parse(tmp.name, filename)
        texts = [c['text'] for c in chunks]
        embeddings = self.embedder.embed(texts)
        ids = []
        metadatas = []
        docs = []
        for c, emb in zip(chunks, embeddings):
            _id = f"{filename}_{c['chunk_index']}"
            ids.append(_id)
            docs.append(c['text'])
            metadatas.append({
                'filename': filename,
                'page_number': c.get('page_number'),
                'chunk_index': c.get('chunk_index')
            })
        try:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        except Exception:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
        summary = {'filename': filename, 'num_chunks': len(chunks), 'token_estimate': sum(c.get('token_estimate',0) for c in chunks)}
        self.uploaded_files.append({'filename': filename, 'num_chunks': len(chunks), 'upload_time': time.time(), 'file_size_bytes': os.path.getsize(tmp.name)})
        return summary

    def remove_document(self, filename: str) -> None:
        try:
            self.collection.delete(where={"filename": {"$eq": filename}})
        except Exception:
            # fallback: iterate
            try:
                ids = [m['id'] for m in self.collection.peek(1000).get('ids', [])]
            except Exception:
                ids = []
        self.uploaded_files = [f for f in self.uploaded_files if f['filename'] != filename]

    def list_documents(self):
        return self.uploaded_files

    def teardown(self):
        try:
            self.indexer.delete_session_collection(self.session_id)
        except Exception:
            logger.info('Teardown best-effort')

if __name__ == '__main__':
    s = SessionDocumentStore('testsess')
    print('Session store ready')
