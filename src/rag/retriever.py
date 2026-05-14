from src.rag.indexer import Indexer
from src.rag.embedder import Embedder
from src.rag.reranker import Reranker
from src.config import Config
from src.observability.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        self.indexer = Indexer()
        self.embedder = Embedder()
        self.reranker = Reranker()

    def retrieve_cases(self, query: str, n_results: int = 20, court_filter: str | None = None,
                       date_after: str | None = None, date_before: str | None = None) -> list[dict]:
        try:
            emb = self.embedder.embed_query(query)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                return []
            
            # Build where clause only if filters are present
            ands = []
            if court_filter:
                ands.append({"court": {"$eq": court_filter}})
            if date_after:
                ands.append({"date_filed": {"$gte": date_after}})
            if date_before:
                ands.append({"date_filed": {"$lte": date_before}})
            
            # Query with or without where clause
            if ands:
                res = self.indexer.cases.query(query_embeddings=[emb], n_results=n_results, where={"$and": ands})
            else:
                res = self.indexer.cases.query(query_embeddings=[emb], n_results=n_results)
            docs = []
            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []
            
            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    docs.append({'text': d, 'metadata': m, 'distance': dist})
            
            reranked = self.reranker.rerank(query, docs)
            out = []
            for r in reranked:
                if isinstance(r, dict):
                    out.append({
                        'text': r.get('text', ''),
                        'metadata': r.get('metadata', {}),
                        'score': r.get('rerank_score'),
                        'distance': r.get('distance'),
                    })
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_cases: {str(e)}")
            return []

    def retrieve_statutes(self, query: str, n_results: int = 20) -> list[dict]:
        try:
            emb = self.embedder.embed_query(query)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                return []
            
            res = self.indexer.statutes.query(query_embeddings=[emb], n_results=n_results)
            out = []
            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []
            
            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    out.append({'text': d, 'metadata': m, 'score': None, 'distance': dist})
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_statutes: {str(e)}")
            return []

    def retrieve_regulations(self, query: str, n_results: int = 20) -> list[dict]:
        try:
            emb = self.embedder.embed_query(query)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                return []
            
            res = self.indexer.regs.query(query_embeddings=[emb], n_results=n_results)
            out = []
            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []
            
            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    out.append({'text': d, 'metadata': m, 'score': None, 'distance': dist})
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_regulations: {str(e)}")
            return []

    def retrieve_session_docs(self, query: str, session_id: str, n_results: int = 10) -> list[dict]:
        try:
            emb = self.embedder.embed_query(query)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                return []
            
            try:
                coll = self.indexer.client.get_collection(f"session_{session_id}")
            except Exception:
                return []
            
            res = coll.query(query_embeddings=[emb], n_results=n_results)
            out = []
            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []
            
            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    out.append({'text': d, 'metadata': m, 'score': None, 'distance': dist})
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_session_docs: {str(e)}")
            return []

if __name__ == '__main__':
    r = Retriever()
    print('Retriever ready')
