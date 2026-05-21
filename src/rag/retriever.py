from src.rag.indexer import Indexer
from src.rag.embedder import Embedder
from src.rag.reranker import Reranker
from src.rag.bm25_index import BM25Index
from src.config import Config
from src.observability.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    TOPICAL_QUERY_EXPANSIONS = {
        'gerrymander': ['vote dilution', 'racial gerrymandering', 'redistricting', 'districting', 'section 2'],
        'gerrymandering': ['vote dilution', 'racial gerrymandering', 'redistricting', 'districting', 'section 2'],
        'vote dilution': ['gerrymandering', 'racial gerrymandering', 'section 2', 'voting rights act'],
        'racial gerrymandering': ['gerrymandering', 'vote dilution', 'equal protection', 'redistricting'],
        'partisan gerrymandering': ['gerrymandering', 'redistricting', 'districting', 'equal protection'],
        'redistricting': ['districting', 'gerrymandering', 'vote dilution'],
        'districting': ['redistricting', 'gerrymandering', 'vote dilution'],
        'section 2': ['voting rights act', '52 usc 10301', '42 usc 1973', 'vote dilution'],
        'section 5': ['voting rights act', 'preclearance', '52 usc 10304'],
        'voting rights act': ['section 2', 'section 5', 'vote dilution'],
        'equal protection': ['gerrymandering', 'racial gerrymandering', 'redistricting'],
        'qualified immunity': ['clearly established', 'civil rights', '1983'],
        'standing': ['justiciability', 'injury in fact', 'mootness', 'ripeness'],
        'free speech': ['first amendment', 'content based', 'prior restraint'],
    }
    MAX_QUERY_VARIANTS = 5

    def __init__(self):
        self.indexer = Indexer()
        self.embedder = Embedder()
        self.reranker = Reranker()
        # load BM25 indexes (if present)
        persist = Config.CHROMA_PERSIST_DIR
        self.bm25_cases = BM25Index(persist, "cases")
        self.bm25_statutes = BM25Index(persist, "statutes")
        self.bm25_regs = BM25Index(persist, "regs")
        self.bm25_titles = BM25Index(persist, "titles")
        try:
            self.bm25_cases.load()
            self.bm25_statutes.load()
            self.bm25_regs.load()
            self.bm25_titles.load()
        except Exception:
            pass

    def retrieve_cases(self, query: str, n_results: int = 20, court_filter: str | None = None,
                       date_after: str | None = None, date_before: str | None = None) -> list[dict]:
        try:
            # Build where clause only if filters are present
            ands = []
            if court_filter:
                ands.append({"court": {"$eq": court_filter}})
            if date_after:
                ands.append({"date_filed": {"$gte": date_after}})
            if date_before:
                ands.append({"date_filed": {"$lte": date_before}})

            query_variants = self._expand_query_variants(query)
            docs = self._collect_candidates(
                collection=self.indexer.cases,
                query_variants=query_variants,
                n_results=n_results,
                where={"$and": ands} if ands else None,
                bm25_index=self.bm25_cases,
            )

            if not docs:
                logger.warning(f"No case results above relevance threshold {Config.RETRIEVAL_MIN_DISTANCE}")
                return []

            reranked = self.reranker.rerank(query, docs)
            
            # Filter by reranker score threshold
            min_rerank = Config.RERANK_MIN_SCORE
            out = []
            for r in reranked:
                if isinstance(r, dict):
                    score = r.get('rerank_score')
                    if score is not None and score >= min_rerank:
                        out.append({
                            'text': r.get('text', ''),
                            'metadata': r.get('metadata', {}),
                            'score': score,
                            'distance': r.get('distance'),
                        })
            
            if len(out) < len(reranked):
                logger.info(f"Filtered out {len(reranked) - len(out)} low-scoring cases (score < {min_rerank})")
            
            if not out:
                logger.warning(f"No case results above reranking threshold {min_rerank}")

            return self._dedupe_results(out, self._case_result_key)
        except Exception as e:
            logger.error(f"Error in retrieve_cases: {str(e)}")
            return []

    def retrieve_statutes(self, query: str, n_results: int = 20) -> list[dict]:
        try:
            emb = self.embedder.embed_query(query)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                return []
            
            # embedding candidates
            res = self.indexer.statutes.query(query_embeddings=[emb], n_results=n_results)
            out = []
            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []

            # Filter by distance threshold
            min_dist = Config.RETRIEVAL_MIN_DISTANCE
            filtered_count = 0
            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    if dist <= min_dist:
                        out.append({'text': d, 'metadata': m, 'score': None, 'distance': dist})
                    else:
                        filtered_count += 1
            
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} low-quality statute results (distance > {min_dist})")
            
            # BM25 union
            try:
                bm25_ids = self.bm25_statutes.top_ids(query, top_k=n_results) if getattr(self, 'bm25_statutes', None) else []
                if bm25_ids:
                    fetched = self.indexer.statutes.get(ids=bm25_ids)
                    f_docs = fetched.get('documents', [])
                    f_metas = fetched.get('metadatas', [])
                    for fd, fm in zip(f_docs, f_metas):
                        if isinstance(fd, str) and isinstance(fm, dict):
                            out.append({'text': fd, 'metadata': fm, 'score': None, 'distance': None})
            except Exception:
                pass

            if not out:
                logger.warning(f"No statute results above relevance threshold {min_dist}")

            return self._dedupe_results(out, self._statute_result_key)
        except Exception as e:
            logger.error(f"Error in retrieve_statutes: {str(e)}")
            return []

    def retrieve_regulations(self, query: str, n_results: int = 20) -> list[dict]:
        try:
            emb = self.embedder.embed_query(query)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                return []
            
            # embedding candidates
            res = self.indexer.regs.query(query_embeddings=[emb], n_results=n_results)
            out = []
            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []

            # Filter by distance threshold
            min_dist = Config.RETRIEVAL_MIN_DISTANCE
            filtered_count = 0
            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    if dist <= min_dist:
                        out.append({'text': d, 'metadata': m, 'score': None, 'distance': dist})
                    else:
                        filtered_count += 1
            
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} low-quality regulation results (distance > {min_dist})")
            
            # BM25 union
            try:
                bm25_ids = self.bm25_regs.top_ids(query, top_k=n_results) if getattr(self, 'bm25_regs', None) else []
                if bm25_ids:
                    fetched = self.indexer.regs.get(ids=bm25_ids)
                    f_docs = fetched.get('documents', [])
                    f_metas = fetched.get('metadatas', [])
                    for fd, fm in zip(f_docs, f_metas):
                        if isinstance(fd, str) and isinstance(fm, dict):
                            out.append({'text': fd, 'metadata': fm, 'score': None, 'distance': None})
            except Exception:
                pass

            if not out:
                logger.warning(f"No regulation results above relevance threshold {min_dist}")

            return self._dedupe_results(out, self._regulation_result_key)
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
            
            # Filter by distance threshold
            min_dist = Config.RETRIEVAL_MIN_DISTANCE
            filtered_count = 0
            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    if dist <= min_dist:
                        out.append({'text': d, 'metadata': m, 'score': None, 'distance': dist})
                    else:
                        filtered_count += 1
            
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} low-quality session results (distance > {min_dist})")
            
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_session_docs: {str(e)}")
            return []

    def _dedupe_results(self, results: list[dict], key_fn):
        seen = set()
        deduped = []
        for result in results:
            key = key_fn(result)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    def _case_result_key(self, result: dict):
        meta = result.get('metadata', {}) if isinstance(result, dict) else {}
        if isinstance(meta, dict):
            parent_id = meta.get('parent_opinion_id')
            if parent_id:
                return ('case', parent_id)
            cite = meta.get('bluebook_cite') or meta.get('case_name')
            if cite:
                return ('case', cite)
        return ('case', result.get('text', ''))

    def _statute_result_key(self, result: dict):
        meta = result.get('metadata', {}) if isinstance(result, dict) else {}
        if isinstance(meta, dict):
            title = meta.get('title_number')
            section = meta.get('section_number')
            if title or section:
                return ('statute', title, section)
            cite = meta.get('usc_citation')
            if cite:
                return ('statute', cite)
        return ('statute', result.get('text', ''))

    def _regulation_result_key(self, result: dict):
        meta = result.get('metadata', {}) if isinstance(result, dict) else {}
        if isinstance(meta, dict):
            title = meta.get('cfr_title')
            part = meta.get('cfr_part')
            section = meta.get('cfr_section')
            if title or part or section:
                return ('regulation', title, part, section)
            cite = meta.get('cfr_citation')
            if cite:
                return ('regulation', cite)
        return ('regulation', result.get('text', ''))

    def retrieve_by_title(self, query: str, source: str = 'cases', n_results: int = 10) -> list[dict]:
        """Search titles (case names, statute headings, regulation headings) using BM25 and return linked documents."""
        try:
            bm25 = None
            coll = None
            if source == 'cases':
                bm25 = getattr(self, 'bm25_titles', None)
                coll = self.indexer.titles
            elif source == 'statutes' or source == 'regs':
                bm25 = getattr(self, 'bm25_titles', None)
                coll = self.indexer.titles
            else:
                bm25 = getattr(self, 'bm25_titles', None)
                coll = self.indexer.titles

            if not bm25:
                return []

            top = bm25.query(query, top_k=n_results)
            ids = [t[0] for t in top]
            if not ids:
                return []

            try:
                res = coll.get(ids=ids)
                docs = res.get('documents', [])
                metas = res.get('metadatas', [])
                out = []
                for d, m in zip(docs, metas):
                    out.append({'text': d, 'metadata': m})
                return out
            except Exception:
                return []
        except Exception as e:
            logger.error(f"Error in retrieve_by_title: {e}")
            return []

    def _expand_query_variants(self, query: str) -> list[str]:
        normalized = self._normalize_query(query)
        variants: list[str] = []
        seen: set[str] = set()

        def add_variant(value: str | None):
            cleaned = self._normalize_query(value or "")
            if not cleaned:
                return
            if cleaned in seen:
                return
            seen.add(cleaned)
            variants.append(cleaned)

        add_variant(query)

        topical_terms: list[str] = []
        for needle, expansions in self.TOPICAL_QUERY_EXPANSIONS.items():
            if needle in normalized:
                topical_terms.extend(expansions)

        if 'section 2' in normalized or '§ 2' in normalized or '52 usc 10301' in normalized or '42 usc 1973' in normalized:
            topical_terms.extend(['section 2', 'voting rights act', '52 usc 10301', '42 usc 1973', 'vote dilution'])
        if 'section 5' in normalized or '§ 5' in normalized or '52 usc 10304' in normalized:
            topical_terms.extend(['section 5', 'voting rights act', 'preclearance', '52 usc 10304'])

        combined_terms: list[str] = []
        for term in topical_terms:
            cleaned = self._normalize_query(term)
            if cleaned and cleaned not in seen:
                add_variant(cleaned)
                combined_terms.append(cleaned)
            if len(variants) >= self.MAX_QUERY_VARIANTS:
                return variants[:self.MAX_QUERY_VARIANTS]

        if combined_terms and len(variants) < self.MAX_QUERY_VARIANTS:
            combo = ' '.join([query] + combined_terms[:3])
            add_variant(combo)

        return variants[:self.MAX_QUERY_VARIANTS]

    def _normalize_query(self, value: str) -> str:
        return ' '.join((value or '').replace('§', 'section').split()).strip().lower()

    def _collect_candidates(self, collection, query_variants: list[str], n_results: int,
                            where: dict | None = None, bm25_index: BM25Index | None = None) -> list[dict]:
        docs: list[dict] = []
        min_dist = Config.RETRIEVAL_MIN_DISTANCE
        filtered_count = 0
        expansion_limit = max(5, n_results // 2)

        for variant_index, variant in enumerate(query_variants):
            limit = n_results if variant_index == 0 else expansion_limit
            emb = self.embedder.embed_query(variant)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                continue

            try:
                query_kwargs = {'query_embeddings': [emb], 'n_results': limit}
                if where:
                    query_kwargs['where'] = where
                res = collection.query(**query_kwargs)
            except Exception:
                continue

            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []

            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    if dist <= min_dist:
                        docs.append({'text': d, 'metadata': m, 'distance': dist, 'id': None})
                    else:
                        filtered_count += 1

        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} low-quality results (distance > {min_dist})")

        if bm25_index:
            bm25_ids: list[str] = []
            seen_ids: set[str] = set()
            for variant in query_variants:
                try:
                    top_ids = bm25_index.top_ids(variant, top_k=expansion_limit)
                except Exception:
                    top_ids = []
                for candidate_id in top_ids:
                    if candidate_id in seen_ids:
                        continue
                    seen_ids.add(candidate_id)
                    bm25_ids.append(candidate_id)
            try:
                if bm25_ids:
                    fetched = collection.get(ids=bm25_ids)
                    f_docs = fetched.get('documents', [])
                    f_metas = fetched.get('metadatas', [])
                    for fd, fm in zip(f_docs, f_metas):
                        if isinstance(fd, str) and isinstance(fm, dict):
                            docs.append({'text': fd, 'metadata': fm, 'distance': None, 'id': None})
            except Exception:
                pass

        return docs

if __name__ == '__main__':
    r = Retriever()
    print('Retriever ready')
