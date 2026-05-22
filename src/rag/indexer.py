import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from src.config import Config
from src.rag.embedder import Embedder
from src.observability.logger import get_logger
import os
from src.rag.bm25_index import BM25Index

logger = get_logger(__name__)

class Indexer:
    def __init__(self):
        persist = Config.CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=persist)
        self.cases = self._get_collection(Config.CHROMA_CASES_COLLECTION)
        self.statutes = self._get_collection(Config.CHROMA_STATUTES_COLLECTION)
        self.regs = self._get_collection(Config.CHROMA_REGULATIONS_COLLECTION)
        self.titles = self._get_collection(Config.CHROMA_TITLES_COLLECTION)
        self.embedder = Embedder()
        # BM25 indexes persisted alongside Chroma DB
        self.bm25_cases = BM25Index(persist, "cases")
        self.bm25_statutes = BM25Index(persist, "statutes")
        self.bm25_regs = BM25Index(persist, "regs")
        self.bm25_titles = BM25Index(persist, "titles")

    def _get_collection(self, name):
        try:
            return self.client.create_collection(name)
        except Exception:
            try:
                return self.client.get_collection(name)
            except Exception:
                return self.client.create_collection(name)

    def _upsert_in_batches(self, collection, ids, documents, metadatas, embeddings=None, batch_size=256, desc="Upserting"):
        from tqdm import tqdm as _tqdm

        for i in _tqdm(range(0, len(ids), batch_size), desc=desc, unit="batch"):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            batch_embeddings = None if embeddings is None else embeddings[i:i + batch_size]
            try:
                if batch_embeddings is None:
                    collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                else:
                    collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embeddings)
            except Exception as e:
                logger.warning(f"Batch upsert failed for {desc.lower()}: {e}")

    def index_cases(self, df):
        if df.empty:
            logger.info("No case data to index")
            return
        ids = []
        texts = []
        metadatas = []
        for _, row in tqdm(df.iterrows(), desc="Indexing Cases", total=len(df), unit="doc"):
            text = row.get('text')
            if not text or not str(text).strip():
                continue
            _id = f"case_{row.parent_opinion_id}_{row.chunk_index}"
            citations_value = row.get('citations')
            if citations_value is None:
                citations_str = ''
            else:
                try:
                    citations_str = ','.join(str(c) for c in citations_value if str(c).strip())
                except TypeError:
                    citations_str = str(citations_value)
            ids.append(_id)
            texts.append(text)
            metadatas.append({
                'case_name': row.case_name,
                'court': row.court,
                'date_filed': row.date_filed,
                'docket_number': row.docket_number,
                'bluebook_cite': row.bluebook_cite,
                'chunk_index': row.chunk_index,
                'parent_opinion_id': row.parent_opinion_id,
                'citations': citations_str
            })
        if not ids:
            logger.info("No case chunks to index")
            return
        # Upsert in batches to reduce peak memory during embedding
        batch_size = 256
        all_ids = []
        all_texts = []
        from tqdm import tqdm as _tqdm
        for i in _tqdm(range(0, len(ids), batch_size), desc="Embedding+Upserting Cases", unit="batch"):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            embeddings = self.embedder.embed(batch_texts)
            try:
                self.cases.upsert(ids=batch_ids, documents=batch_texts, metadatas=batch_metas, embeddings=embeddings)
            except Exception as e:
                logger.warning(f"Batch upsert failed for cases: {e}; retrying without embeddings")
                try:
                    self.cases.upsert(ids=batch_ids, documents=batch_texts, metadatas=batch_metas)
                except Exception as e2:
                    logger.error(f"Fallback batch upsert also failed: {e2}")
            all_ids.extend(batch_ids)
            all_texts.extend(batch_texts)
        logger.info(f"Indexed {len(all_ids)} case chunks (batched)")
        ids = all_ids
        texts = all_texts
        # build BM25 on chunk texts
        try:
            self.bm25_cases.build(ids, texts)
            logger.info("Built BM25 index for cases")
        except Exception as e:
            logger.warning(f"Could not build BM25 for cases: {e}")

        # build title entries: unique parent opinion id -> case name
        try:
            title_ids = []
            title_texts = []
            title_metas = []
            seen = set()
            for m in metadatas:
                pid = m.get('parent_opinion_id')
                name = m.get('case_name') or ''
                if pid and pid not in seen:
                    seen.add(pid)
                    tid = f"title_case_{pid}"
                    title_ids.append(tid)
                    title_texts.append(name)
                    title_metas.append({'parent_opinion_id': pid, 'case_name': name, 'source': 'case'})
            if title_ids:
                self._upsert_in_batches(self.titles, title_ids, title_texts, title_metas, batch_size=256, desc="Upserting Case Titles")
                try:
                    self.bm25_titles.build(title_ids, title_texts)
                except Exception:
                    # ignore bm25 build failure for titles
                    pass
                logger.info(f"Indexed {len(title_ids)} case titles")
        except Exception as e:
            logger.warning(f"Could not index case titles: {e}")

    def index_statutes(self, df):
        if df.empty:
            logger.info("No statute data to index")
            return
        ids = []
        texts = []
        metadatas = []
        for _, row in tqdm(df.iterrows(), desc="Indexing Statutes", total=len(df), unit="doc"):
            # include granule/package id when available to ensure uniqueness across granules
            gid = None
            try:
                gid = row.get('granule_id') or row.get('package_id')
            except Exception:
                gid = None
            if gid:
                _id = f"stat_{row.title_number}_{row.section_number}_{gid}_{row.chunk_index}"
            else:
                _id = f"stat_{row.title_number}_{row.section_number}_{row.chunk_index}"
            ids.append(_id)
            texts.append(row.section_text)
            metadatas.append({
                'title_number': row.title_number,
                'section_number': row.section_number,
                'section_heading': row.section_heading,
                'usc_citation': row.usc_citation,
            })
        if not ids:
            logger.info("No statute chunks to index")
            return
        # Upsert in batches
        batch_size = 256
        all_ids = []
        all_texts = []
        from tqdm import tqdm as _tqdm
        for i in _tqdm(range(0, len(ids), batch_size), desc="Embedding+Upserting Statutes", unit="batch"):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            embeddings = self.embedder.embed(batch_texts)
            try:
                self.statutes.upsert(ids=batch_ids, documents=batch_texts, metadatas=batch_metas, embeddings=embeddings)
            except Exception as e:
                logger.warning(f"Batch upsert failed for statutes: {e}")
            all_ids.extend(batch_ids)
            all_texts.extend(batch_texts)
        logger.info(f"Indexed {len(all_ids)} statute chunks (batched)")
        ids = all_ids
        texts = all_texts
        try:
            self.bm25_statutes.build(ids, texts)
            logger.info("Built BM25 index for statutes")
        except Exception as e:
            logger.warning(f"Could not build BM25 for statutes: {e}")

        # index statute titles/headings
        try:
            title_ids = []
            title_texts = []
            title_metas = []
            for m, _id in zip(metadatas, ids):
                heading = m.get('section_heading') or ''
                if heading:
                    tid = f"title_stat_{_id}"
                    title_ids.append(tid)
                    title_texts.append(heading)
                    title_metas.append({'source': 'statute', 'ref_id': _id})
            if title_ids:
                self._upsert_in_batches(self.titles, title_ids, title_texts, title_metas, batch_size=256, desc="Upserting Statute Titles")
                try:
                    self.bm25_titles.build(title_ids, title_texts)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not index statute titles: {e}")

    def index_regulations(self, df):
        if df.empty:
            logger.info("No regulation data to index")
            return
        ids = []
        texts = []
        metadatas = []
        for _, row in tqdm(df.iterrows(), desc="Indexing Regulations", total=len(df), unit="doc"):
            _id = f"reg_{row.cfr_title}_{row.cfr_part}_{row.cfr_section}_{row.chunk_index}"
            ids.append(_id)
            texts.append(row.section_text)
            metadatas.append({
                'cfr_title': row.cfr_title,
                'cfr_part': row.cfr_part,
                'cfr_section': row.cfr_section,
                'cfr_citation': row.cfr_citation,
                'section_heading': row.section_heading,
            })
        if not ids:
            logger.info("No regulation chunks to index")
            return
        # Upsert in batches
        batch_size = 256
        all_ids = []
        all_texts = []
        from tqdm import tqdm as _tqdm
        for i in _tqdm(range(0, len(ids), batch_size), desc="Embedding+Upserting Regs", unit="batch"):
            batch_ids = ids[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            embeddings = self.embedder.embed(batch_texts)
            try:
                self.regs.upsert(ids=batch_ids, documents=batch_texts, metadatas=batch_metas, embeddings=embeddings)
            except Exception as e:
                logger.warning(f"Batch upsert failed for regs: {e}")
            all_ids.extend(batch_ids)
            all_texts.extend(batch_texts)
        logger.info(f"Indexed {len(all_ids)} regulation chunks (batched)")
        ids = all_ids
        texts = all_texts
        try:
            self.bm25_regs.build(ids, texts)
            logger.info("Built BM25 index for regulations")
        except Exception as e:
            logger.warning(f"Could not build BM25 for regulations: {e}")

        # index regulation titles/headings
        try:
            title_ids = []
            title_texts = []
            title_metas = []
            for m, _id in zip(metadatas, ids):
                heading = m.get('section_heading') or ''
                if heading:
                    tid = f"title_reg_{_id}"
                    title_ids.append(tid)
                    title_texts.append(heading)
                    title_metas.append({'source': 'regulation', 'ref_id': _id})
            if title_ids:
                self._upsert_in_batches(self.titles, title_ids, title_texts, title_metas, batch_size=256, desc="Upserting Regulation Titles")
                try:
                    self.bm25_titles.build(title_ids, title_texts)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Could not index regulation titles: {e}")

    def index_statute_titles(self, df):
        if df.empty:
            logger.info("No statute data to title-index")
            return 0

        title_ids = []
        title_texts = []
        title_metas = []
        for _, row in tqdm(df.iterrows(), desc="Indexing Statute Titles", total=len(df), unit="doc"):
            gid = None
            try:
                gid = row.get('granule_id') or row.get('package_id')
            except Exception:
                gid = None
            if gid and str(gid).strip().lower() != 'nan':
                chunk_id = f"stat_{row.title_number}_{row.section_number}_{gid}_{row.chunk_index}"
            else:
                chunk_id = f"stat_{row.title_number}_{row.section_number}_{row.chunk_index}"

            heading = row.get('section_heading') if hasattr(row, 'get') else None
            if heading and str(heading).strip() and str(heading).strip().lower() != 'nan':
                title_ids.append(f"title_stat_{chunk_id}")
                title_texts.append(heading)
                title_metas.append({'source': 'statute', 'ref_id': chunk_id})

        if not title_ids:
            logger.info("No statute titles to index")
            return 0

        self._upsert_in_batches(self.titles, title_ids, title_texts, title_metas, batch_size=256, desc="Upserting Statute Titles")
        try:
            self.bm25_titles.build(title_ids, title_texts)
        except Exception as e:
            logger.warning(f"Could not rebuild BM25 for statute titles: {e}")

        logger.info(f"Indexed {len(title_ids)} statute titles")
        return len(title_ids)

    def get_collection_stats(self):
        return {
            'cases': getattr(self.cases, 'count', lambda: 0)(),
            'statutes': getattr(self.statutes, 'count', lambda: 0)(),
            'regulations': getattr(self.regs, 'count', lambda: 0)(),
        }

    def get_entity_counts(self):
        return {
            'cases': self._count_unique_entities(self.cases, lambda meta: meta.get('parent_opinion_id')),
            'statutes': self._count_unique_entities(
                self.statutes,
                lambda meta: (meta.get('title_number'), meta.get('section_number')),
            ),
            'regulations': self._count_unique_entities(
                self.regs,
                lambda meta: (meta.get('cfr_title'), meta.get('cfr_part'), meta.get('cfr_section')),
            ),
        }

    def _count_unique_entities(self, collection, key_fn):
        try:
            payload = collection.get()
        except Exception:
            return 0

        metadatas = payload.get('metadatas', []) or []
        unique_keys = set()
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
            key = key_fn(meta)
            if key is None:
                continue
            if isinstance(key, tuple):
                if not any(str(part).strip() for part in key):
                    continue
            elif not str(key).strip():
                continue
            unique_keys.add(key)
        return len(unique_keys)

    def create_session_collection(self, session_id: str):
        name = f"session_{session_id}"
        try:
            return self.client.create_collection(name)
        except Exception:
            try:
                return self.client.get_collection(name)
            except Exception:
                return self.client.create_collection(name)

    def delete_session_collection(self, session_id: str):
        name = f"session_{session_id}"
        try:
            self.client.delete_collection(name)
        except Exception:
            logger.info(f"Could not delete session collection {name}")

if __name__ == '__main__':
    import pandas as pd
    import os
    
    idx = Indexer()
    processed_dir = os.path.join(os.getcwd(), "data", "processed")
    
    # Index cases from parquet
    cases_file = os.path.join(processed_dir, "cases.parquet")
    if os.path.exists(cases_file):
        try:
            df_cases = pd.read_parquet(cases_file)
            idx.index_cases(df_cases)
        except Exception as e:
            logger.warning(f"Could not index cases: {e}")
    else:
        logger.info("No cases.parquet found")
    
    # Index statutes from parquet
    statutes_file = os.path.join(processed_dir, "statutes.parquet")
    if os.path.exists(statutes_file):
        try:
            df_statutes = pd.read_parquet(statutes_file)
            idx.index_statutes(df_statutes)
        except Exception as e:
            logger.warning(f"Could not index statutes: {e}")
    else:
        logger.info("No statutes.parquet found")
    
    # Index regulations from parquet
    regulations_file = os.path.join(processed_dir, "regulations.parquet")
    if os.path.exists(regulations_file):
        try:
            df_regs = pd.read_parquet(regulations_file)
            idx.index_regulations(df_regs)
        except Exception as e:
            logger.warning(f"Could not index regulations: {e}")
    else:
        logger.info("No regulations.parquet found")
    
    stats = idx.get_collection_stats()
    print(f"Index stats: {stats}")
