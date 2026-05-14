import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from src.config import Config
from src.rag.embedder import Embedder
from src.observability.logger import get_logger
import os

logger = get_logger(__name__)

class Indexer:
    def __init__(self):
        persist = Config.CHROMA_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=persist)
        self.cases = self._get_collection(Config.CHROMA_CASES_COLLECTION)
        self.statutes = self._get_collection(Config.CHROMA_STATUTES_COLLECTION)
        self.regs = self._get_collection(Config.CHROMA_REGULATIONS_COLLECTION)
        self.embedder = Embedder()

    def _get_collection(self, name):
        try:
            return self.client.create_collection(name)
        except Exception:
            try:
                return self.client.get_collection(name)
            except Exception:
                return self.client.create_collection(name)

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
        embeddings = self.embedder.embed(texts)
        try:
            self.cases.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            logger.info(f"Indexed {len(ids)} case chunks")
        except Exception as e:
            logger.error(f"Error upserting cases: {e}")
            try:
                self.cases.upsert(ids=ids, documents=texts, metadatas=metadatas)
            except Exception as e2:
                logger.error(f"Fallback upsert also failed: {e2}")

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
        embeddings = self.embedder.embed(texts)
        try:
            self.statutes.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            logger.info(f"Indexed {len(ids)} statute chunks")
        except Exception as e:
            logger.error(f"Error upserting statutes: {e}")

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
        embeddings = self.embedder.embed(texts)
        try:
            self.regs.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            logger.info(f"Indexed {len(ids)} regulation chunks")
        except Exception as e:
            logger.error(f"Error upserting regulations: {e}")

    def get_collection_stats(self):
        return {
            'cases': getattr(self.cases, 'count', lambda: 0)(),
            'statutes': getattr(self.statutes, 'count', lambda: 0)(),
            'regulations': getattr(self.regs, 'count', lambda: 0)(),
        }

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
