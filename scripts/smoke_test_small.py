"""
Small safe smoke test for LexIQ
- Fetch tiny samples from CourtListener, GovInfo (if API key), and eCFR
- Run Preprocessor.clean_* with small caps
- Perform a lightweight local index (uses small dummy embeddings)

Run locally from repo root (do not run automatically here).
"""

import os
import logging

# Ensure project root is on sys.path so `from src...` imports work
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import Config
from src.data.courtlistener import CourtListenerClient
from src.data.uscode import USCodeClient
from src.data.ecfr import ECFRClient
from src.data.preprocessor import Preprocessor

import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_test")


def safe_fetch_courtlistener():
    try:
        client = CourtListenerClient()
        res = client.fetch_opinions(query="test", max_pages=1)
        logger.info(f"CourtListener fetched: {len(res)} items")
    except Exception as e:
        logger.warning(f"CourtListener fetch failed: {e}")


def safe_fetch_uscode():
    try:
        client = USCodeClient()
        if not client.api_key:
            logger.info("GOVINFO_API_KEY not set; skipping USC fetch")
            return []
        res = client.fetch_sections(max_pages=1)
        logger.info(f"USC fetched: {len(res)} granules")
        return res
    except Exception as e:
        logger.warning(f"USC fetch failed: {e}")
        return []


def safe_fetch_ecfr():
    try:
        client = ECFRClient()
        titles = client.list_titles()
        saved = 0
        for t in titles[:2]:
            if t.get("reserved"):
                continue
            tn = t.get("number")
            issue_date = t.get("latest_issue_date") or t.get("up_to_date_as_of")
            if not tn or not issue_date:
                continue
            try:
                path = client.fetch_title_xml(tn, issue_date)
                logger.info(f"Saved eCFR xml: {path}")
                saved += 1
            except Exception as exc:
                logger.warning(f"Failed fetching eCFR title {tn}: {exc}")
        logger.info(f"eCFR fetched titles saved: {saved}")
    except Exception as e:
        logger.warning(f"eCFR fetch failed: {e}")


def run_preprocessors():
    pre = Preprocessor()
    try:
        pre.clean_opinions()
        logger.info("clean_opinions completed")
    except Exception as e:
        logger.warning(f"clean_opinions failed: {e}")

    try:
        pre.clean_statutes()
        logger.info("clean_statutes completed")
    except Exception as e:
        logger.warning(f"clean_statutes failed: {e}")

    try:
        pre.clean_regulations(max_sections_per_title=10)
        logger.info("clean_regulations completed (capped)")
    except Exception as e:
        logger.warning(f"clean_regulations failed: {e}")


def light_index_from_parquet():
    persist = Config.CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=persist)
    names = [Config.CHROMA_CASES_COLLECTION, Config.CHROMA_STATUTES_COLLECTION, Config.CHROMA_REGULATIONS_COLLECTION]
    collections = {}
    for n in names:
        try:
            collections[n] = client.create_collection(n)
        except Exception:
            try:
                collections[n] = client.get_collection(n)
            except Exception:
                collections[n] = client.create_collection(n)

    processed = os.path.join(os.getcwd(), "data", "processed")
    # helper to upsert with dummy embeddings
    def upsert_from_parquet(fname, collection, id_fn, text_field, meta_fn):
        import pandas as pd
        path = os.path.join(processed, fname)
        if not os.path.exists(path):
            logger.info(f"No {fname} found, skipping")
            return 0
        df = pd.read_parquet(path)
        ids = []
        docs = []
        metas = []
        for _, row in df.iterrows():
            txt = row.get(text_field) if text_field in row else row.get('section_text') if 'section_text' in row else row.get('text')
            if not txt or not str(txt).strip():
                continue
            _id = id_fn(row)
            ids.append(_id)
            docs.append(txt)
            metas.append(meta_fn(row))
        if not ids:
            logger.info(f"No docs to index for {fname}")
            return 0
        # Try upsert without embeddings first (safer for smoke tests).
        try:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"Upserted {len(ids)} into {collection.name} (no embeddings)")
            return len(ids)
        except Exception as e:
            logger.warning(f"Upsert without embeddings failed for {collection.name}: {e}")
            # If error mentions expected embedding dimension, try with zero vectors of that size
            msg = str(e)
            import re
            m = re.search(r"dimension\D*(\d+)", msg)
            if m:
                dim = int(m.group(1))
                logger.info(f"Retrying upsert with zero embeddings of dim={dim}")
                embs = [[0.0] * dim for _ in ids]
                try:
                    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
                    logger.info(f"Upserted {len(ids)} into {collection.name} (zero embeddings)")
                    return len(ids)
                except Exception as e2:
                    logger.error(f"Upsert with zero embeddings also failed: {e2}")
                    return 0
            else:
                logger.error(f"Upsert failed and no embedding-dim info found: {e}")
                return 0

    # cases
    def case_id_fn(row):
        return f"case_{row.parent_opinion_id}_{row.chunk_index}"
    def case_meta_fn(row):
        return {"case_name": row.get('case_name'), "court": row.get('court')}

    # statutes
    def stat_id_fn(row):
        # include granule/package id when available to ensure uniqueness across granules
        gid = row.get('granule_id') or row.get('package_id') or ''
        if gid:
            return f"stat_{row.title_number}_{row.section_number}_{gid}_{row.chunk_index}"
        # fallback: use a short hash of the text to avoid collisions
        import hashlib
        txt = (row.get('section_text') or '')[:200]
        h = hashlib.sha1(txt.encode('utf-8')).hexdigest()[:8]
        return f"stat_{row.title_number}_{row.section_number}_{h}_{row.chunk_index}"
    def stat_meta_fn(row):
        return {"title_number": row.get('title_number'), "section_number": row.get('section_number')}

    # regs
    def reg_id_fn(row):
        return f"reg_{row.cfr_title}_{row.cfr_part}_{row.cfr_section}_{row.chunk_index}"
    def reg_meta_fn(row):
        return {"cfr_title": row.get('cfr_title'), "cfr_section": row.get('cfr_section')}

    counts = {}
    counts['cases'] = upsert_from_parquet('cases.parquet', collections[names[0]], case_id_fn, 'text', case_meta_fn)
    counts['statutes'] = upsert_from_parquet('statutes.parquet', collections[names[1]], stat_id_fn, 'section_text', stat_meta_fn)
    counts['regs'] = upsert_from_parquet('regulations.parquet', collections[names[2]], reg_id_fn, 'section_text', reg_meta_fn)

    logger.info(f"Indexing summary: {counts}")
    return counts


if __name__ == '__main__':
    logger.info("Starting LexIQ small smoke test (safe mode)")
    safe_fetch_courtlistener()
    safe_fetch_uscode()
    safe_fetch_ecfr()
    run_preprocessors()
    counts = light_index_from_parquet()
    logger.info("Smoke test finished. Collections counts: %s", counts)
