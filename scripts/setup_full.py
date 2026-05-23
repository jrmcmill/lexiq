#!/usr/bin/env python3
"""CLI for fetching, preprocessing, and indexing the corpus.

Usage examples:
  python scripts/setup_full.py --cases 1000 --granules 500 --ecfr-titles 10 --reindex

This module exposes functions that other modules (like the Streamlit data-refresh
page) can import and call programmatically.
"""
import os
import sys
import argparse
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.courtlistener import CourtListenerClient
from src.data.uscode import USCodeClient
from src.data.ecfr import ECFRClient
from src.data.preprocessor import Preprocessor
from src.rag.indexer import Indexer
from src.rag.citation_graph import build_citation_graph
from src.config import Config
from src.observability.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)


def fetch_courtlistener(target_cases: int = 1000, max_pages: int = 1000, skip_existing: bool = False) -> list:
    client = CourtListenerClient()
    max_pages = max_pages or getattr(Config, 'COURTLISTENER_MAX_PAGES', 10)
    print(f"Fetching CourtListener opinions (up to {max_pages} pages, stopping after {target_cases} saved opinions)...")
    saved = client.fetch_opinions(query="", max_pages=max_pages, skip_existing=skip_existing, stop_after_seen=None, stop_after_count=target_cases)
    print(f"Saved {len(saved)} opinions")
    return saved


def fetch_uscode(target_granules: int = 500, max_pages: int = 200) -> list:
    client = USCodeClient()
    if not client.api_key:
        print("GOVINFO_API_KEY not set — skipping U.S. Code fetch")
        return []
    print(f"Fetching U.S. Code granules (up to {max_pages} pages, stopping after {target_granules} granules)...")
    saved = client.fetch_sections(max_pages=max_pages, sort_by="DATE", stop_after_count=target_granules)
    print(f"Processed {len(saved)} USC granules")
    return saved


def fetch_ecfr(titles_to_check: int = 10) -> int:
    client = ECFRClient()
    titles = client.list_titles()
    # sort newest first
    titles_sorted = sorted(
        titles,
        key=lambda t: t.get('latest_issue_date') or t.get('up_to_date_as_of') or "",
        reverse=True,
    )
    saved = 0
    for t in tqdm(titles_sorted[:max(1, int(titles_to_check))], desc="eCFR Titles", unit="title"):
        if t.get('reserved'):
            continue
        tn = t.get('number')
        issue_date = t.get('latest_issue_date') or t.get('up_to_date_as_of')
        if not tn or not issue_date:
            continue
        path = os.path.join(client.raw_dir, f"title_{tn}_{issue_date}.xml")
        if os.path.exists(path):
            continue
        try:
            client.fetch_title_xml(tn, issue_date)
            saved += 1
        except Exception as exc:
            logger.warning(f"Could not fetch eCFR title {tn}: {exc}")
    print(f"Saved {saved} new eCFR titles")
    return saved


def preprocess_and_index():
    print("Preprocessing raw files into parquet... (this may take time)")
    pre = Preprocessor()
    cases = pre.clean_opinions()
    statutes = pre.clean_statutes()
    regs = pre.clean_regulations()

    print("Indexing into ChromaDB...")
    idx = Indexer()
    idx.index_cases(cases)
    idx.index_statutes(statutes)
    idx.index_regulations(regs)

    print("Building citation graph...")
    graph_stats = build_citation_graph(
        cases_parquet=os.path.join(os.getcwd(), 'data', 'processed', 'cases.parquet'),
        statutes_parquet=os.path.join(os.getcwd(), 'data', 'processed', 'statutes.parquet'),
        regs_parquet=os.path.join(os.getcwd(), 'data', 'processed', 'regulations.parquet'),
        raw_cases_dir=os.path.join(os.getcwd(), 'data', 'raw', 'courtlistener'),
        persist_dir=Config.CHROMA_PERSIST_DIR,
    )
    print(f"Citation graph: {graph_stats.get('nodes')} nodes, {graph_stats.get('out_edges')} edges")

    stats = idx.get_collection_stats()
    print(f"Index stats: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Fetch, preprocess and index legal corpora')
    parser.add_argument('--cases', type=int, default=1000, help='Number of court opinions to fetch')
    parser.add_argument('--case-pages', type=int, default=1000, help='Max pages for CourtListener fetch')
    parser.add_argument('--granules', type=int, default=500, help='Number of US Code granules to fetch')
    parser.add_argument('--granule-pages', type=int, default=200, help='Max pages for US Code fetch')
    parser.add_argument('--ecfr-titles', type=int, default=10, help='Number of eCFR titles to fetch (newest first)')
    parser.add_argument('--no-reindex', dest='reindex', action='store_false', help='Skip preprocessing and reindex step')
    parser.add_argument('--embed-model', type=str, default=None, help='Override embed model with env var EMBED_MODEL')
    args = parser.parse_args()

    if args.embed_model:
        os.environ['EMBED_MODEL'] = args.embed_model

    start = datetime.utcnow()
    print(f"Starting setup at {start.isoformat()}Z")

    total_steps = 4 if args.reindex else 3
    steps = tqdm(total=total_steps, desc='LexIQ setup', unit='step')

    # fetch
    fetch_courtlistener(target_cases=args.cases, max_pages=args.case_pages)
    steps.update(1)
    fetch_uscode(target_granules=args.granules, max_pages=args.granule_pages)
    steps.update(1)
    fetch_ecfr(titles_to_check=args.ecfr_titles)
    steps.update(1)

    # preprocess + index
    if args.reindex:
        preprocess_and_index()
        steps.update(1)
    steps.close()

    end = datetime.utcnow()
    print(f"Completed setup at {end.isoformat()}Z — duration: {end - start}")


if __name__ == '__main__':
    main()
