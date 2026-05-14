#!/usr/bin/env python3
"""Full setup: fetch all available documents, preprocess, and index.

This script intentionally removes artificial caps used in quick-setup flows
and attempts to fetch as much as the upstream APIs allow. Use with care
— this may be network- and time-intensive.

Usage:
    source .venv/bin/activate
    python scripts/setup_full.py
"""
import os
import sys
import logging
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.courtlistener import CourtListenerClient
from src.data.uscode import USCodeClient
from src.data.ecfr import ECFRClient
from src.data.preprocessor import Preprocessor
from src.rag.indexer import Indexer
from src.config import Config
from src.observability.logger import get_logger

logger = get_logger(__name__)


def fetch_all_courtlistener():
    client = CourtListenerClient()
    # Use a very large max_pages to attempt exhaustiveness; CourtListener will stop if no next page
    max_pages = max(1000, getattr(Config, 'COURTLISTENER_MAX_PAGES', 10))
    print(f"Fetching CourtListener opinions (up to {max_pages} pages)...")
    saved = client.fetch_opinions(query="", max_pages=max_pages, skip_existing=False, stop_after_seen=None)
    print(f"Saved {len(saved)} opinions")
    return saved


def fetch_all_uscode():
    client = USCodeClient()
    if not client.api_key:
        print("GOVINFO_API_KEY not set — skipping U.S. Code fetch")
        return []
    # Request date-sorted results first, increase max pages
    print("Fetching U.S. Code granules (many pages)...")
    saved = client.fetch_sections(max_pages=1000, sort_by="DATE")
    print(f"Processed {len(saved)} USC granules")
    return saved


def fetch_all_ecfr():
    client = ECFRClient()
    titles = client.list_titles()
    # fetch all non-reserved titles, newest first
    titles_sorted = sorted(
        titles,
        key=lambda t: t.get('latest_issue_date') or t.get('up_to_date_as_of') or "",
        reverse=True,
    )
    saved = 0
    for t in titles_sorted:
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

    stats = idx.get_collection_stats()
    print(f"Index stats: {stats}")
    return stats


def main():
    start = datetime.utcnow()
    print(f"Starting full setup at {start.isoformat()}Z")

    # 1) fetch
    fetch_all_courtlistener()
    fetch_all_uscode()
    fetch_all_ecfr()

    # 2) preprocess + index
    stats = preprocess_and_index()

    end = datetime.utcnow()
    print(f"Completed full setup at {end.isoformat()}Z — duration: {end - start}")
    print(f"Final index stats: {stats}")


if __name__ == '__main__':
    main()
