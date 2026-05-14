"""
Incremental update script for LexIQ data sources.
- Only saves new raw items and avoids duplicate raw files
- Uses a consecutive-seen heuristic to stop early when no new items are found

Run from repo root inside venv:

source .venv/bin/activate
python scripts/update_database.py

"""
import os
import sys
import logging

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.courtlistener import CourtListenerClient
from src.data.uscode import USCodeClient
from src.data.ecfr import ECFRClient
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('update_db')


def update_courtlistener(max_pages=Config.COURTLISTENER_MAX_PAGES, stop_after_seen=20):
    client = CourtListenerClient()
    existing = set()
    raw_dir = client.raw_dir
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith('.json'):
                existing.add(os.path.splitext(f)[0])
    logger.info(f"Existing courtlistener raw items: {len(existing)}")
    # fetch recent opinions, skip existing, stop after encountering consecutive existing items
    new = client.fetch_opinions(query="", max_pages=max_pages, skip_existing=True, stop_after_seen=stop_after_seen)
    logger.info(f"Saved {len(new)} new CourtListener items")
    return len(new)


def update_uscode(max_pages=2):
    client = USCodeClient()
    if not client.api_key:
        logger.info("GOVINFO_API_KEY not set; skipping USC update")
        return 0
    # fetch sections; USCodeClient._save_granule skips existing files
    saved = client.fetch_sections(max_pages=max_pages, sort_by="DATE")
    # fetch_sections returns list of saved records (including skipped ones)
    new_count = sum(1 for r in saved if r.get('raw_path') and os.path.exists(r.get('raw_path')))
    logger.info(f"USC fetch processed {len(saved)} granules (new or existing); new files present: {new_count}")
    return new_count


def update_ecfr(titles_to_check=2):
    client = ECFRClient()
    titles = client.list_titles()
    titles_sorted = sorted(
        titles,
        key=lambda t: t.get('latest_issue_date') or t.get('up_to_date_as_of') or "",
        reverse=True,
    )
    saved = 0
    for t in titles_sorted[:titles_to_check]:
        if t.get('reserved'):
            continue
        tn = t.get('number')
        issue_date = t.get('latest_issue_date') or t.get('up_to_date_as_of')
        if not tn or not issue_date:
            continue
        path = os.path.join(client.raw_dir, f"title_{tn}_{issue_date}.xml")
        if os.path.exists(path):
            logger.debug(f"Skipping existing eCFR title file {path}")
            continue
        try:
            client.fetch_title_xml(tn, issue_date)
            saved += 1
        except Exception as exc:
            logger.warning(f"Could not fetch title {tn}: {exc}")
    logger.info(f"Saved {saved} new eCFR title XML files")
    return saved


if __name__ == '__main__':
    logger.info("Starting incremental update")
    c = update_courtlistener()
    u = update_uscode()
    e = update_ecfr()
    logger.info(f"Update summary: courtlistener={c}, uscode={u}, ecfr={e}")
