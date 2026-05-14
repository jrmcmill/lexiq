import streamlit as st
import os
import sys
import time

# ensure repo root on path for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

from src.data.courtlistener import CourtListenerClient
from src.data.uscode import USCodeClient
from src.data.ecfr import ECFRClient
from src.data.preprocessor import Preprocessor
from src.rag.indexer import Indexer
from src.config import Config

st.set_page_config(page_title="Data Refresh")
st.title("Data Refresh")
st.write("Pull and re-index data from external sources.")


def _rebuild_pipeline(progress_callback=None):
	if progress_callback:
		progress_callback(0.70, "Preprocessing: rebuilding cases/statutes/regulations parquet files")
	pre = Preprocessor()
	cases_df = pre.clean_opinions()
	statutes_df = pre.clean_statutes()
	regs_df = pre.clean_regulations()

	if progress_callback:
		progress_callback(0.82, "Indexing: updating vector collections for all sources")
	idx = Indexer()
	idx.index_cases(cases_df)
	idx.index_statutes(statutes_df)
	idx.index_regulations(regs_df)

	stats = idx.get_collection_stats()
	processed = {
		"cases_chunks": len(cases_df),
		"statutes_chunks": len(statutes_df),
		"regulations_chunks": len(regs_df),
	}
	return {"processed": processed, "index_stats": stats}


def _update_all(progress_callback=None, run_reindex=False, ecfr_titles_to_check=10):
	results = {"courtlistener": 0, "uscode": 0, "ecfr": 0}
	# CourtListener
	client = CourtListenerClient()
	raw_dir = client.raw_dir
	existing = 0
	if os.path.exists(raw_dir):
		existing = len([f for f in os.listdir(raw_dir) if f.endswith('.json')])
	if progress_callback:
		progress_callback(0.1, f"CourtListener: {existing} existing raw files")
	new = client.fetch_opinions(query="", max_pages=Config.COURTLISTENER_MAX_PAGES, skip_existing=True, stop_after_seen=20)
	results["courtlistener"] = len(new)
	if progress_callback:
		progress_callback(0.35, f"CourtListener: saved {len(new)} new items")

	# US Code
	usc_client = USCodeClient()
	if not usc_client.api_key:
		if progress_callback:
			progress_callback(0.45, "USC: API key not set; skipped")
	else:
		saved = usc_client.fetch_sections(max_pages=2, sort_by="DATE")
		new_files = sum(1 for r in saved if r.get('raw_path') and os.path.exists(r.get('raw_path')))
		results["uscode"] = new_files
		if progress_callback:
			progress_callback(0.6, f"USC: processed {len(saved)} granules, new files: {new_files}")

	# eCFR
	ecfr = ECFRClient()
	titles = ecfr.list_titles()
	def _title_date_key(title):
		raw = title.get('latest_issue_date') or title.get('up_to_date_as_of') or ""
		return raw
	to_check = sorted(titles, key=_title_date_key, reverse=True)[:max(1, int(ecfr_titles_to_check))]
	saved_ecfr = 0
	for t in to_check:
		if t.get('reserved'):
			continue
		tn = t.get('number')
		issue_date = t.get('latest_issue_date') or t.get('up_to_date_as_of')
		if not tn or not issue_date:
			continue
		path = os.path.join(ecfr.raw_dir, f"title_{tn}_{issue_date}.xml")
		if os.path.exists(path):
			continue
		try:
			ecfr.fetch_title_xml(tn, issue_date)
			saved_ecfr += 1
		except Exception as exc:
			# show but continue
			if progress_callback:
				progress_callback(0.8, f"eCFR: failed to fetch title {tn}: {exc}")
	results["ecfr"] = saved_ecfr
	if progress_callback:
		progress_callback(0.68, f"eCFR: saved {saved_ecfr} new titles")

	if run_reindex:
		pipeline = _rebuild_pipeline(progress_callback=progress_callback)
		results.update(pipeline)

	if progress_callback:
		progress_callback(0.98, "Refresh pipeline complete")
	return results


st.write("Use the controls below to run an incremental data update. You can optionally preprocess and re-index so fresh data is immediately usable in all search and chat features.")

run_reindex = st.checkbox("After fetch, rebuild processed data and vector index", value=True)
ecfr_titles_to_check = st.slider("eCFR titles to check (newest first)", min_value=2, max_value=50, value=10, step=2)

if st.button("Update Database"):
	status = st.empty()
	prog = st.progress(0)

	def cb(pct, msg=None):
		try:
			prog.progress(min(100, int(pct * 100)))
		except Exception:
			pass
		if msg:
			status.text(msg)

	mode = "with reindex" if run_reindex else "fetch-only"
	status.text(f"Starting incremental update ({mode})...")
	try:
		results = _update_all(progress_callback=cb, run_reindex=run_reindex, ecfr_titles_to_check=ecfr_titles_to_check)
		prog.progress(100)
		status.text(f"Update complete: {results}")
		st.success("Update finished")
		if run_reindex:
			try:
				st.cache_resource.clear()
			except Exception:
				pass
		st.json(results)
	except Exception as e:
		status.text(f"Update failed: {e}")
		st.error(str(e))
