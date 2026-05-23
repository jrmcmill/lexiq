#!/usr/bin/env python3
"""Build or refresh the legal citation graph used by graph-aware retrieval."""
import argparse
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tqdm import tqdm

from src.config import Config
from src.rag.citation_graph import build_citation_graph


def main():
    parser = argparse.ArgumentParser(description='Build the LexIQ legal citation graph')
    parser.add_argument('--cases-parquet', type=str, default=None, help='Path to processed cases parquet')
    parser.add_argument('--statutes-parquet', type=str, default=None, help='Path to processed statutes parquet')
    parser.add_argument('--regs-parquet', type=str, default=None, help='Path to processed regulations parquet')
    parser.add_argument('--raw-cases-dir', type=str, default=None, help='Path to raw CourtListener JSON files')
    parser.add_argument('--persist-dir', type=str, default=None, help='Directory for the persisted citation graph')
    args = parser.parse_args()

    processed_dir = os.path.join(os.getcwd(), 'data', 'processed')
    raw_cases_dir = args.raw_cases_dir or os.path.join(os.getcwd(), 'data', 'raw', 'courtlistener')

    cases_parquet = args.cases_parquet or os.path.join(processed_dir, 'cases.parquet')
    statutes_parquet = args.statutes_parquet or os.path.join(processed_dir, 'statutes.parquet')
    regs_parquet = args.regs_parquet or os.path.join(processed_dir, 'regulations.parquet')
    persist_dir = args.persist_dir or Config.CHROMA_PERSIST_DIR

    start = datetime.utcnow()
    print(f"Starting citation graph build at {start.isoformat()}Z")

    steps = tqdm(total=1, desc='Citation graph setup', unit='step')
    steps.set_postfix_str('building graph')
    graph_stats = build_citation_graph(
        cases_parquet=cases_parquet,
        statutes_parquet=statutes_parquet,
        regs_parquet=regs_parquet,
        raw_cases_dir=raw_cases_dir,
        persist_dir=persist_dir,
    )
    steps.update(1)
    steps.close()

    end = datetime.utcnow()
    print(f"Built citation graph at {graph_stats.get('path')} with {graph_stats.get('nodes')} nodes and {graph_stats.get('out_edges')} edges")
    print(f"Completed citation graph build at {end.isoformat()}Z — duration: {end - start}")


if __name__ == '__main__':
    main()
