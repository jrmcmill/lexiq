#!/usr/bin/env python3
"""Process local textbook PDFs into the shared LexIQ index.

Usage:
  .venv/bin/python scripts/setup_textbooks.py
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.preprocessor import Preprocessor
from src.rag.indexer import Indexer


def main():
    pre = Preprocessor()
    textbooks = pre.clean_textbooks()

    idx = Indexer()
    idx.index_textbooks(textbooks)

    stats = idx.get_collection_stats()
    print(f"Processed {len(textbooks)} textbook chunks")
    print(f"Index stats: {stats}")


if __name__ == '__main__':
    main()
