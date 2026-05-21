#!/usr/bin/env python3
"""
Debug utility to understand embedding distance distribution.
Run this to see what distances are actually returned from your ChromaDB.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.rag.retriever import Retriever
from src.config import Config

def test_distances():
    """Test queries and show actual distance distributions."""
    retriever = Retriever()
    
    test_queries = [
        "fraud in contract law",
        "negligence liability",
        "employment discrimination",
        "intellectual property rights",
        "criminal sentencing guidelines",
    ]
    
    print("=" * 80)
    print("DISTANCE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"\nCurrent RETRIEVAL_MIN_DISTANCE: {Config.RETRIEVAL_MIN_DISTANCE}")
    print(f"Current RERANK_MIN_SCORE: {Config.RERANK_MIN_SCORE}")
    print("\nTesting queries to understand actual distance ranges:\n")
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")
        
        # Get unfiltered results directly from ChromaDB
        try:
            emb = retriever.embedder.embed_query(query)
            if emb:
                # Cases
                res = retriever.indexer.cases.query(query_embeddings=[emb], n_results=20)
                docs = res.get('documents', [[]])[0] if res.get('documents') else []
                distances = res.get('distances', [[]])[0] if res.get('distances') else []
                
                if distances:
                    distances = list(distances)
                    print(f"\nCASE RESULTS (showing all 20):")
                    print(f"  Min distance: {min(distances):.4f}")
                    print(f"  Max distance: {max(distances):.4f}")
                    print(f"  Mean distance: {sum(distances)/len(distances):.4f}")
                    print(f"  Median distance: {sorted(distances)[len(distances)//2]:.4f}")
                    
                    # Show breakdown by threshold
                    for threshold in [0.3, 0.5, 0.7, 0.8, 0.9]:
                        count = sum(1 for d in distances if d <= threshold)
                        pct = 100 * count / len(distances)
                        print(f"  Results with distance ≤ {threshold}: {count}/20 ({pct:.0f}%)")
                    
                    # Show first 5 results
                    print(f"\n  First 5 results:")
                    for i, (doc, dist) in enumerate(zip(docs[:5], distances[:5]), 1):
                        snippet = doc[:60].replace('\n', ' ') + "..."
                        print(f"    {i}. distance={dist:.4f} | {snippet}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS:")
    print(f"{'=' * 80}")
    print("""
If most results have distance > 0.7:
  → Set RETRIEVAL_MIN_DISTANCE=0.8 or higher (current: 0.8) ✓
  → Results should now come through

If some results filtered but LLM still hallucinates:
  → The LLM parameters control creativity:
     OLLAMA_TEMPERATURE=0.1  (keep deterministic)
     OLLAMA_TOP_P=0.15       (restrict token space)
  → These prevent hallucination better than distance filtering

If you want stricter filtering:
  → Lower RETRIEVAL_MIN_DISTANCE to 0.6 or 0.5
  → But expect fewer results to come through
  → Pair with lower RERANK_MIN_SCORE (0.05-0.1)

Current settings (permissive, source-grounded):
  RETRIEVAL_MIN_DISTANCE = 0.8
  RERANK_MIN_SCORE = 0.1
  OLLAMA_TEMPERATURE = 0.1  ← Main hallucination prevention
  MIN_RETRIEVED_RESULTS = 1
""")

if __name__ == '__main__':
    test_distances()
