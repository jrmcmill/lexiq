import pandas as pd
import os
import json
from glob import glob


def load_benchmark_results(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compute_aggregate_metrics(df: pd.DataFrame) -> dict:
    scores = df['score'].apply(lambda s: s if isinstance(s, dict) else {'faithfulness':0,'relevance':0,'completeness':0})
    faith = [s.get('faithfulness',0) for s in scores]
    rel = [s.get('relevance',0) for s in scores]
    comp = [s.get('completeness',0) for s in scores]
    mean_f = sum(faith)/len(faith) if faith else 0
    mean_r = sum(rel)/len(rel) if rel else 0
    mean_c = sum(comp)/len(comp) if comp else 0
    pass_rate = sum(1 for s in scores if s.get('faithfulness',0)>=3 and s.get('relevance',0)>=3 and s.get('completeness',0)>=3)/len(scores) if scores else 0
    return {'mean_faithfulness': mean_f, 'mean_relevance': mean_r, 'mean_completeness': mean_c, 'pass_rate': pass_rate}


def load_all_run_metrics() -> pd.DataFrame:
    files = glob(os.path.join(os.getcwd(), 'logs', 'benchmark_*.jsonl'))
    all_rows = []
    for fpath in files:
        df = load_benchmark_results(fpath)
        df['run_file'] = os.path.basename(fpath)
        all_rows.append(df)
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
