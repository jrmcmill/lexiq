from datetime import datetime
import json
import os
from src.evaluation.judge import LLMJudge
from src.agent.graph import run_query
from src.observability.logger import get_logger

logger = get_logger(__name__)

BENCHMARK_CASES = [
    {"id": "case_1", "question": "What did Brown v. Board hold?", "expected_topics": ["Brown v. Board"], "source_type": "case"},
    {"id": "case_2", "question": "What is the holding in Miranda?", "expected_topics": ["Miranda"], "source_type": "case"},
    {"id": "case_3", "question": "Standard for summary judgment?", "expected_topics": ["summary judgment"], "source_type": "case"},
    {"id": "case_4", "question": "What is Chevron deference?", "expected_topics": ["Chevron"], "source_type": "case"},
    {"id": "case_5", "question": "What constitutes probable cause?", "expected_topics": ["probable cause"], "source_type": "case"},
    {"id": "stat_1", "question": "What does 42 U.S.C. § 1983 provide?", "expected_topics": ["42 U.S.C. § 1983"], "source_type": "statute"},
    {"id": "stat_2", "question": "Title VII prohibits what?", "expected_topics": ["Title VII"], "source_type": "statute"},
    {"id": "stat_3", "question": "What is the ADA's definition of disability?", "expected_topics": ["ADA"], "source_type": "statute"},
    {"id": "stat_4", "question": "What does 26 U.S.C. say about deductions?", "expected_topics": ["26 U.S.C."], "source_type": "statute"},
    {"id": "stat_5", "question": "What remedies does the Fair Housing Act provide?", "expected_topics": ["Fair Housing Act"], "source_type": "statute"},
    {"id": "reg_1", "question": "What does 12 C.F.R. § 226.1 cover?", "expected_topics": ["12 C.F.R. § 226.1"], "source_type": "regulation"},
    {"id": "reg_2", "question": "What is the FMLA eligibility?", "expected_topics": ["FMLA"], "source_type": "regulation"},
    {"id": "reg_3", "question": "Truth in Lending regulation basics?", "expected_topics": ["Truth in Lending"], "source_type": "regulation"},
    {"id": "mix_1", "question": "Does case X interpret 42 U.S.C. § 1983 in this way?", "expected_topics": ["1983","case law"], "source_type": "mixed"},
    {"id": "mix_2", "question": "How does regulation Y interact with statute Z?", "expected_topics": ["regulation","statute"], "source_type": "mixed"},
]

def run_benchmarks(graph=None):
    judge = LLMJudge()
    results = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    for case in BENCHMARK_CASES:
        try:
            state = run_query(case['question'], session_id='bench', history=[], court_filter=None)
            answer = state.get('final_answer','')
            context = '\n'.join([r.get('text','') for r in state.get('retrieved_cases', []) + state.get('retrieved_statutes', []) + state.get('retrieved_regs', [])])
            score = judge.score(case['question'], context, answer)
            out = {**case, 'answer': answer, 'score': score, 'timestamp': timestamp}
            results.append(out)
        except Exception as e:
            logger.error(str(e))
    logpath = os.path.join(os.getcwd(), 'logs', f"benchmark_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.jsonl")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results

if __name__ == '__main__':
    run_benchmarks()
