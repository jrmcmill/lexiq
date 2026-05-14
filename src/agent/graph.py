from src.agent.nodes import route_query, retrieve_node, generate_answer, format_citations_node
from src.agent.state import AgentState
from datetime import datetime
from src.observability.logger import get_logger

logger = get_logger(__name__)

class CompiledGraph:
    def __init__(self):
        pass


def build_graph():
    return CompiledGraph()


def run_query(query: str, session_id: str, history: list[dict], court_filter: str | None = None, date_after: str | None = None, date_before: str | None = None) -> dict:
    state: AgentState = {
        'messages': history or [],
        'query': query,
        'retrieved_cases': [],
        'retrieved_statutes': [],
        'retrieved_regs': [],
        'retrieved_session': [],
        'final_answer': '',
        'citations': [],
        'tool_calls': [],
        'session_id': session_id,
        'court_filter': court_filter,
        'date_after': date_after,
        'date_before': date_before,
        'error': None,
    }
    state = route_query(state)
    state = retrieve_node(state)
    state = generate_answer(state)
    state = format_citations_node(state)
    return state

if __name__ == '__main__':
    print('Graph module ready')
