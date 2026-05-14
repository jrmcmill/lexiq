from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    query: str
    retrieved_cases: List[Dict[str, Any]]
    retrieved_statutes: List[Dict[str, Any]]
    retrieved_regs: List[Dict[str, Any]]
    retrieved_session: List[Dict[str, Any]]
    final_answer: str
    citations: List[str]
    tool_calls: List[str]
    session_id: str
    court_filter: str | None
    date_after: str | None
    date_before: str | None
    error: str | None
