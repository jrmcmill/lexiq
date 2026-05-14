import requests
import json
from src.agent.state import AgentState
from src.observability.logger import get_logger
from src.config import Config
from src.agent import tools
from src.agent.citation import extract_citations_from_text, format_bluebook_case, parse_citation, format_bluebook_statute, format_bluebook_regulation

logger = get_logger(__name__)

OLLAMA = Config.OLLAMA_BASE_URL
MODEL = Config.OLLAMA_MODEL


def _call_ollama(prompt: str) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    try:
        # Use longer timeout for local LLM inference (up to 5 minutes)
        resp = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get('response', '')
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out. The model may still be loading.")
        raise RuntimeError("Ollama request timed out (5 min). Make sure the model is loaded: ollama pull llama3.1:8b")
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama")
        raise RuntimeError("Could not connect to Ollama. Start it with: ollama serve")
    except requests.exceptions.RequestException as e:
        logger.error(str(e))
        raise RuntimeError("Ollama is not running. Start it with: ollama serve")


def route_query(state: AgentState) -> AgentState:
    prompt = (
        f"Given the legal query below, determine which knowledge sources are needed. "
        f"Return a JSON object with boolean fields: "
        f'{{\\"needs_cases\\": true/false, \\"needs_statutes\\": true/false, \\"needs_regulations\\": true/false, \\"needs_session_docs\\": true/false}} '
        f"Query: {state['query']}"
    )
    try:
        resp = _call_ollama(prompt)
        # try parse JSON
        j = json.loads(resp)
        calls = []
        if j.get('needs_cases'):
            calls.append('case_law')
        if j.get('needs_statutes'):
            calls.append('statute')
        if j.get('needs_regulations'):
            calls.append('regulation')
        if j.get('needs_session_docs'):
            calls.append('session')
        state['tool_calls'] = calls
    except Exception:
        state['tool_calls'] = ['case_law','statute','regulation','session']
    logger.info(f"Routing decision: {state['tool_calls']}")
    return state


def retrieve_node(state: AgentState) -> AgentState:
    calls = state.get('tool_calls', [])
    try:
        if 'case_law' in calls:
            state['retrieved_cases'] = tools.case_law_search(state['query'], state.get('court_filter'), state.get('date_after'), state.get('date_before'))
    except Exception as e:
        logger.error(str(e))
    try:
        if 'statute' in calls:
            state['retrieved_statutes'] = tools.statute_search(state['query'])
    except Exception as e:
        logger.error(str(e))
    try:
        if 'regulation' in calls:
            state['retrieved_regs'] = tools.regulation_search(state['query'])
    except Exception as e:
        logger.error(str(e))
    try:
        if 'session' in calls and state.get('session_id'):
            state['retrieved_session'] = tools.session_document_search(state['query'], state['session_id'])
    except Exception as e:
        logger.error(str(e))
    return state


def generate_answer(state: AgentState) -> AgentState:
    parts = []
    def add_results(label, results):
        if not results:
            return
        if not isinstance(results, list):
            logger.warning(f"Expected list for {label}, got {type(results)}")
            return
        for r in results:
            if not isinstance(r, dict):
                logger.warning(f"Expected dict in {label} results, got {type(r)}")
                continue
            try:
                meta = r.get('metadata', {})
                cite = meta.get('bluebook_cite') or meta.get('usc_citation') or meta.get('cfr_citation')
                text = r.get('text', '')
                parts.append(f"SOURCE: {label}\nCITATION: {cite}\nTEXT: {text}\n")
            except Exception as e:
                logger.error(f"Error processing result in {label}: {str(e)}")
    
    add_results('CASE', state.get('retrieved_cases', []))
    add_results('STATUTE', state.get('retrieved_statutes', []))
    add_results('REGULATION', state.get('retrieved_regs', []))
    add_results('SESSION', state.get('retrieved_session', []))
    formatted_context = "\n\n".join(parts)
    prompt = f"You are LexIQ, an expert legal research assistant for lawyers and paralegals. Answer the following legal question using ONLY the provided sources. If the sources do not contain enough information, say so clearly. Always cite your sources using Bluebook format at the end of your answer. Do not give legal advice; present findings objectively.\n\nSOURCES:\n{formatted_context}\n\nQUESTION: {state['query']}\n\nProvide a thorough, well-organized answer followed by a 'Citations' section."
    try:
        resp = _call_ollama(prompt)
    except Exception as e:
        state['error'] = str(e)
        return state
    state['final_answer'] = resp
    raw_cites = extract_citations_from_text(resp)
    state['citations'] = raw_cites
    return state


def format_citations_node(state: AgentState) -> AgentState:
    # dedupe
    unique = []
    for c in state.get('citations', []):
        if c not in unique:
            unique.append(c)
    formatted = []
    for raw in unique:
        parsed = parse_citation(raw)
        if not parsed:
            formatted.append(raw)
            continue
        if parsed['type'] == 'case':
            formatted.append(format_bluebook_case('', parsed.get('reporter'), parsed.get('volume'), parsed.get('page'), '', ''))
        elif parsed['type'] == 'statute':
            formatted.append(format_bluebook_statute(parsed.get('title'), parsed.get('section')))
        elif parsed['type'] == 'regulation':
            formatted.append(format_bluebook_regulation(parsed.get('title'), '', parsed.get('section')))
    state['citations'] = formatted
    if state.get('final_answer') and 'Citations' not in state['final_answer']:
        state['final_answer'] = state.get('final_answer','') + "\n\nCitations:\n" + "\n".join(formatted)
    return state
