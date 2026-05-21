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


def _normalize_source_type(label: str) -> str:
    mapping = {
        'CASE': 'Case Law',
        'STATUTE': 'U.S. Code',
        'REGULATION': 'Regulation',
        'SESSION': 'Session Doc',
    }
    return mapping.get(label, label.title() if isinstance(label, str) else 'Source')


def _call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": Config.OLLAMA_TEMPERATURE,
        "top_p": Config.OLLAMA_TOP_P,
        "top_k": Config.OLLAMA_TOP_K,
        "num_predict": 2048,
    }
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
    case_parts = []
    other_parts = []
    total_results = 0
    seen_context = set()
    used_sources = []

    stop_words = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'about', 'how',
        'could', 'would', 'should', 'what', 'when', 'where', 'which', 'who', 'why',
        'have', 'has', 'had', 'been', 'being', 'are', 'was', 'were', 'not', 'but',
        'can', 'may', 'might', 'your', 'their', 'them', 'they', 'you', 'our', 'its',
        'his', 'her', 'also', 'than', 'then', 'there', 'here', 'over', 'under', 'any',
        'recent', 'case', 'cases', 'court', 'supreme', 'involving', 'interaction',
    }

    query_terms = {
        t for t in ''.join(ch if ch.isalnum() else ' ' for ch in (state.get('query') or '').lower()).split()
        if len(t) > 2 and t not in stop_words
    }

    def relevance_of(r: dict) -> float:
        # prefer reranker score, otherwise use inverted distance as proxy
        try:
            score = r.get('score')
            if score is not None:
                return float(score)
        except Exception:
            pass
        try:
            dist = r.get('distance')
            if dist is not None:
                return max(0.0, 1.0 - float(dist))
        except Exception:
            pass
        return 0.0

    def collect_and_limit(label, results):
        """Return a trimmed, sorted list of results to include in prompt."""
        if not results or not isinstance(results, list):
            return []

        def is_topically_relevant(r: dict) -> bool:
            if label == 'CASE':
                return True
            if not query_terms:
                return True
            meta = r.get('metadata', {}) if isinstance(r, dict) else {}
            text = (r.get('text') or '') if isinstance(r, dict) else ''
            combined = ' '.join([
                text,
                str(meta.get('usc_citation', '')),
                str(meta.get('cfr_citation', '')),
                str(meta.get('title', '')),
                str(meta.get('heading', '')),
                str(meta.get('section_title', '')),
            ]).lower()
            hits = sum(1 for term in query_terms if term in combined)
            # keep if at least one query term overlaps, or if high relevance score exists
            if hits >= 1:
                return True
            try:
                rel = relevance_of(r)
                return rel >= 0.65
            except Exception:
                return False

        # compute relevance for each result
        annotated = []
        for r in results:
            if not isinstance(r, dict):
                continue
            if not is_topically_relevant(r):
                continue
            rel = relevance_of(r)
            annotated.append((rel, r))
        # sort by relevance desc
        annotated.sort(key=lambda x: x[0], reverse=True)
        # limit by per-tool budget (favor cases, constrain non-case context)
        default_limit = getattr(Config, 'MAX_DOCS_PER_TOOL', 8)
        non_case_limit = getattr(Config, 'MAX_NON_CASE_DOCS_PER_TOOL', 3)
        limit = default_limit if label == 'CASE' else min(default_limit, non_case_limit)

        # Upweight canonical case entities: prefer one representative chunk per parent_opinion_id
        if label == 'CASE':
            # group by parent_opinion_id (fall back to bluebook_cite or case_name)
            groups: dict = {}
            others: list = []
            for rel, r in annotated:
                meta = r.get('metadata', {}) if isinstance(r, dict) else {}
                parent = meta.get('parent_opinion_id') or meta.get('bluebook_cite') or meta.get('case_name')
                if parent:
                    groups.setdefault(parent, []).append((rel, r))
                else:
                    others.append((rel, r))

            # pick top item from each group
            picked: list = []
            for parent, items in groups.items():
                items.sort(key=lambda x: x[0], reverse=True)
                picked.append(items[0])

            # sort picked by relevance and take up to limit
            picked.sort(key=lambda x: x[0], reverse=True)
            kept = [r for _, r in picked[:limit]]

            # if we haven't filled the budget, fill with next-best remaining items across all groups and others
            if len(kept) < limit:
                remaining = []
                for parent, items in groups.items():
                    for it in items[1:]:
                        remaining.append(it)
                remaining.extend(others)
                remaining.sort(key=lambda x: x[0], reverse=True)
                need = limit - len(kept)
                kept.extend([r for _, r in remaining[:need]])

            if len(annotated) > len(kept):
                logger.info(f"Trimming {len(annotated) - len(kept)} {label} results to top {limit} by grouped parent_opinion_id")
            return kept

        kept = [r for _, r in annotated[:limit]]
        # log excluded count
        if len(annotated) > len(kept):
            logger.info(f"Trimming {len(annotated) - len(kept)} {label} results to top {limit} by relevance")
        return kept

    cases_for_prompt = collect_and_limit('CASE', state.get('retrieved_cases', []))
    stats_for_prompt = collect_and_limit('STATUTE', state.get('retrieved_statutes', []))
    regs_for_prompt = collect_and_limit('REGULATION', state.get('retrieved_regs', []))
    sess_for_prompt = collect_and_limit('SESSION', state.get('retrieved_session', []))

    def add_results(label, results):
        nonlocal total_results
        source_type = _normalize_source_type(label)
        for r in results:
            try:
                meta = r.get('metadata', {})
                cite = meta.get('bluebook_cite') or meta.get('usc_citation') or meta.get('cfr_citation') or meta.get('case_name')
                text = r.get('text', '')
                score = r.get('score')
                dedupe_key = (source_type, cite or text[:200])
                if dedupe_key in seen_context:
                    continue
                seen_context.add(dedupe_key)
                score_str = f" [relevance: {score:.2f}]" if score is not None else ""
                court = meta.get('court') or meta.get('jurisdiction') or ''
                date = meta.get('decision_date') or meta.get('date') or ''
                court_line = f"COURT: {court}\n" if court else ""
                date_line = f"DATE: {date}\n" if date else ""
                block = f"SOURCE: {source_type}\nCITATION: {cite}\n{court_line}{date_line}TEXT: {text}{score_str}\n"
                if source_type == 'Case Law':
                    case_parts.append(block)
                else:
                    other_parts.append(block)
                used_sources.append({'type': source_type, 'citation': cite, 'score': score, 'distance': r.get('distance')})
                total_results += 1
            except Exception as e:
                logger.error(f"Error processing result in {label}: {str(e)}")

    add_results('CASE', cases_for_prompt)
    add_results('STATUTE', stats_for_prompt)
    add_results('REGULATION', regs_for_prompt)
    add_results('SESSION', sess_for_prompt)
    
    # Check if we have sufficient sources
    min_sources = Config.MIN_RETRIEVED_RESULTS
    if total_results < min_sources:
        warning = f"⚠️ INSUFFICIENT SOURCES: Only {total_results} source(s) found (minimum required: {min_sources}). Answer may be incomplete."
        if not case_parts and not other_parts:
            state['final_answer'] = f"I cannot provide a reliable answer because no relevant sources were found in the knowledge base. Please try rephrasing your question or consulting other legal resources.\n\n{warning}"
            state['citations'] = []
            state['used_sources'] = used_sources
            logger.warning(f"No sources retrieved for query: {state['query']}")
            return state
        logger.warning(f"Only {total_results} source(s) found for query: {state['query']}")
    
    case_context = "\n\n".join(case_parts)
    other_context = "\n\n".join(other_parts)

    # Create an explicit lead-case summary to make cases highly salient to the LLM
    lead_case_block = ""
    try:
        if case_parts:
            # case_parts are blocks starting with SOURCE/CITATION/TEXT
            # use the first case's citation and a short excerpt as a summary
            first_case = case_parts[0]
            # extract CITATION line and TEXT snippet
            lines = first_case.splitlines()
            citation_line = next((l for l in lines if l.startswith('CITATION:')), '')
            text_line = next((l for l in lines if l.startswith('TEXT:')), '')
            cite = citation_line.replace('CITATION:', '').strip() if citation_line else ''
            text = text_line.replace('TEXT:', '').strip() if text_line else ''
            excerpt = (text[:400] + '...') if len(text) > 400 else text
            lead_case_block = f"LEAD CASE: {cite}\nSUMMARY: {excerpt}\n"
    except Exception:
        lead_case_block = ""

    formatted_context = "\n\n".join([
        lead_case_block,
        f"CASE SOURCES (prioritize these if present):\n{case_context}" if case_context else "",
        f"OTHER SOURCES:\n{other_context}" if other_context else "",
    ]).strip()
    
    # Improved system prompt with stronger grounding
    prompt = f"""You are LexIQ, an expert legal research assistant. Your responses must be grounded in the provided sources.

CRITICAL RULES:
1. ONLY answer using the provided sources below
2. If sources don't adequately answer the question, clearly state: "Based on the available sources, I cannot provide a complete answer to this question."
3. Do NOT make inferences beyond what the sources explicitly state
4. Do NOT use general legal knowledge not found in sources
5. ALWAYS cite every claim with proper Bluebook format
6. If a source has a relevance score shown, consider lower scores (<0.3) as weak evidence
7. Explicitly state any limitations in the available sources
8. If any CASE SOURCES are present, lead your answer with them and explain how they answer the question before mentioning other authorities
8b. If CASE SOURCES are present, begin your substantive answer with a `CASE ANALYSIS:` section that cites the lead case(s) and includes a 1-2 sentence summary drawn explicitly from the TEXT provided above
9. If the question is about a recent court case, do not substitute a different older case when a newer case source is present
10. If a source directly discusses the question's key term, use that source rather than substituting a generic legal explanation

SOURCES (with relevance scores where available):
{formatted_context}

QUESTION: {state['query']}

Instructions:
- Provide a thorough, well-organized answer using ONLY the sources above
- Start with the most directly relevant case source if present and explain how it answers the question
- If sources are insufficient, clearly state the limitation
- Include a "Citations" section at the end with Bluebook-formatted citations
- Be precise and objective; do not provide legal advice
"""
    
    try:
        resp = _call_ollama(prompt)
    except Exception as e:
        state['error'] = str(e)
        logger.error(f"Error generating answer: {str(e)}")
        state['used_sources'] = used_sources
        return state

    # If CASE SOURCES exist but the draft omits them, denies their presence,
    # or introduces out-of-source citations, run constrained conversational
    # rewrite attempts (non-deterministic).
    try:
        case_cites = [s.get('citation') for s in used_sources if s.get('type') == 'Case Law' and s.get('citation')]
        allowed_citations = [s.get('citation') for s in used_sources if s.get('citation')]
        case_names = []
        for c in cases_for_prompt:
            meta = c.get('metadata', {}) if isinstance(c, dict) else {}
            nm = meta.get('case_name')
            if nm:
                case_names.append(nm)
        case_markers = [m for m in (case_cites + case_names) if m]
        primary_case_cite = case_cites[0] if case_cites else (case_names[0] if case_names else "")

        def contains_negative_case_claim(text: str) -> bool:
            low = (text or '').lower()
            return any(p in low for p in [
                "could not find",
                "cannot find",
                "i couldn't find",
                "i cannot find",
                "no direct references",
                "no recent",
            ])

        def mentions_any_marker(text: str, markers: list[str]) -> bool:
            low = (text or '').lower()
            return any(str(m).lower() in low for m in markers if m)

        def has_disallowed_citation(text: str) -> bool:
            extracted = extract_citations_from_text(text or '')
            if not extracted:
                return False
            allowed_low = [str(a).lower() for a in allowed_citations if a]
            for c in extracted:
                cl = str(c).lower()
                # allow if citation matches or overlaps an allowed citation string
                if any((cl in a) or (a in cl) for a in allowed_low):
                    continue
                return True
            return False

        repair_attempts = max(1, getattr(Config, 'MAX_CASE_REPAIR_ATTEMPTS', 2))
        need_repair = (
            bool(case_markers)
            and (
                contains_negative_case_claim(resp)
                or not mentions_any_marker(resp, case_markers)
                or has_disallowed_citation(resp)
            )
        )

        for _ in range(repair_attempts):
            if not need_repair:
                break

            repair_prompt = f"""You wrote a draft answer that did not properly use available case sources.

QUESTION:
{state['query']}

MANDATORY CASE SOURCES (must be used and discussed first):
{case_context}

OTHER SOURCES (optional, use only if directly relevant):
{other_context}

DRAFT ANSWER TO REVISE:
{resp}

Rewrite the answer so it is conversational and summary-like.
Rules:
- Start with the most relevant case source and explain it in plain language.
- The first paragraph must explicitly mention this case citation: {primary_case_cite}
- Explicitly connect that case to the Voting Rights Act only using the provided sources.
- Do not claim that no recent case exists if a case source is provided.
- Do not cite or discuss authorities that are not in the provided sources.
- Keep the response concise and readable.
- End with a short citations list using only provided citations.
"""
            resp = _call_ollama(repair_prompt)
            need_repair = (
                contains_negative_case_claim(resp)
                or not mentions_any_marker(resp, case_markers)
                or has_disallowed_citation(resp)
            )
    except Exception as e:
        logger.error(f"Error in case-repair pass: {e}")

    state['final_answer'] = resp
    state['citations'] = extract_citations_from_text(resp)
    state['used_sources'] = used_sources
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
