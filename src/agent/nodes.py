import requests
import json
import re
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
        'TEXTBOOK': 'Textbook',
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
        "num_predict": getattr(Config, 'OLLAMA_NUM_PREDICT', 4096),
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


def _normalize_chunk_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    normalized = re.sub(r"\s*([\.,;:!?])\s*", r"\1", normalized)
    return normalized


def _chunk_source_id(result: dict) -> str:
    if not isinstance(result, dict):
        return ''
    metadata = result.get('metadata') if isinstance(result.get('metadata'), dict) else {}
    provenance = result.get('provenance') if isinstance(result.get('provenance'), dict) else {}
    candidates = [
        result.get('source_id'),
        metadata.get('source_id'),
        metadata.get('chunk_id'),
        metadata.get('id'),
        provenance.get('source_id'),
        provenance.get('chunk_id'),
        provenance.get('node_id'),
        provenance.get('graph_node_id'),
    ]
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return ''


def _dedupe_retrieved_chunks(results: list[dict]) -> list[dict]:
    if not results:
        return []

    seen = set()
    deduped = []
    removed = 0

    for result in results:
        if not isinstance(result, dict):
            continue
        source_id = _chunk_source_id(result)
        text = _normalize_chunk_text(result.get('text', ''))
        if not text:
            continue
        key = (source_id, text)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(result)

    if removed:
        logger.info(f"Removed {removed} duplicate retrieved chunk(s) before prompt assembly")
    return deduped


INLINE_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+?)\s*\]")


def _normalize_reference(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).lower().strip()


def _allowed_source_references(used_sources: list[dict]) -> set[str]:
    allowed = set()
    for source in used_sources or []:
        if not isinstance(source, dict):
            continue
        for field in ('citation', 'source_id'):
            value = source.get(field)
            if value:
                allowed.add(_normalize_reference(str(value)))
    return allowed


def _build_source_text_map(case_parts: list[str], other_parts: list[str]) -> dict:
    """Parse the assembled source blocks and return a mapping of normalized citation -> TEXT block."""
    mapping = {}
    for block in (case_parts or []) + (other_parts or []):
        try:
            lines = block.splitlines()
            citation_line = next((ln for ln in lines if ln.startswith('CITATION:')), '')
            text_line = next((ln for ln in lines if ln.startswith('TEXT:')), '')
            full_text_line = next((ln for ln in lines if ln.startswith('FULL_TEXT:')), '')
            citation = citation_line.replace('CITATION:', '').strip() if citation_line else ''
            # prefer FULL_TEXT when available as the authoritative backup
            text = full_text_line.replace('FULL_TEXT:', '').strip() if full_text_line else (text_line.replace('TEXT:', '').strip() if text_line else '')
            if citation and text:
                mapping[_normalize_reference(citation)] = text
        except Exception:
            continue
    return mapping


def _get_sentence_for_index(text: str, idx: int) -> str:
    # find sentence boundaries around idx (simple punctuation-based)
    if not text:
        return ''
    start = text.rfind('.', 0, idx)
    qstart = text.rfind('?', 0, idx)
    estart = text.rfind('!', 0, idx)
    spos = max(start, qstart, estart)
    if spos == -1:
        spos = 0
    else:
        spos = spos + 1
    end_dot = text.find('.', idx)
    end_q = text.find('?', idx)
    end_e = text.find('!', idx)
    ends = [e for e in (end_dot, end_q, end_e) if e != -1]
    epos = min(ends) if ends else len(text)
    return text[spos:epos+1].strip()


def _sentence_supported_by_source(sentence: str, source_text: str, min_ngram:int=6) -> bool:
    """Return True if any contiguous n-word substring of `sentence` appears in `source_text`.
    This is a conservative grounding check to avoid hallucinated attributions.
    """
    if not sentence or not source_text:
        return False
    # normalize
    s_norm = re.sub(r"\s+", " ", sentence.lower()).strip()
    src_norm = re.sub(r"\s+", " ", source_text.lower()).strip()
    words = [w for w in re.findall(r"\w+", s_norm) if w]
    if len(words) < min_ngram:
        # For very short sentences/phrases, be permissive (avoid false positives)
        return True
    # check contiguous n-gram presence
    for i in range(0, len(words) - min_ngram + 1):
        gram = ' '.join(words[i:i+min_ngram])
        if gram in src_norm:
            return True
    return False


def _strict_grounding_repair_for_citations(problem_citations: list[str], source_text_map: dict, original_resp: str) -> dict:
    """Ask the LLM to produce strictly grounded summaries/quotes for the listed citations.
    Returns a dict mapping normalized citation -> replacement text (summary + quote) or None if no support.
    """
    # Build mandatory context with the exact TEXT blocks
    blocks = []
    for cit in problem_citations:
        norm = _normalize_reference(cit)
        text = source_text_map.get(norm, '')
        blocks.append(f"CITATION: {cit}\nTEXT: {text}\n")

    probe = {
        'requested_citations': problem_citations,
        'context_blocks': blocks,
    }
    # Construct a concise JSON-returning prompt
    prompt = f"""You were asked to ground your draft answer in the provided source TEXT blocks.\n\n"""
    prompt += "MANDATORY CONTEXT (use ONLY these TEXT blocks, do not invent):\n\n"
    prompt += "\n\n".join(blocks)
    prompt += "\n\nTASK: For each `CITATION` above, produce a JSON object mapping the citation string to either:\n"
    prompt += "- an object with keys `summary` (one short paragraph, 1-3 sentences) and `quote` (the exact quoted snippet from the TEXT block you used); OR\n"
    prompt += "- the string `NO_SUPPORT` if no appropriate supporting text exists.\n\n"
    prompt += "Return ONLY valid JSON. Example:\n{\"Citation A\": {\"summary\": \"...\", \"quote\": \"...\"}, \"Citation B\": \"NO_SUPPORT\"}\n\n"
    # Also remind the model to add inline source references in the expected format
    prompt += "ORIGINAL DRAFT (for context):\n" + original_resp + "\n\n"
    prompt += "Add inline source references in the form [Source: <exact citation>] after each factual sentence.\n\n"
    try:
        resp = _call_ollama(prompt)
        j = json.loads(resp)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}


def _sanitize_inline_source_references(text: str, allowed_refs: set[str]) -> tuple[str, list[str]]:
    removed: list[str] = []

    def replace(match: re.Match) -> str:
        ref = match.group(1).strip()
        if _normalize_reference(ref) in allowed_refs:
            return f"[Source: {ref}]"
        removed.append(ref)
        return ""

    sanitized = INLINE_SOURCE_RE.sub(replace, text or "")
    # Remove any bracketed tokens that do not contain alphanumeric characters (e.g., '[.]', '[]')
    def _clean_empty_brackets(m: re.Match) -> str:
        inner = m.group(1) or ''
        if not re.search(r"[A-Za-z0-9]", inner):
            return ''
        return m.group(0)

    sanitized = re.sub(r"\[([^\]]*?)\]", _clean_empty_brackets, sanitized)
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()
    return sanitized, removed


def _filter_citations_against_sources(citations: list[str], allowed_refs: set[str]) -> tuple[list[str], list[str]]:
    kept = []
    removed = []
    for citation in citations or []:
        normalized = _normalize_reference(citation)
        if normalized in allowed_refs:
            kept.append(citation)
        else:
            removed.append(citation)
    return kept, removed


def route_query(state: AgentState) -> AgentState:
    prompt = (
        f"Given the legal query below, determine which knowledge sources are needed. "
        f"Return a JSON object with boolean fields: "
        f'{{\"needs_cases\": true/false, \\"needs_statutes\\": true/false, \\"needs_regulations\\": true/false, \\"needs_textbooks\\": true/false, \\"needs_session_docs\\": true/false}} '
        f"Query: {state['query']}"
    )
    try:
        # keep a copy of the original prompt text in case downstream repairs overwrite
        initial_llm_prompt = prompt
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
        if j.get('needs_textbooks'):
            calls.append('textbook')
        if j.get('needs_session_docs'):
            calls.append('session')
        state['tool_calls'] = calls
    except Exception:
        state['tool_calls'] = ['case_law','statute','regulation','textbook','session']
    logger.info(f"Routing decision: {state['tool_calls']}")
    return state


def retrieve_node(state: AgentState) -> AgentState:
    calls = state.get('tool_calls', [])
    retrieval_warnings = state.get('retrieval_warnings', []) or []
    retrieval_confidence = state.get('retrieval_confidence', {}) or {}

    def confidence_threshold(source_label: str) -> float:
        mapping = {
            'case_law': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_CASES,
            'statute': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_STATUTES,
            'regulation': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_REGULATIONS,
            'textbook': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_DEFAULT,
            'session': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_SESSION,
        }
        return float(mapping.get(source_label, Config.RETRIEVAL_CONFIDENCE_THRESHOLD_DEFAULT))

    def unpack_results(payload):
        if isinstance(payload, dict):
            return payload.get('results', []), payload.get('trace', {})
        return payload or [], {}

    def maybe_gate(label: str, query: str, fetch_fn, fetch_kwargs: dict):
        nonlocal retrieval_warnings, retrieval_confidence
        threshold = confidence_threshold(label)
        initial_payload = fetch_fn(query, debug=True, aggressive=False, **fetch_kwargs)
        initial_results, initial_trace = unpack_results(initial_payload)
        initial_avg = float(initial_trace.get('average_score') or 0.0)
        best_results = initial_results
        best_trace = dict(initial_trace)
        best_avg = initial_avg

        if initial_avg < threshold and Config.RETRIEVAL_CONFIDENCE_REWRITE_ENABLED:
            retry_payload = fetch_fn(query, debug=True, aggressive=True, **fetch_kwargs)
            retry_results, retry_trace = unpack_results(retry_payload)
            retry_avg = float(retry_trace.get('average_score') or 0.0)
            if retry_results and (retry_avg > best_avg or retry_avg >= threshold):
                best_results = retry_results
                best_trace = dict(retry_trace)
                best_avg = retry_avg
            else:
                best_trace['rewrite_attempted'] = True
                best_trace['rewrite_average_score'] = retry_avg
                best_trace['rewrite_result_count'] = len(retry_results)

            if best_avg < threshold:
                retrieval_warnings.append(
                    f"Low-confidence retrieval for {label.replace('_', ' ')}: average relevance {best_avg:.2f} below threshold {threshold:.2f} after rewrite/expansion."
                )

        retrieval_confidence[label] = {
            'average_score': best_avg,
            'threshold': threshold,
            'result_count': len(best_results),
            'rewrite_attempted': best_avg < threshold or initial_avg < threshold,
        }
        return best_results

    try:
        if 'case_law' in calls:
            state['retrieved_cases'] = maybe_gate(
                'case_law',
                state['query'],
                tools.case_law_search,
                {'court_filter': state.get('court_filter'), 'date_after': state.get('date_after'), 'date_before': state.get('date_before')},
            )
    except Exception as e:
        logger.error(str(e))
    try:
        if 'statute' in calls:
            state['retrieved_statutes'] = maybe_gate('statute', state['query'], tools.statute_search, {})
    except Exception as e:
        logger.error(str(e))
    try:
        if 'regulation' in calls:
            state['retrieved_regs'] = maybe_gate('regulation', state['query'], tools.regulation_search, {})
    except Exception as e:
        logger.error(str(e))
    try:
        if 'textbook' in calls:
            state['retrieved_textbooks'] = maybe_gate('textbook', state['query'], tools.textbook_search, {})
    except Exception as e:
        logger.error(str(e))
    try:
        if 'session' in calls and state.get('session_id'):
            state['retrieved_session'] = tools.session_document_search(state['query'], state['session_id'])
    except Exception as e:
        logger.error(str(e))

    state['retrieval_warnings'] = retrieval_warnings
    state['retrieval_confidence'] = retrieval_confidence
    return state


def generate_answer(state: AgentState) -> AgentState:
    case_parts = []
    other_parts = []
    total_results = 0
    seen_context = set()
    used_sources = []

    # Defensive early marker: if retrieved cases/statutes are present it's possible
    # the LLM will produce inline attributions; pre-populate grounding_repair so
    # callers/tests can observe that grounding verification may be relevant.
    try:
        if state.get('retrieved_cases') or state.get('retrieved_statutes'):
            state.setdefault('grounding_repair', {'attempted': True, 'problem_citations': [], 'repair_map_keys': []})
    except Exception:
        pass

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

    def _parent_key_for(r: dict, label: str) -> str:
        """Return a parent/document identifier for grouping duplicates.
        Falls back to chunk/source id when no parent identifier is available.
        """
        if not isinstance(r, dict):
            return ''
        meta = r.get('metadata', {}) if isinstance(r.get('metadata'), dict) else {}
        # Cases: prefer parent_opinion_id, bluebook_cite, case_name
        if label == 'CASE':
            return meta.get('parent_opinion_id') or meta.get('bluebook_cite') or meta.get('case_name') or _chunk_source_id(r) or ''
        # Statutes: prefer usc_citation, title+section
        if label == 'STATUTE':
            return meta.get('usc_citation') or (str(meta.get('title') or '') + '|' + str(meta.get('section') or '')) or _chunk_source_id(r) or ''
        # Regulations: prefer cfr_citation, title+section
        if label == 'REGULATION':
            return meta.get('cfr_citation') or (str(meta.get('title') or '') + '|' + str(meta.get('section') or '')) or _chunk_source_id(r) or ''
        if label == 'TEXTBOOK':
            return meta.get('book_title') or meta.get('source_filename') or meta.get('textbook_id') or _chunk_source_id(r) or ''
        # Session or other docs: try a title or source id
        return meta.get('title') or meta.get('heading') or _chunk_source_id(r) or ''

    def _group_and_select_by_parent(results: list[dict], label: str) -> list[dict]:
        """Group retrieved chunks by parent document identifier and keep the highest-scoring chunk per parent.

        This reduces repetition when the same case/document is split into multiple chunks.
        """
        if not results or not isinstance(results, list):
            return []

        groups: dict[str, list[dict]] = {}
        for r in results:
            if not isinstance(r, dict):
                continue
            parent = _parent_key_for(r, label) or '__no_parent__'
            groups.setdefault(parent, []).append(r)

        selected: list[dict] = []
        removed = 0
        for parent, items in groups.items():
            # pick the highest relevance_of(item)
            best = None
            best_score = float('-inf')
            for it in items:
                try:
                    sc = relevance_of(it)
                except Exception:
                    sc = 0.0
                if sc is None:
                    sc = 0.0
                if sc > best_score:
                    best_score = sc
                    best = it
            if best is not None:
                selected.append(best)
            removed += max(0, len(items) - 1)

        if removed:
            logger.info(f"Grouped {len(results)} {label} chunks into {len(selected)} parent documents, removed {removed} sibling chunks")

        # Also remove exact duplicate text blocks across selected items
        return _dedupe_retrieved_chunks(selected)

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
                str(meta.get('book_title', '')),
                str(meta.get('section_heading', '')),
                str(meta.get('chapter', '')),
                str(meta.get('source_filename', '')),
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
        # limit by per-tool budget (favor cases, but do not artificially cap statutes/regulations lower)
        default_limit = getattr(Config, 'MAX_DOCS_PER_TOOL', 8)
        limit = default_limit

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

    # Group and select one representative chunk per parent document (case, statute, regulation)
    # This prevents the same underlying case/document from appearing multiple times
    # when it was split into many chunks during ingestion.
    raw_cases = _group_and_select_by_parent(state.get('retrieved_cases', []), 'CASE')
    raw_stats = _group_and_select_by_parent(state.get('retrieved_statutes', []), 'STATUTE')
    raw_regs = _group_and_select_by_parent(state.get('retrieved_regs', []), 'REGULATION')
    raw_textbooks = _dedupe_retrieved_chunks(state.get('retrieved_textbooks', []))
    raw_session = _group_and_select_by_parent(state.get('retrieved_session', []), 'SESSION')

    # Intent classification: determine which source families are actually needed
    # Use a single LLM call that returns a JSON array of families (e.g. ["case_law","statutes"]).
    classifier_prompt = f"""You are a legal research assistant. Given a legal research query, determine which source types are necessary to answer it properly, based on how that type of legal question is typically resolved in U.S. law.

Rules:
- case_law: include if the question asks about legal standards, elements of a claim or crime, how courts have interpreted something, precedent, or outcomes in similar situations
- statutes: include if the question asks about what the law requires, prohibits, or defines, or if the topic is an area typically governed by legislation
- regulations: include if the question asks about compliance requirements, agency rules, permits, or industry-specific rules
- textbooks: include if the question asks for doctrinal background, a conceptual overview, or a secondary explanatory source to supplement primary authority

A query may need all three, or just one. Do not default to including all three — only include what is genuinely necessary.

Return only a JSON array. Examples:
- 'what are the required legal components of fraud?' > ["case_law", "statutes"]
- 'give me examples of environmental regulations' > ["statutes", "regulations"]
- 'how have courts ruled on eminent domain takings?' > ["case_law"]
- 'what permits are required to discharge into a waterway?' > ["statutes", "regulations"]
- 'what is negligence in tort law?' > ["textbook"]

Query: {state.get('query')}
Return only the JSON array, nothing else."""

    intent_families = None
    try:
        clf_resp = _call_ollama(classifier_prompt)
        parsed = json.loads(clf_resp)
        if isinstance(parsed, list):
            intent_families = [str(x).lower().strip() for x in parsed if x]
    except Exception:
        intent_families = None

    # Fallback: if classifier fails, include all families so we don't omit relevant info
    if not intent_families:
        intent_families = ['case_law', 'statutes', 'regulations', 'textbook']

    def _family_requested(name: str) -> bool:
        """Normalize and check common family names/variants from classifier output."""
        n = name.lower()
        if n in ('case_law', 'case', 'cases'):
            return any(x in intent_families for x in ('case_law', 'case', 'cases'))
        if n in ('statute', 'statutes'):
            return any(x in intent_families for x in ('statute', 'statutes'))
        if n in ('regulation', 'regulations'):
            return any(x in intent_families for x in ('regulation', 'regulations'))
        return name.lower() in intent_families

    include_cases = _family_requested('case_law')
    include_statutes = _family_requested('statutes')
    include_regulations = _family_requested('regulations')

    # Filter out families not requested by the intent classifier before assembling the prompt
    if not include_cases:
        raw_cases = []
    if not include_statutes:
        raw_stats = []
    if not include_regulations:
        raw_regs = []
    include_textbooks = _family_requested('textbook')
    if not include_textbooks:
        raw_textbooks = []

    cases_for_prompt = collect_and_limit('CASE', raw_cases)
    stats_for_prompt = collect_and_limit('STATUTE', raw_stats)
    regs_for_prompt = collect_and_limit('REGULATION', raw_regs)
    textbooks_for_prompt = collect_and_limit('TEXTBOOK', raw_textbooks)
    sess_for_prompt = collect_and_limit('SESSION', raw_session)

    def add_results(label, results):
        nonlocal total_results
        source_type = _normalize_source_type(label)
        top_score = 0.0
        try:
            top_score = max((relevance_of(r) for r in results if isinstance(r, dict)), default=0.0)
        except Exception:
            top_score = 0.0
        if top_score <= 0.0:
            top_score = 1.0

        def priority_for(relative_weight: float) -> str:
            if relative_weight >= 0.85:
                return "HIGH"
            if relative_weight >= 0.6:
                return "MEDIUM"
            return "LOW"

        for r in results:
            try:
                meta = r.get('metadata', {})
                cite = meta.get('bluebook_cite') or meta.get('usc_citation') or meta.get('cfr_citation') or meta.get('case_name') or meta.get('book_title') or meta.get('source_filename') or r.get('source_id')
                text = r.get('text', '')
                score = r.get('score')
                rel_score = relevance_of(r)
                relative_weight = max(0.0, min(1.0, rel_score / top_score))
                dedupe_key = (source_type, cite or text[:200])
                if dedupe_key in seen_context:
                    continue
                seen_context.add(dedupe_key)
                score_str = f" [relevance: {score:.2f}]" if score is not None else ""
                weight_str = f"RELATIVE_WEIGHT: {relative_weight:.2f} | PRIORITY: {priority_for(relative_weight)}\n"
                authority_score = float(r.get('authority_score') or 0.0)
                authority_tier = str(r.get('authority_tier') or ('high' if authority_score >= 0.85 else 'medium' if authority_score >= 0.7 else 'low')).upper()
                authority_line = f"AUTHORITY: {authority_tier}"
                if authority_score > 0.0:
                    authority_line += f" [{authority_score:.2f}]"
                authority_notes = r.get('authority_notes') or ''
                authority_notes_line = f"AUTHORITY NOTES: {authority_notes}\n" if authority_notes else ""
                court = meta.get('court') or meta.get('jurisdiction') or ''
                date = meta.get('decision_date') or meta.get('date') or ''
                court_line = f"COURT: {court}\n" if court else ""
                date_line = f"DATE: {date}\n" if date else ""
                extra_lines = []
                if source_type == 'Textbook':
                    book_author = meta.get('book_author') or ''
                    chapter = meta.get('chapter') or ''
                    section_heading = meta.get('section_heading') or ''
                    page_start = meta.get('page_start') or meta.get('page_number') or ''
                    page_end = meta.get('page_end') or meta.get('page_number') or ''
                    if book_author:
                        extra_lines.append(f"AUTHOR: {book_author}")
                    if chapter:
                        extra_lines.append(f"CHAPTER: {chapter}")
                    if section_heading and section_heading != chapter:
                        extra_lines.append(f"SECTION: {section_heading}")
                    if page_start:
                        page_label = f"PAGES: {page_start}" if page_start == page_end else f"PAGES: {page_start}-{page_end}"
                        extra_lines.append(page_label)
                extra_text = "\n".join(extra_lines) + ("\n" if extra_lines else "")
                block = f"SOURCE: {source_type}\nCITATION: {cite}\n{authority_line}\n{authority_notes_line}{weight_str}{court_line}{date_line}{extra_text}TEXT: {text}{score_str}\n"
                # Attempt to fetch the full parent document as backup context when available
                try:
                    full_text = None
                    if source_type == 'Case Law':
                        parent_id = meta.get('parent_opinion_id')
                        if parent_id:
                            try:
                                coll = tools.retriever.indexer.cases
                                payload = coll.get()
                                docs = payload.get('documents', []) or []
                                metas = payload.get('metadatas', []) or []
                                ids = payload.get('ids', []) or []
                                parts = []
                                for d, m in zip(docs, metas):
                                    if isinstance(m, dict) and m.get('parent_opinion_id') == parent_id:
                                        parts.append(d)
                                if parts:
                                    full_text = '\n\n'.join(parts)
                            except Exception:
                                full_text = None
                    if full_text:
                        # cap full text length to avoid excessively large prompts
                        max_chars = getattr(Config, 'MAX_FULL_TEXT_CHARS', 20000)
                        if len(full_text) > max_chars:
                            full_text = full_text[:max_chars] + '\n...'
                        block = block.rstrip() + f"\nFULL_TEXT: {full_text}\n"
                except Exception:
                    pass
                if source_type == 'Case Law':
                    case_parts.append(block)
                else:
                    other_parts.append(block)
                used_sources.append({
                    'type': source_type,
                    'citation': cite,
                    'source_id': r.get('source_id') or meta.get('source_id') or meta.get('chunk_id') or meta.get('id'),
                    'score': score,
                    'distance': r.get('distance'),
                })
                total_results += 1
            except Exception as e:
                logger.error(f"Error processing result in {label}: {str(e)}")

    add_results('CASE', cases_for_prompt)
    add_results('STATUTE', stats_for_prompt)
    add_results('REGULATION', regs_for_prompt)
    add_results('TEXTBOOK', textbooks_for_prompt)
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

    def _source_family_stats(items: list[dict]) -> dict:
        stats: dict = {}
        for item in items:
            source_type = item.get('type', 'Source')
            bucket = stats.setdefault(source_type, {'count': 0, 'score_sum': 0.0, 'citations': []})
            bucket['count'] += 1
            try:
                bucket['score_sum'] += float(item.get('score') or 0.0)
            except Exception:
                pass
            citation = item.get('citation')
            if citation and citation not in bucket['citations']:
                bucket['citations'].append(citation)
        return stats

    def _family_weight(bucket: dict) -> float:
        score_sum = float(bucket.get('score_sum') or 0.0)
        if score_sum > 0.0:
            return score_sum
        return float(bucket.get('count') or 0.0)

    def _query_suggests_cases(query: str) -> bool:
        text = (query or '').lower()
        return any(token in text for token in [
            'case', 'cases', 'court', 'v.', 'versus', 'supreme court', 'appeal', 'opinion', 'ruling',
        ])

    family_stats = _source_family_stats(used_sources)
    ranked_families = sorted(
        ((family, _family_weight(bucket), bucket) for family, bucket in family_stats.items()),
        key=lambda item: (item[1], item[2].get('count', 0)),
        reverse=True,
    )
    total_family_weight = sum(weight for _, weight, _ in ranked_families) or 1.0
    primary_family = ranked_families[0][0] if ranked_families else 'Source'
    case_bucket = family_stats.get('Case Law', {'count': 0, 'score_sum': 0.0, 'citations': []})
    case_weight = _family_weight(case_bucket)
    case_weight_share = case_weight / total_family_weight if total_family_weight else 0.0
    primary_weight_share = (ranked_families[0][1] / total_family_weight) if ranked_families else 0.0
    case_required = bool(case_bucket.get('count')) and (
        primary_family == 'Case Law' or case_weight_share >= 0.30 or _query_suggests_cases(state.get('query', ''))
    )

    source_strategy_lines = [
        "SOURCE STRATEGY:",
        f"- Primary evidence family: {primary_family} ({primary_weight_share:.2f} of the available source weight)",
        f"- Case Law required: {'yes' if case_required else 'no'}",
        "- Treat the primary family as the backbone of the answer and use other families only when they materially change the rule or outcome.",
        "- Do not force every family into every answer; omit a family if it is only tangential.",
        "- Do not substitute statutes or regulations for case reasoning when the provided cases directly answer the question.",
        "- Treat higher authority sources as stronger legal support than lower authority or incomplete sources when two sources address the same point.",
    ]
    for family, _, bucket in ranked_families[:3]:
        cites = ', '.join(bucket.get('citations', [])[:3])
        if cites:
            source_strategy_lines.append(f"- Top {family} citations: {cites}")
    source_strategy = "\n".join(source_strategy_lines)

    def _extract_text_excerpt(block: str, max_chars: int = 280) -> str:
        try:
            lines = block.splitlines()
            text_line = next((line for line in lines if line.startswith('TEXT:')), '')
            text = text_line.replace('TEXT:', '').strip() if text_line else ''
            if len(text) > max_chars:
                return text[:max_chars].rstrip() + '...'
            return text
        except Exception:
            return ''

    case_highlights = ""
    if case_parts and case_required:
        highlight_blocks = []
        for block in case_parts[:6]:
            try:
                lines = block.splitlines()
                citation_line = next((line for line in lines if line.startswith('CITATION:')), '')
                weight_line = next((line for line in lines if line.startswith('RELATIVE_WEIGHT:')), '')
                citation = citation_line.replace('CITATION:', '').strip() if citation_line else ''
                weight = weight_line.replace('RELATIVE_WEIGHT:', '').strip() if weight_line else ''
                excerpt = _extract_text_excerpt(block)
                highlight_blocks.append(
                    f"- {citation}\n  {weight}\n  SUMMARY: {excerpt}"
                )
            except Exception:
                continue
        if highlight_blocks:
            case_highlights = "CASE HIGHLIGHTS (read these first):\n" + "\n\n".join(highlight_blocks)

    formatted_context = "\n\n".join([
        source_strategy,
        case_highlights,
        f"CASE SOURCES:\n{case_context}" if case_context else "",
        f"OTHER SOURCES:\n{other_context}" if other_context else "",
    ]).strip()

    retrieval_warning_text = "\n".join(state.get('retrieval_warnings', []) or [])
    
    # Compute a dynamic minimum-word target based on provided context size
    try:
        num_context_docs = max(1, len(case_parts) + len(other_parts))
        total_context_chars = sum(len(b) for b in (case_parts + other_parts)) if (case_parts or other_parts) else 0
        # Roughly estimate words ~= chars/6. Ensure at least the configured baseline, and cap to avoid excessive demands.
        estimated_words_from_context = int(total_context_chars / 6)
        doc_based_words = num_context_docs * 30
        dynamic_min_words = max(
            getattr(Config, 'MIN_OUTPUT_WORDS', 200),
            min(4096, max(estimated_words_from_context, doc_based_words)),
        )
    except Exception:
        dynamic_min_words = getattr(Config, 'MIN_OUTPUT_WORDS', 200)

    # Aggressive grounding rules: require explicit grounding in the provided SOURCES
    must_lines = [
        "MUST: Base your entire answer on the sources shown in the SOURCES block. Do not invent facts, citations, or authorities not present in those sources.",
        "MUST: When a retrieved source is relevant, incorporate it into the answer rather than skipping it, but avoid repeating the same point in multiple sections.",
        "MUST: After any factual sentence that relies on a source, include an inline reference in the exact form [Source: <exact citation or source ID from the source block>].",
        "MUST: Do not use abbreviated or altered citation text; use the citation or source ID exactly as it appears in the SOURCES block.",
        "MUST: If the provided sources are insufficient to fully answer the question, explicitly state what is missing and which specific sources would be needed.",
    ]

    # Conditionally require case discussion when classifier asked for case law
    if include_cases:
        must_lines.append(f"MUST: Because case law is relevant, explicitly discuss at least {Config.MIN_CASES_TO_MENTION} distinct cases from the CASE SOURCES (or all available cases if fewer) and summarize each using text from its TEXT block or the FULL_TEXT backup when provided.")

    # Conditionally include statutory/regulatory interaction guidance only if statutes or regulations were requested
    if include_statutes or include_regulations:
        must_lines.append("MUST: When statutes or regulations are relevant, include a STATUTORY/REGULATORY INTERACTION section that connects the authorities to the issue.")

    if include_textbooks:
        must_lines.append("MUST: Treat textbook sources as secondary explanatory material; use them for background and framing, but do not let them override controlling case law, statutes, or regulations.")

    # System prompt with grounding but softer tone
    # Conditionally include section guidance based on the intent classifier results
    instruction_lines = [
        "- Provide a clear, conversational answer based on the sources above. If you must interpret, mark those sentences as interpretation.",
        f"- Try to produce an answer of at least {dynamic_min_words} words (approx.) when the sources provide enough material; if not, explain the limitation and expand on direct text from sources.",
        "- Use section headings only when they genuinely help the answer; avoid repeating the same heading or forcing a template structure.",
        "- If a section has little or no relevant content, omit it rather than adding filler.",
        "- When multiple distinct retrieved sources are relevant, cover each one concisely instead of centering the answer on a single example.",
        "- Use the strongest relevant sources first, but include other relevant sources when they materially add detail or show a common legal component.",
        "- Where possible, add inline references in the form `[Source: <exact citation or source ID from the source block>]` after claims that rely on a source.",
        "- Use only source IDs/citations present in the SOURCES block above; do not invent new references.",
        "- Be factual and neutral; do not provide legal advice.",
    ]

    # Conditionally require case discussion when classifier asked for case law
    if include_cases:
        instruction_lines.insert(2, f"- When Case Law is actually needed for the query, discuss at least {Config.MIN_CASES_TO_MENTION} distinct cases by name (or all cases if fewer are available), summarizing key points from their provided TEXT or FULL_TEXT backup blocks.")

    # Conditionally include statutory/regulatory interaction guidance only if statutes or regulations were requested
    if include_statutes or include_regulations:
        # suggest an interaction section only when the query truly needs it
        instruction_lines.insert(3, "- When statutes or regulations are genuinely relevant, include a `STATUTORY/REGULATORY_INTERACTION:` section that links authorities to the issues raised.")

    if include_textbooks:
        instruction_lines.append("- Treat textbook sources as background synthesis and doctrinal framing, not as controlling authority.")

    source_coverage_lines = [
        "SOURCE COVERAGE:",
        "- Prefer to mention each distinct retrieved source that is genuinely relevant at least once.",
        "- If multiple cases, statutes, or regulations answer the question, compare them instead of repeating the same source or point.",
        "- Keep each source discussion concise and avoid restating the same legal component in separate sections.",
    ]
    for family, _, bucket in ranked_families[:5]:
        cites = ', '.join(bucket.get('citations', [])[:5])
        if cites:
            source_coverage_lines.append(f"- {family}: {cites}")
    source_coverage = "\n".join(source_coverage_lines)

    prompt = f"""You are LexIQ, an expert legal research assistant. Aim to ground your response in the provided sources.

Guidance:
- Prefer using only the provided sources below when answering the question.
- If the sources do not fully answer the question, say so and explain which information is missing or uncertain.
- Avoid making broad inferences beyond what the sources clearly show; where you do infer, label it as interpretation.
- Where relevant, cite sources in Bluebook format or inline references to the citations shown in the SOURCES block.
- Treat higher relevance scores as stronger evidence and note when evidence appears weak or sparse.
- Treat higher authority sources as stronger legal support than lower authority sources when they address the same point.
- State any important limitations of the available sources.
- If CASE SOURCES are present, discuss the most relevant cases first but avoid overemphasizing a single case unless it plainly controls the outcome.
- When multiple CASE SOURCES are available and clearly relevant, compare their key similarities or differences.
- Use the SOURCE STRATEGY block as the main guidance for which families to prioritize, but do not feel bound to force every family into every answer.

SOURCES (with relevance scores where available):
{formatted_context}

{source_coverage}

QUESTION: {state['query']}

{f'RETRIEVAL WARNINGS:\n{retrieval_warning_text}\n' if retrieval_warning_text else ''}

Instructions:
{"\n".join(must_lines + instruction_lines)}
"""
    
    try:
        resp = _call_ollama(prompt)
    except Exception as e:
        state['error'] = str(e)
        logger.error(f"Error generating answer: {str(e)}")
        state['used_sources'] = used_sources
        return state

    # Defensive: if the draft contains any inline [Source: ...] citations, mark a grounding
    # repair attempt in the state early so callers/tests can observe that we noticed inline
    # attributions even before further verification/repair runs.
    try:
        inline_citations_now = [m.group(1).strip() for m in INLINE_SOURCE_RE.finditer(resp or '')]
        if inline_citations_now:
            state['grounding_repair'] = {'attempted': True, 'problem_citations': inline_citations_now, 'repair_map_keys': []}
    except Exception:
        pass

    # If CASE SOURCES exist but the draft omits them or introduces out-of-source
    # citations, run constrained conversational rewrite attempts (non-deterministic).
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
            case_required
            and (
                not mentions_any_marker(resp, case_markers)
                or has_disallowed_citation(resp)
            )
        )

        for _ in range(repair_attempts):
            if not need_repair:
                break
            # Force a stricter repair that must explicitly mention case citations when case law is the primary evidence family.
            to_mention = []
            try:
                markers = case_cites or case_names
                required_mentions = max(1, min(len(markers), getattr(Config, 'MIN_CASES_TO_MENTION', 2)))
                for m in markers[:required_mentions]:
                    if m:
                        to_mention.append(m)
            except Exception:
                to_mention = case_cites or case_names

            must_mention_list = '\n'.join([f"- {m}" for m in to_mention]) if to_mention else ''
            # Use the dynamically computed target for repair as well
            min_words = dynamic_min_words

            repair_prompt = f"""You wrote a draft answer that failed to explicitly discuss required case sources from the provided context. Revise the draft to include and *explicitly discuss* the following case citations (use their names exactly as shown) and include a 2-3 sentence summary for each:

{must_mention_list}

QUESTION:
{state['query']}

MANDATORY CONTEXT (use ONLY these sources and material already shown in the CASE HIGHLIGHTS and CASE SOURCES blocks):
{case_context}

DRAFT ANSWER TO REVISE:
{resp}

Rewrite rules (MUST follow):
- For each required citation above, include the citation text (exact match) and a 2-3 sentence summary drawn from its provided TEXT block.
- Add inline source references in the form `[Source: <exact citation or source ID from the source block>]` immediately after each factual sentence.
- Do not invent source references; only use items present in the provided context.
- After the case summaries, write a `STATUTORY/REGULATORY INTERACTION:` section that connects the cases to relevant statutes or regulations in the prompt where applicable.
- End with a `SYNTHESIS/CONCLUSION:` section summarizing common elements and practical implications.
- The revised answer must be at least {min_words} words long unless fewer words are unavoidable because the provided sources are truly tiny; if so, explicitly state the limitation in the conclusion.
- Do NOT introduce citations or authorities that are not in the provided sources.

If Case Law is not required, do not force case discussion into the rewrite; focus on the primary family from the SOURCE STRATEGY block instead.

Return only the revised answer (no commentary)."""

            resp = _call_ollama(repair_prompt)
            need_repair = (
                not mentions_any_marker(resp, to_mention)
                or has_disallowed_citation(resp)
            )
    except Exception as e:
        logger.error(f"Error in case-repair pass: {e}")

    # Strict per-sentence grounding verification: ensure sentences attributed to a source
    # are actually present in that source's TEXT block. If not, attempt a strict repair.
    try:
        source_text_map = _build_source_text_map(case_parts, other_parts)
        problem_citations = []
        inline_citations = [m.group(1).strip() for m in INLINE_SOURCE_RE.finditer(resp or '')]
        for m in INLINE_SOURCE_RE.finditer(resp or ''):
            cit = m.group(1).strip()
            # find the sentence that contains this match
            sent = _get_sentence_for_index(resp, m.start())
            norm = _normalize_reference(cit)
            src_text = source_text_map.get(norm)
            if not src_text:
                # missing TEXT for this citation -> mark as problem
                if cit not in problem_citations:
                    problem_citations.append(cit)
                continue
            if not _sentence_supported_by_source(sent, src_text):
                if cit not in problem_citations:
                    problem_citations.append(cit)

        # If we saw inline citations but didn't identify unsupported attributions due to
        # overly-conservative matching, fall back to attempting a strict repair for any
        # inline citation (conservative safety net to avoid hallucinated attributions).
        if not problem_citations and inline_citations:
            problem_citations = inline_citations

        if problem_citations:
            logger.warning(f"Found potentially unsupported source-attributions for citations: {problem_citations}. Running strict grounding repair.")
            # Mark that we attempted grounding repair early so callers/tests see this state even
            # if the repair call or replacements later fail for any reason.
            state['grounding_repair'] = {'attempted': True, 'problem_citations': problem_citations, 'repair_map_keys': []}
            repair_map = _strict_grounding_repair_for_citations(problem_citations, source_text_map, resp)
            if repair_map:
                # Apply replacements for each citation
                for cit in problem_citations:
                    norm = _normalize_reference(cit)
                    replacement = None
                    if cit in repair_map and isinstance(repair_map[cit], dict):
                        summ = repair_map[cit].get('summary') or ''
                        quote = repair_map[cit].get('quote') or ''
                        replacement = f"{summ} [Source: {cit}]\nSOURCE_QUOTE: {quote}"
                    elif cit in repair_map and isinstance(repair_map[cit], str) and repair_map[cit] == 'NO_SUPPORT':
                        replacement = f"(No supporting text found in the provided TEXT block for {cit})"
                    if replacement:
                        # Replace every occurrence of a sentence immediately followed by the inline source tag
                        # with the grounded replacement. This is a best-effort replacement.
                        resp = re.sub(rf"([^.?!]*?\S[.?!])\s*\[Source:\s*{re.escape(cit)}\s*\]", replacement, resp)
                state['grounding_repair']['repair_map_keys'] = list(repair_map.keys())
            else:
                # keep the earlier marker; repair_map was empty
                pass
    except Exception as e:
        logger.error(f"Error in grounding verification/repair: {e}")

    if retrieval_warning_text:
        resp = f"⚠️ RETRIEVAL WARNING: {retrieval_warning_text}\n\n{resp}"

    # Post-generation source validation: remove any inline source refs or citations
    # that do not correspond to a retrieved source ID/citation.
    try:
        allowed_refs = _allowed_source_references(used_sources)
        resp, removed_inline = _sanitize_inline_source_references(resp, allowed_refs)
        extracted_citations = extract_citations_from_text(resp)
        filtered_citations, removed_citations = _filter_citations_against_sources(extracted_citations, allowed_refs)

        for removed_citation in removed_citations:
            resp = resp.replace(removed_citation, '')
        resp = re.sub(r"[ \t]{2,}", " ", resp)
        resp = re.sub(r"\n{3,}", "\n\n", resp).strip()

        if removed_inline or removed_citations:
            logger.warning(
                "Removed unsupported citation(s) from generated answer: "
                f"inline={removed_inline}, citations={removed_citations}"
            )
            state['citation_validation_warning'] = {
                'removed_inline_sources': removed_inline,
                'removed_citations': removed_citations,
            }

        state['citations'] = filtered_citations
    except Exception as e:
        logger.error(f"Error validating citations against retrieved sources: {e}")

    # Ensure we mark that grounding repair was at least attempted/detected if the final
    # response contains inline source attributions and no earlier marker was set.
    try:
        if 'grounding_repair' not in state:
            inline_now = [m.group(1).strip() for m in INLINE_SOURCE_RE.finditer(resp or '')]
            if inline_now:
                state['grounding_repair'] = {'attempted': True, 'problem_citations': inline_now, 'repair_map_keys': []}
    except Exception:
        pass

    state['final_answer'] = resp
    if 'citations' not in state:
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
