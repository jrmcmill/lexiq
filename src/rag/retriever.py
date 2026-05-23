from src.rag.indexer import Indexer
from src.rag.embedder import Embedder
from src.rag.reranker import Reranker
from src.rag.bm25_index import BM25Index
from src.rag.citation_graph import CitationGraph
from src.config import Config
from src.observability.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    TOPICAL_QUERY_EXPANSIONS = {
        'gerrymander': ['vote dilution', 'racial gerrymandering', 'redistricting', 'districting', 'section 2'],
        'gerrymandering': ['vote dilution', 'racial gerrymandering', 'redistricting', 'districting', 'section 2'],
        'vote dilution': ['gerrymandering', 'racial gerrymandering', 'section 2', 'voting rights act'],
        'racial gerrymandering': ['gerrymandering', 'vote dilution', 'equal protection', 'redistricting'],
        'partisan gerrymandering': ['gerrymandering', 'redistricting', 'districting', 'equal protection'],
        'redistricting': ['districting', 'gerrymandering', 'vote dilution'],
        'districting': ['redistricting', 'gerrymandering', 'vote dilution'],
        'section 2': ['voting rights act', '52 usc 10301', '42 usc 1973', 'vote dilution'],
        'section 5': ['voting rights act', 'preclearance', '52 usc 10304'],
        'voting rights act': ['section 2', 'section 5', 'vote dilution'],
        'equal protection': ['gerrymandering', 'racial gerrymandering', 'redistricting'],
        'qualified immunity': ['clearly established', 'civil rights', '1983'],
        'standing': ['justiciability', 'injury in fact', 'mootness', 'ripeness'],
        'free speech': ['first amendment', 'content based', 'prior restraint'],
    }
    MAX_QUERY_VARIANTS = 5
    MAX_AGGRESSIVE_QUERY_VARIANTS = 8

    def __init__(self):
        self.indexer = Indexer()
        self.embedder = Embedder()
        self.reranker = Reranker()
        self.retrieval_debug = Config.RETRIEVAL_DEBUG
        self.citation_graph = CitationGraph()
        if Config.ENABLE_CITATION_GRAPH:
            self.citation_graph.load()
        # load BM25 indexes (if present)
        persist = Config.CHROMA_PERSIST_DIR
        self.bm25_cases = BM25Index(persist, "cases")
        self.bm25_statutes = BM25Index(persist, "statutes")
        self.bm25_regs = BM25Index(persist, "regs")
        self.bm25_textbooks = BM25Index(persist, "textbooks")
        self.bm25_titles = BM25Index(persist, "titles")
        try:
            self.bm25_cases.load()
            self.bm25_statutes.load()
            self.bm25_regs.load()
            self.bm25_textbooks.load()
            self.bm25_titles.load()
        except Exception:
            pass

    def _safe_float(self, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number or number in (float('inf'), float('-inf')):
            return None
        return number

    def _distance_to_similarity(self, distance):
        value = self._safe_float(distance)
        if value is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - value))

    def _normalize_bm25_score(self, score, top_score):
        score_value = self._safe_float(score)
        top_value = self._safe_float(top_score)
        if score_value is None or top_value is None or top_value <= 0:
            return 0.0
        return max(0.0, min(1.0, score_value / top_value))

    def _source_weights(self, source_kind: str, query: str) -> dict[str, float]:
        base_weights = {
            'cases': {'semantic': 0.42, 'content': 0.23, 'title': 0.30, 'synergy': 0.05},
            'statutes': {'semantic': 0.50, 'content': 0.33, 'title': 0.12, 'synergy': 0.05},
            'regs': {'semantic': 0.50, 'content': 0.33, 'title': 0.12, 'synergy': 0.05},
            'textbooks': {'semantic': 0.48, 'content': 0.34, 'title': 0.13, 'synergy': 0.05},
            'session': {'semantic': 0.85, 'content': 0.15, 'title': 0.0, 'synergy': 0.0},
        }
        weights = dict(base_weights.get(source_kind, base_weights['statutes']))
        normalized_query = self._normalize_query(query)

        if source_kind == 'cases' and (' v ' in f' {normalized_query} ' or 'v.' in normalized_query or ' vs ' in f' {normalized_query} '):
            weights['title'] += 0.08
            weights['content'] -= 0.03
            weights['semantic'] -= 0.05

        if source_kind in {'statutes', 'regs'} and (
            any(token in normalized_query for token in ('section', 'usc', 'cfr', 'title', 'part')) or any(ch.isdigit() for ch in normalized_query)
        ):
            weights['title'] += 0.05
            weights['content'] += 0.02
            weights['semantic'] -= 0.07

        if source_kind == 'textbooks':
            weights['content'] += 0.05
            weights['semantic'] -= 0.03

        total = sum(max(value, 0.0) for value in weights.values()) or 1.0
        return {key: max(value, 0.0) / total for key, value in weights.items()}

    def _combine_scores(self, source_kind: str, query: str, semantic_score: float, bm25_score: float, title_score: float) -> float:
        weights = self._source_weights(source_kind, query)
        semantic = max(0.0, min(1.0, semantic_score or 0.0))
        bm25 = max(0.0, min(1.0, bm25_score or 0.0))
        title = max(0.0, min(1.0, title_score or 0.0))

        synergy = 0.0
        if semantic > 0 and bm25 > 0:
            synergy += weights['synergy'] * min(semantic, bm25)
        if title > 0 and source_kind == 'cases' and semantic > 0:
            synergy += 0.03 * min(semantic, title)
        if title > 0 and source_kind in {'statutes', 'regs'} and bm25 > 0:
            synergy += 0.02 * min(bm25, title)

        return (
            weights['semantic'] * semantic
            + weights['content'] * bm25
            + weights['title'] * title
            + synergy
        )

    def _rerank_blend_weights(self, source_kind: str) -> tuple[float, float]:
        if source_kind == 'cases':
            return (0.58, 0.42)
        if source_kind in {'statutes', 'regs'}:
            return (0.52, 0.48)
        if source_kind == 'textbooks':
            return (0.50, 0.50)
        return (0.50, 0.50)

    def _graph_blend_weight(self, source_kind: str) -> float:
        if source_kind == 'cases':
            return 0.10
        if source_kind in {'statutes', 'regs'}:
            return 0.08
        if source_kind == 'textbooks':
            return 0.03
        return 0.05

    def _authority_weight(self, source_kind: str) -> float:
        if source_kind == 'cases':
            return 0.40
        if source_kind in {'statutes', 'regs'}:
            return 0.22
        if source_kind == 'textbooks':
            return 0.10
        return 0.08

    def _authority_profile(self, source_kind: str, metadata: dict | None) -> tuple[float, str, str]:
        meta = metadata if isinstance(metadata, dict) else {}
        source_kind = self._normalize_graph_source_kind(source_kind)

        def _text(*keys: str) -> str:
            parts = []
            for key in keys:
                value = meta.get(key)
                if value:
                    parts.append(str(value))
            return ' '.join(parts).strip().lower()

        if source_kind == 'cases':
            court_text = _text('court', 'court_name', 'court_level', 'jurisdiction', 'court_type')
            notes = []
            if not court_text:
                score = 0.55
                tier = 'medium'
                notes.append('court metadata missing')
            elif any(token in court_text for token in ('supreme court', 'scotus', 'u.s. supreme court')):
                score = 0.98
                tier = 'high'
                notes.append('supreme court authority')
            elif any(token in court_text for token in ('court of appeals', 'court appeals', 'appellate', 'circuit')):
                score = 0.88
                tier = 'high'
                notes.append('appellate authority')
            elif any(token in court_text for token in ('district court', 'bankruptcy court')):
                score = 0.70
                tier = 'medium'
                notes.append('trial-level federal authority')
            elif any(token in court_text for token in ('state supreme', 'state high court')):
                score = 0.92
                tier = 'high'
                notes.append('state high-court authority')
            elif any(token in court_text for token in ('superior court', 'trial court', 'circuit court', 'county court', 'family court')):
                score = 0.56
                tier = 'low'
                notes.append('trial-level state authority')
            else:
                score = 0.64
                tier = 'medium'
                notes.append('general court metadata present')

            if meta.get('bluebook_cite') or meta.get('case_name'):
                score = min(1.0, score + 0.05)
                notes.append('citation metadata present')
            else:
                score = max(0.0, score - 0.05)
                notes.append('citation metadata missing')

            return score, tier, '; '.join(notes)

        if source_kind == 'statutes':
            has_citation = bool(meta.get('usc_citation'))
            has_location = bool(meta.get('title_number') and meta.get('section_number'))
            if has_citation or has_location:
                score = 0.96
                tier = 'high'
                notes = ['codified statute citation present']
            else:
                score = 0.72
                tier = 'medium'
                notes = ['statute citation metadata incomplete']
            if meta.get('section_title'):
                score = min(1.0, score + 0.02)
                notes.append('section title present')
            return score, tier, '; '.join(notes)

        if source_kind == 'regs':
            has_citation = bool(meta.get('cfr_citation'))
            has_location = bool(meta.get('cfr_title') and (meta.get('cfr_part') or meta.get('cfr_section')))
            if has_citation or has_location:
                score = 0.94
                tier = 'high'
                notes = ['regulation citation metadata present']
            else:
                score = 0.70
                tier = 'medium'
                notes = ['regulation citation metadata incomplete']
            if meta.get('section_title'):
                score = min(1.0, score + 0.02)
                notes.append('section title present')
            return score, tier, '; '.join(notes)

        if source_kind == 'textbooks':
            has_title = bool(meta.get('book_title') or meta.get('source_filename') or meta.get('textbook_id'))
            has_location = bool(meta.get('chapter') or meta.get('section_heading') or meta.get('page_number'))
            if has_title or has_location:
                score = 0.62
                tier = 'medium'
                notes = ['textbook background source present']
            else:
                score = 0.45
                tier = 'low'
                notes = ['textbook metadata incomplete']
            if meta.get('page_number'):
                score = min(1.0, score + 0.03)
                notes.append('page metadata present')
            return score, tier, '; '.join(notes)

        has_reference = bool(meta.get('citation') or meta.get('source_id') or meta.get('chunk_id'))
        score = 0.72 if has_reference else 0.58
        tier = 'medium' if has_reference else 'low'
        notes = ['source identifier present' if has_reference else 'source identifier missing']
        return score, tier, '; '.join(notes)

    def _source_collection(self, source_kind: str):
        normalized = self._normalize_graph_source_kind(source_kind)
        if normalized == 'cases':
            return self.indexer.cases
        if normalized == 'statutes':
            return self.indexer.statutes
        if normalized == 'textbooks':
            return self.indexer.textbooks
        return self.indexer.regs

    def _normalize_graph_source_kind(self, source_kind: str) -> str:
        normalized = str(source_kind or '').strip().lower()
        if normalized in {'case', 'cases'}:
            return 'cases'
        if normalized in {'statute', 'statutes', 'usc'}:
            return 'statutes'
        if normalized in {'reg', 'regs', 'regulation', 'regulations', 'cfr'}:
            return 'regs'
        if normalized in {'textbook', 'textbooks', 'book', 'books', 'treatise'}:
            return 'textbooks'
        return normalized

    def _source_confidence_threshold(self, source_kind: str) -> float:
        mapping = {
            'cases': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_CASES,
            'statutes': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_STATUTES,
            'regs': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_REGULATIONS,
            'textbooks': getattr(Config, 'RETRIEVAL_CONFIDENCE_THRESHOLD_TEXTBOOKS', Config.RETRIEVAL_CONFIDENCE_THRESHOLD_DEFAULT),
            'session': Config.RETRIEVAL_CONFIDENCE_THRESHOLD_SESSION,
        }
        return float(mapping.get(self._normalize_graph_source_kind(source_kind), Config.RETRIEVAL_CONFIDENCE_THRESHOLD_DEFAULT))

    def _average_result_score(self, results: list[dict]) -> float:
        scores = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            score = self._safe_float(item.get('score'))
            if score is None:
                score = self._safe_float(item.get('hybrid_score'))
            if score is None:
                score = self._safe_float(item.get('rerank_score'))
            if score is not None:
                scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def _trace_confidence(self, results: list[dict], source_kind: str) -> dict:
        average_score = self._average_result_score(results)
        threshold = self._source_confidence_threshold(source_kind)
        return {
            'average_score': average_score,
            'confidence_threshold': threshold,
            'confidence_below_threshold': average_score < threshold,
            'result_count': len(results or []),
        }

    def _merge_candidate(self, candidate_map: dict, item: dict, source_kind: str):
        key_fn = self._case_result_key if source_kind == 'cases' else self._statute_result_key if source_kind == 'statutes' else self._regulation_result_key
        key = key_fn(item)
        candidate = candidate_map.get(key)
        if candidate is None:
            authority_score, authority_tier, authority_notes = self._authority_profile(source_kind, item.get('metadata') or {})
            candidate = {
                'text': item.get('text', ''),
                'metadata': item.get('metadata', {}) or {},
                'distance': item.get('distance'),
                'semantic_score': item.get('semantic_score', 0.0),
                'bm25_score': item.get('bm25_score', 0.0),
                'title_score': item.get('title_score', 0.0),
                'graph_score': item.get('graph_score', 0.0),
                'authority_score': authority_score,
                'authority_tier': authority_tier,
                'authority_notes': authority_notes,
                'source_id': item.get('source_id'),
                'provenance': item.get('provenance') or {},
                'source_kind': source_kind,
            }
            candidate_map[key] = candidate
            return candidate

        if item.get('text') and not candidate.get('text'):
            candidate['text'] = item.get('text')
        if item.get('metadata') and not candidate.get('metadata'):
            candidate['metadata'] = item.get('metadata')
        item_distance = item.get('distance')
        candidate_distance = candidate.get('distance')
        if candidate_distance is None or (item_distance is not None and item_distance < candidate_distance):
            candidate['distance'] = item_distance
        candidate['semantic_score'] = max(candidate.get('semantic_score', 0.0), item.get('semantic_score', 0.0) or 0.0)
        candidate['bm25_score'] = max(candidate.get('bm25_score', 0.0), item.get('bm25_score', 0.0) or 0.0)
        candidate['title_score'] = max(candidate.get('title_score', 0.0), item.get('title_score', 0.0) or 0.0)
        candidate['graph_score'] = max(candidate.get('graph_score', 0.0), item.get('graph_score', 0.0) or 0.0)
        authority_score, authority_tier, authority_notes = self._authority_profile(source_kind, item.get('metadata') or {})
        candidate['authority_score'] = max(candidate.get('authority_score', 0.0), authority_score)
        if authority_tier == 'high' or candidate.get('authority_tier') not in {'high', 'medium', 'low'}:
            candidate['authority_tier'] = authority_tier
        if authority_notes and not candidate.get('authority_notes'):
            candidate['authority_notes'] = authority_notes
        if item.get('source_id') and not candidate.get('source_id'):
            candidate['source_id'] = item.get('source_id')
        if item.get('provenance') and not candidate.get('provenance'):
            candidate['provenance'] = item.get('provenance')
        return candidate

    def _fetch_graph_documents(self, node_info: dict, source_kind: str) -> list[dict]:
        if not node_info:
            return []
        collection = self._source_collection(source_kind)
        if not collection:
            return []
        chunk_ids = node_info.get('chunk_ids') or []
        if not chunk_ids:
            return []
        try:
            payload = collection.get(ids=chunk_ids)
        except Exception:
            return []

        documents = payload.get('documents', []) or []
        metadatas = payload.get('metadatas', []) or []
        ids = payload.get('ids', []) or chunk_ids
        out = []
        for doc, meta, chunk_id in zip(documents, metadatas, ids):
            if not isinstance(doc, str) or not isinstance(meta, dict):
                continue
            out.append({
                'text': doc,
                'metadata': meta,
                'distance': None,
                'semantic_score': 0.0,
                'bm25_score': 0.0,
                'title_score': 0.0,
                'graph_score': node_info.get('score', 0.0),
                'source_id': chunk_id,
                'provenance': {'graph_node_id': node_info.get('node_id'), 'graph_relation': node_info.get('relation')},
                'source_kind': source_kind,
            })
        return out

    def _expand_with_citation_graph(self, results: list[dict], source_kind: str) -> tuple[list[dict], dict]:
        if not Config.ENABLE_CITATION_GRAPH or not getattr(self.citation_graph, 'nodes', None):
            return results, {'graph_expanded': 0}

        seed_nodes = []
        for item in results[:max(1, min(len(results), 5))]:
            node_id = self.citation_graph.node_for_result(item, source_kind)
            if node_id:
                seed_nodes.append(node_id)
        if not seed_nodes:
            return results, {'graph_expanded': 0}

        expansions = self.citation_graph.expand(
            seed_nodes,
            max_hops=Config.CITATION_GRAPH_HOPS,
            max_nodes=Config.CITATION_GRAPH_MAX_NODES,
        )
        if not expansions:
            return results, {'graph_expanded': 0}

        candidate_map = {}
        for item in results:
            self._merge_candidate(candidate_map, item, source_kind)

        graph_docs = []
        for node_info in expansions:
            node = self.citation_graph.fetch_spec(node_info.get('node_id')) or {}
            graph_docs.extend(self._fetch_graph_documents({**node_info, **node}, source_kind=self._normalize_graph_source_kind(node_info.get('kind', source_kind))))

        for item in graph_docs:
            self._merge_candidate(candidate_map, item, item.get('source_kind', source_kind))

        merged = list(candidate_map.values())
        trace = {'graph_expanded': len(graph_docs), 'graph_nodes': len(expansions), 'seed_nodes': seed_nodes}
        return merged, trace

    def _finalize_results(self, query: str, candidates: list[dict], source_kind: str, n_results: int) -> list[dict]:
        if not candidates:
            return []

        candidate_ordered = sorted(candidates, key=lambda item: item.get('hybrid_score', 0.0), reverse=True)
        rerank_limit = min(len(candidate_ordered), max(n_results, Config.RERANK_TOP_K, n_results * 2))
        try:
            reranked = self.reranker.rerank(query, candidate_ordered, top_k=rerank_limit)
        except Exception as exc:
            logger.warning(f"Reranker failed, falling back to hybrid scores: {exc}")
            reranked = candidate_ordered[:rerank_limit]

        if not reranked:
            return candidate_ordered[:n_results]

        raw_scores = []
        for item in reranked:
            score = self._safe_float(item.get('rerank_score'))
            if score is not None:
                raw_scores.append(score)

        min_rerank = min(raw_scores) if raw_scores else None
        max_rerank = max(raw_scores) if raw_scores else None
        hybrid_weight, rerank_weight = self._rerank_blend_weights(source_kind)

        scored_results = []
        for item in reranked:
            rerank_raw = self._safe_float(item.get('rerank_score'))
            if rerank_raw is None:
                rerank_norm = item.get('hybrid_score', 0.0)
            elif min_rerank is not None and max_rerank is not None and max_rerank > min_rerank:
                rerank_norm = (rerank_raw - min_rerank) / (max_rerank - min_rerank)
            else:
                rerank_norm = 1.0

            hybrid_score = max(0.0, min(1.0, self._safe_float(item.get('hybrid_score')) or 0.0))
            graph_score = max(0.0, min(1.0, self._safe_float(item.get('graph_score')) or 0.0))
            graph_weight = self._graph_blend_weight(source_kind)
            base_weight = max(0.0, 1.0 - graph_weight)
            evidence_score = base_weight * ((hybrid_weight * hybrid_score) + (rerank_weight * rerank_norm)) + (graph_weight * graph_score)
            authority_score = self._safe_float(item.get('authority_score'))
            if authority_score is None:
                authority_score, authority_tier, authority_notes = self._authority_profile(source_kind, item.get('metadata') or {})
                item['authority_score'] = authority_score
                item['authority_tier'] = authority_tier
                item['authority_notes'] = authority_notes
            authority_score = max(0.0, min(1.0, authority_score or 0.0))
            authority_weight = self._authority_weight(source_kind)
            final_score = ((1.0 - authority_weight) * evidence_score) + (authority_weight * authority_score)
            # compute contribution breakdown for transparency
            try:
                weights = self._source_weights(source_kind, query)
            except Exception:
                weights = {'semantic': 0.5, 'content': 0.3, 'title': 0.15, 'synergy': 0.05}
            semantic = max(0.0, min(1.0, item.get('semantic_score') or 0.0))
            bm25 = max(0.0, min(1.0, item.get('bm25_score') or 0.0))
            title = max(0.0, min(1.0, item.get('title_score') or 0.0))
            synergy_val = 0.0
            if semantic > 0 and bm25 > 0:
                synergy_val += weights.get('synergy', 0.0) * min(semantic, bm25)
            if title > 0 and source_kind == 'cases' and semantic > 0:
                synergy_val += 0.03 * min(semantic, title)
            if title > 0 and source_kind in {'statutes', 'regs'} and bm25 > 0:
                synergy_val += 0.02 * min(bm25, title)

            contributions = {
                'semantic': round(weights.get('semantic', 0.0) * semantic, 6),
                'bm25': round(weights.get('content', 0.0) * bm25, 6),
                'title': round(weights.get('title', 0.0) * title, 6),
                'synergy': round(synergy_val, 6),
                'rerank': round(rerank_norm * rerank_weight, 6),
                'graph': round(graph_weight * graph_score, 6),
                'authority': round(authority_weight * authority_score, 6),
            }

            scored_results.append({
                'text': item.get('text', ''),
                'metadata': item.get('metadata', {}),
                'score': final_score,
                'hybrid_score': hybrid_score,
                'rerank_score': rerank_raw,
                'distance': item.get('distance'),
                'semantic_score': item.get('semantic_score'),
                'bm25_score': item.get('bm25_score'),
                'title_score': item.get('title_score'),
                'authority_score': authority_score,
                'authority_tier': item.get('authority_tier', 'medium' if authority_score >= 0.7 else 'low'),
                'authority_notes': item.get('authority_notes', ''),
                'contributions': contributions,
                'source_id': item.get('source_id'),
                'provenance': item.get('provenance') or {},
            })

        # provenance verification: require minimal citation/ID fields when configured
        def _verify_provenance(item: dict) -> tuple[bool, str]:
            meta = item.get('metadata') or {}
            prov = item.get('provenance') or {}
            # cases: prefer parent_opinion_id or bluebook_cite or case_name
            if source_kind == 'cases':
                if meta.get('parent_opinion_id') or prov.get('cite') or meta.get('case_name'):
                    # check date not in future
                    df = meta.get('date_filed')
                    if df:
                        try:
                            from datetime import datetime, date
                            parsed = datetime.fromisoformat(str(df)).date()
                            if parsed > date.today():
                                return (False, 'future_date')
                        except Exception:
                            pass
                    return (True, '')
                return (False, 'missing_case_id')
            # statutes: require usc_citation or title_number+section_number
            if source_kind == 'statutes':
                if meta.get('usc_citation') or (meta.get('title_number') and meta.get('section_number')):
                    return (True, '')
                return (False, 'missing_statute_citation')
            # regs: require cfr_citation or title/part/section
            if source_kind == 'regs':
                if meta.get('cfr_citation') or (meta.get('cfr_title') and (meta.get('cfr_part') or meta.get('cfr_section'))):
                    return (True, '')
                return (False, 'missing_reg_citation')
            return (True, '')

        # annotate verification and filter if configured to require sources
        verified = []
        removed_count = 0
        for item in scored_results:
            ok, reason = _verify_provenance(item)
            item['provenance_verified'] = ok
            if not ok and Config.REQUIRE_SOURCES:
                removed_count += 1
                continue
            verified.append(item)

        if removed_count > 0:
            logger.info(f"Excluded {removed_count} results lacking required provenance ({source_kind})")

        verified.sort(key=lambda item: item.get('score', 0.0), reverse=True)
        key_fn = self._case_result_key if source_kind == 'cases' else self._statute_result_key if source_kind == 'statutes' else self._regulation_result_key
        return self._dedupe_results(verified[:n_results], key_fn)

    def retrieve_cases(self, query: str, n_results: int = 20, court_filter: str | None = None,
                       date_after: str | None = None, date_before: str | None = None,
                       debug: bool = False, aggressive: bool = False):
        try:
            # Build where clause only if filters are present
            ands = []
            if court_filter:
                ands.append({"court": {"$eq": court_filter}})
            if date_after:
                ands.append({"date_filed": {"$gte": date_after}})
            if date_before:
                ands.append({"date_filed": {"$lte": date_before}})

            query_variants = self._expand_query_variants(query, aggressive=aggressive)
            docs, trace = self._collect_candidates(
                collection=self.indexer.cases,
                query_variants=query_variants,
                n_results=n_results,
                where={"$and": ands} if ands else None,
                bm25_index=self.bm25_cases,
                source_kind='cases',
            )
            docs, graph_trace = self._expand_with_citation_graph(docs, 'cases')
            trace.update(graph_trace)

            if not docs:
                logger.warning(f"No case results above relevance threshold {Config.RETRIEVAL_MIN_DISTANCE}")
                return []

            out = self._finalize_results(query, docs, 'cases', n_results)
            if not out:
                logger.warning(f"No case results above reranking threshold {Config.RERANK_MIN_SCORE}")
            trace.update(self._trace_confidence(out, 'cases'))
            trace['aggressive_rewrite'] = aggressive
            if debug or self.retrieval_debug:
                return {'results': out, 'trace': trace}
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_cases: {str(e)}")
            return []

    def retrieve_statutes(self, query: str, n_results: int = 20, debug: bool = False, aggressive: bool = False):
        try:
            query_variants = self._expand_query_variants(query, aggressive=aggressive)
            docs, trace = self._collect_candidates(
                collection=self.indexer.statutes,
                query_variants=query_variants,
                n_results=n_results,
                bm25_index=self.bm25_statutes,
                source_kind='statutes',
            )
            docs, graph_trace = self._expand_with_citation_graph(docs, 'statutes')
            trace.update(graph_trace)

            if not docs:
                logger.warning(f"No statute results above relevance threshold {Config.RETRIEVAL_MIN_DISTANCE}")
                return []

            out = self._finalize_results(query, docs, 'statutes', n_results)
            trace.update(self._trace_confidence(out, 'statutes'))
            trace['aggressive_rewrite'] = aggressive
            if debug or self.retrieval_debug:
                return {'results': out, 'trace': trace}
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_statutes: {str(e)}")
            return []

    def retrieve_textbooks(self, query: str, n_results: int = 20, debug: bool = False, aggressive: bool = False):
        try:
            query_variants = self._expand_query_variants(query, aggressive=aggressive)
            docs, trace = self._collect_candidates(
                collection=self.indexer.textbooks,
                query_variants=query_variants,
                n_results=n_results,
                bm25_index=self.bm25_textbooks,
                source_kind='textbooks',
            )

            if not docs:
                logger.warning(f"No textbook results above relevance threshold {Config.RETRIEVAL_MIN_DISTANCE}")
                return []

            out = self._finalize_results(query, docs, 'textbooks', n_results)
            trace.update(self._trace_confidence(out, 'textbooks'))
            trace['aggressive_rewrite'] = aggressive
            if debug or self.retrieval_debug:
                return {'results': out, 'trace': trace}
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_textbooks: {str(e)}")
            return []

    def retrieve_regulations(self, query: str, n_results: int = 20, debug: bool = False, aggressive: bool = False):
        try:
            query_variants = self._expand_query_variants(query, aggressive=aggressive)
            docs, trace = self._collect_candidates(
                collection=self.indexer.regs,
                query_variants=query_variants,
                n_results=n_results,
                bm25_index=self.bm25_regs,
                source_kind='regs',
            )
            docs, graph_trace = self._expand_with_citation_graph(docs, 'regs')
            trace.update(graph_trace)

            if not docs:
                logger.warning(f"No regulation results above relevance threshold {Config.RETRIEVAL_MIN_DISTANCE}")
                return []

            out = self._finalize_results(query, docs, 'regs', n_results)
            trace.update(self._trace_confidence(out, 'regs'))
            trace['aggressive_rewrite'] = aggressive
            if debug or self.retrieval_debug:
                return {'results': out, 'trace': trace}
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_regulations: {str(e)}")
            return []

    def retrieve_session_docs(self, query: str, session_id: str, n_results: int = 10) -> list[dict]:
        try:
            emb = self.embedder.embed_query(query)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                return []
            
            try:
                coll = self.indexer.client.get_collection(f"session_{session_id}")
            except Exception:
                return []
            
            res = coll.query(query_embeddings=[emb], n_results=n_results)
            out = []
            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []
            
            # Filter by distance threshold
            min_dist = Config.RETRIEVAL_MIN_DISTANCE
            filtered_count = 0
            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    if dist <= min_dist:
                        out.append({'text': d, 'metadata': m, 'score': None, 'distance': dist})
                    else:
                        filtered_count += 1
            
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} low-quality session results (distance > {min_dist})")
            
            return out
        except Exception as e:
            logger.error(f"Error in retrieve_session_docs: {str(e)}")
            return []

    def _dedupe_results(self, results: list[dict], key_fn):
        seen = set()
        deduped = []
        for result in results:
            key = key_fn(result)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    def _case_result_key(self, result: dict):
        meta = result.get('metadata', {}) if isinstance(result, dict) else {}
        if isinstance(meta, dict):
            parent_id = meta.get('parent_opinion_id')
            if parent_id:
                return ('case', parent_id)
            cite = meta.get('bluebook_cite') or meta.get('case_name')
            if cite:
                return ('case', cite)
        return ('case', result.get('text', ''))

    def _statute_result_key(self, result: dict):
        meta = result.get('metadata', {}) if isinstance(result, dict) else {}
        if isinstance(meta, dict):
            title = meta.get('title_number')
            section = meta.get('section_number')
            if title or section:
                return ('statute', title, section)
            cite = meta.get('usc_citation')
            if cite:
                return ('statute', cite)
        return ('statute', result.get('text', ''))

    def _regulation_result_key(self, result: dict):
        meta = result.get('metadata', {}) if isinstance(result, dict) else {}
        if isinstance(meta, dict):
            title = meta.get('cfr_title')
            part = meta.get('cfr_part')
            section = meta.get('cfr_section')
            if title or part or section:
                return ('regulation', title, part, section)
            cite = meta.get('cfr_citation')
            if cite:
                return ('regulation', cite)
        return ('regulation', result.get('text', ''))

    def retrieve_by_title(self, query: str, source: str = 'cases', n_results: int = 10) -> list[dict]:
        """Search titles (case names, statute headings, regulation headings) using BM25 and return linked documents."""
        try:
            bm25 = None
            coll = None
            if source == 'cases':
                bm25 = getattr(self, 'bm25_titles', None)
                coll = self.indexer.titles
            elif source == 'statutes' or source == 'regs':
                bm25 = getattr(self, 'bm25_titles', None)
                coll = self.indexer.titles
            else:
                bm25 = getattr(self, 'bm25_titles', None)
                coll = self.indexer.titles

            if not bm25:
                return []

            top = bm25.query(query, top_k=n_results)
            ids = [t[0] for t in top]
            if not ids:
                return []

            try:
                res = coll.get(ids=ids)
                docs = res.get('documents', [])
                metas = res.get('metadatas', [])
                out = []
                for d, m in zip(docs, metas):
                    out.append({'text': d, 'metadata': m})
                return out
            except Exception:
                return []
        except Exception as e:
            logger.error(f"Error in retrieve_by_title: {e}")
            return []

    def _expand_query_variants(self, query: str, aggressive: bool = False) -> list[str]:
        normalized = self._normalize_query(query)
        # guard: if expansion disabled or query looks like a citation/case name, avoid expansions
        if not Config.ENABLE_QUERY_EXPANSION:
            return [normalized] if normalized else []
        # treat queries that look like citations or structured refs as ineligible for expansion
        citation_like = False
        if any(tok in normalized for tok in ('usc', 'cfr', 'section', '§')):
            citation_like = True
        if ' v ' in f' {normalized} ' or 'v.' in normalized or ' vs ' in f' {normalized} ':
            citation_like = citation_like or True
        # numeric-heavy queries (likely statute/regulation refs)
        import re
        if re.search(r"\b\d{1,3}\b", normalized) and re.search(r"(usc|cfr|section|title|part)", normalized):
            citation_like = True
        if citation_like:
            return [normalized] if normalized else []
        variants: list[str] = []
        seen: set[str] = set()

        def add_variant(value: str | None):
            cleaned = self._normalize_query(value or "")
            if not cleaned:
                return
            if cleaned in seen:
                return
            seen.add(cleaned)
            variants.append(cleaned)

        add_variant(query)

        topical_terms: list[str] = []
        for needle, expansions in self.TOPICAL_QUERY_EXPANSIONS.items():
            if needle in normalized:
                topical_terms.extend(expansions)

        if 'section 2' in normalized or '§ 2' in normalized or '52 usc 10301' in normalized or '42 usc 1973' in normalized:
            topical_terms.extend(['section 2', 'voting rights act', '52 usc 10301', '42 usc 1973', 'vote dilution'])
        if 'section 5' in normalized or '§ 5' in normalized or '52 usc 10304' in normalized:
            topical_terms.extend(['section 5', 'voting rights act', 'preclearance', '52 usc 10304'])

        combined_terms: list[str] = []
        for term in topical_terms:
            cleaned = self._normalize_query(term)
            if cleaned and cleaned not in seen:
                add_variant(cleaned)
                combined_terms.append(cleaned)
            if len(variants) >= self.MAX_QUERY_VARIANTS:
                return variants[:self.MAX_QUERY_VARIANTS]

        if combined_terms and len(variants) < (self.MAX_AGGRESSIVE_QUERY_VARIANTS if aggressive else self.MAX_QUERY_VARIANTS):
            combo_terms = combined_terms if aggressive else combined_terms[:3]
            combo = ' '.join([query] + combo_terms[:5])
            add_variant(combo)

        limit = self.MAX_AGGRESSIVE_QUERY_VARIANTS if aggressive else self.MAX_QUERY_VARIANTS
        return variants[:limit]

    def _normalize_query(self, value: str) -> str:
        return ' '.join((value or '').replace('§', 'section').split()).strip().lower()

    def _collect_candidates(self, collection, query_variants: list[str], n_results: int,
                            where: dict | None = None, bm25_index: BM25Index | None = None,
                            source_kind: str = 'statutes') -> tuple[list[dict], dict]:
        candidate_map: dict = {}
        min_dist = Config.RETRIEVAL_MIN_DISTANCE
        filtered_count = 0
        expansion_limit = max(5, n_results // 2)
        key_fn = self._case_result_key if source_kind == 'cases' else self._statute_result_key if source_kind == 'statutes' else self._regulation_result_key
        trace = {
            'query_variants': list(query_variants),
            'filtered_by_distance': 0,
            'bm25_hits': 0,
            'title_hits': 0,
            'candidate_count': 0,
        }

        def ensure_candidate(key, text='', metadata=None):
            candidate = candidate_map.get(key)
            if candidate is None:
                candidate = {
                    'text': text,
                    'metadata': metadata or {},
                    'distance': None,
                    'semantic_score': 0.0,
                    'bm25_score': 0.0,
                    'title_score': 0.0,
                    'source_id': None,
                    'provenance': {},
                    'source_kind': source_kind,
                }
                candidate_map[key] = candidate
            else:
                if text and not candidate.get('text'):
                    candidate['text'] = text
                if metadata and not candidate.get('metadata'):
                    candidate['metadata'] = metadata
            # populate a lightweight source identifier and provenance when available
            try:
                meta = metadata or {}
                if isinstance(meta, dict):
                    if not candidate.get('source_id'):
                        # prefer parent_opinion_id for cases, else try known id fields
                        candidate['source_id'] = meta.get('parent_opinion_id') or meta.get('ref_id') or meta.get('id') or meta.get('usc_citation') or meta.get('cfr_citation')
                    if not candidate.get('provenance'):
                        prov = {}
                        if meta.get('bluebook_cite'):
                            prov['cite'] = meta.get('bluebook_cite')
                        if meta.get('case_name'):
                            prov.setdefault('case_name', meta.get('case_name'))
                        if meta.get('usc_citation'):
                            prov.setdefault('usc', meta.get('usc_citation'))
                        if meta.get('cfr_citation'):
                            prov.setdefault('cfr', meta.get('cfr_citation'))
                        if meta.get('date_filed'):
                            prov.setdefault('date_filed', meta.get('date_filed'))
                        candidate['provenance'] = prov
            except Exception:
                pass
            return candidate

        for variant_index, variant in enumerate(query_variants):
            limit = n_results if variant_index == 0 else expansion_limit
            emb = self.embedder.embed_query(variant)
            if not emb or not isinstance(emb, list):
                logger.error(f"Invalid embedding returned: {type(emb)}")
                continue

            try:
                query_kwargs = {'query_embeddings': [emb], 'n_results': limit}
                if where:
                    query_kwargs['where'] = where
                res = collection.query(**query_kwargs)
            except Exception:
                continue

            documents = res.get('documents', [[]])[0] if res.get('documents') else []
            metadatas = res.get('metadatas', [[]])[0] if res.get('metadatas') else []
            distances = res.get('distances', [[]])[0] if res.get('distances') else []

            for d, m, dist in zip(documents, metadatas, distances):
                if isinstance(d, str) and isinstance(m, dict):
                    if dist <= min_dist:
                        key = key_fn({'text': d, 'metadata': m})
                        candidate = ensure_candidate(key, d, m)
                        semantic_score = self._distance_to_similarity(dist)
                        candidate['semantic_score'] = max(candidate.get('semantic_score', 0.0), semantic_score)
                        candidate['distance'] = dist if candidate.get('distance') is None else min(candidate['distance'], dist)
                        trace['candidate_count'] = trace.get('candidate_count', 0) + 1
                    else:
                        filtered_count += 1

        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} low-quality results (distance > {min_dist})")

        if bm25_index:
            bm25_scores_by_id: dict[str, float] = {}
            for variant in query_variants:
                try:
                    top_pairs = bm25_index.query(variant, top_k=expansion_limit)
                except Exception:
                    top_pairs = []
                if not top_pairs:
                    continue
                top_score = max((score for _, score in top_pairs), default=0.0)
                for candidate_id, raw_score in top_pairs:
                    norm_score = self._normalize_bm25_score(raw_score, top_score)
                    if norm_score <= 0:
                        continue
                    bm25_scores_by_id[candidate_id] = max(bm25_scores_by_id.get(candidate_id, 0.0), norm_score)
                trace['bm25_hits'] = trace.get('bm25_hits', 0) + len(top_pairs)

            try:
                if bm25_scores_by_id:
                    fetched = collection.get(ids=list(bm25_scores_by_id.keys()))
                    f_docs = fetched.get('documents', []) or []
                    f_metas = fetched.get('metadatas', []) or []
                    f_ids = fetched.get('ids', []) or list(bm25_scores_by_id.keys())
                    for candidate_id, fd, fm in zip(f_ids, f_docs, f_metas):
                        if isinstance(fd, str) and isinstance(fm, dict):
                            key = key_fn({'text': fd, 'metadata': fm})
                            candidate = ensure_candidate(key, fd, fm)
                            candidate['bm25_score'] = max(candidate.get('bm25_score', 0.0), bm25_scores_by_id.get(candidate_id, 0.0))
                            trace['candidate_count'] = trace.get('candidate_count', 0) + 1
            except Exception:
                pass

        if getattr(self, 'bm25_titles', None) and getattr(self.indexer, 'titles', None):
            title_scores_by_id: dict[str, float] = {}
            for variant in query_variants:
                try:
                    top_pairs = self.bm25_titles.query(variant, top_k=expansion_limit)
                except Exception:
                    top_pairs = []
                if not top_pairs:
                    continue
                top_score = max((score for _, score in top_pairs), default=0.0)
                for title_id, raw_score in top_pairs:
                    norm_score = self._normalize_bm25_score(raw_score, top_score)
                    if norm_score <= 0:
                        continue
                    title_scores_by_id[title_id] = max(title_scores_by_id.get(title_id, 0.0), norm_score)
                trace['title_hits'] = trace.get('title_hits', 0) + len(top_pairs)

            try:
                if title_scores_by_id:
                    title_payload = self.indexer.titles.get(ids=list(title_scores_by_id.keys()))
                    title_docs = title_payload.get('documents', []) or []
                    title_metas = title_payload.get('metadatas', []) or []
                    title_ids = title_payload.get('ids', []) or list(title_scores_by_id.keys())

                    case_title_boosts: dict = {}
                    source_ref_ids: dict[str, dict[str, float]] = {'statute': {}, 'regulation': {}}

                    for title_id, title_text, title_meta in zip(title_ids, title_docs, title_metas):
                        if not isinstance(title_meta, dict):
                            continue
                        source = title_meta.get('source')
                        if source == 'case':
                            parent_id = title_meta.get('parent_opinion_id')
                            if parent_id:
                                case_title_boosts[parent_id] = max(case_title_boosts.get(parent_id, 0.0), title_scores_by_id.get(title_id, 0.0))
                        elif source in {'statute', 'regulation'}:
                            ref_id = title_meta.get('ref_id')
                            if ref_id:
                                source_ref_ids[source][ref_id] = max(source_ref_ids[source].get(ref_id, 0.0), title_scores_by_id.get(title_id, 0.0))

                    if source_kind == 'cases' and case_title_boosts:
                        try:
                            for parent_id, title_score in case_title_boosts.items():
                                fetched = collection.get(where={'parent_opinion_id': {'$eq': parent_id}})
                                docs = fetched.get('documents', []) or []
                                metas = fetched.get('metadatas', []) or []
                                for fd, fm in zip(docs, metas):
                                    if isinstance(fd, str) and isinstance(fm, dict):
                                        key = key_fn({'text': fd, 'metadata': fm})
                                        candidate = ensure_candidate(key, fd, fm)
                                        candidate['title_score'] = max(candidate.get('title_score', 0.0), title_score)
                                        trace['candidate_count'] = trace.get('candidate_count', 0) + 1
                        except Exception:
                            pass

                    for source, ref_map in source_ref_ids.items():
                        if not ref_map:
                            continue
                        try:
                            fetched = collection.get(ids=list(ref_map.keys()))
                            docs = fetched.get('documents', []) or []
                            metas = fetched.get('metadatas', []) or []
                            ids = fetched.get('ids', []) or list(ref_map.keys())
                            for ref_id, fd, fm in zip(ids, docs, metas):
                                if isinstance(fd, str) and isinstance(fm, dict):
                                    key = key_fn({'text': fd, 'metadata': fm})
                                    candidate = ensure_candidate(key, fd, fm)
                                    candidate['title_score'] = max(candidate.get('title_score', 0.0), ref_map.get(ref_id, 0.0))
                                    trace['candidate_count'] = trace.get('candidate_count', 0) + 1
                        except Exception:
                            pass
            except Exception:
                pass

        fused: list[dict] = []
        for candidate in candidate_map.values():
            hybrid_score = self._combine_scores(
                source_kind=source_kind,
                query=query_variants[0] if query_variants else '',
                semantic_score=candidate.get('semantic_score', 0.0),
                bm25_score=candidate.get('bm25_score', 0.0),
                title_score=candidate.get('title_score', 0.0),
            )
            candidate['hybrid_score'] = hybrid_score
            fused.append(candidate)

        trace['final_candidate_count'] = len(fused)
        trace['filtered_by_distance'] = filtered_count

        return fused, trace

if __name__ == '__main__':
    r = Retriever()
    print('Retriever ready')
