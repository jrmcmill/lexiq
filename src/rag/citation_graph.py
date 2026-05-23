import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

from src.agent.citation import extract_citations_from_text
from src.config import Config
from src.observability.logger import get_logger

logger = get_logger(__name__)


def _clean(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _clean_id(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return ""
        if value.is_integer():
            return str(int(value))
    text = _clean(value)
    if text.endswith('.0') and text.replace('.0', '').isdigit():
        return text[:-2]
    return text


def _normalize_alias(value):
    return " ".join(_clean(value).replace("§", "section").split()).lower()


def _case_node_id(parent_opinion_id):
    return f"case:{_clean_id(parent_opinion_id)}"


def _statute_node_id(title_number, section_number):
    return f"statute:{_clean_id(title_number)}:{_clean_id(section_number)}"


def _regulation_node_id(cfr_title, cfr_part, cfr_section):
    return f"regulation:{_clean_id(cfr_title)}:{_clean_id(cfr_part)}:{_clean_id(cfr_section)}"


@dataclass
class CitationNode:
    node_id: str
    kind: str
    label: str
    metadata: dict = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


class CitationGraph:
    def __init__(self, persist_dir: str | None = None, filename: str | None = None):
        persist_dir = persist_dir or Config.CHROMA_PERSIST_DIR
        filename = filename or getattr(Config, "CITATION_GRAPH_FILENAME", "citation_graph.json")
        self.persist_dir = persist_dir
        self.path = os.path.join(persist_dir, filename)
        self.nodes: dict[str, CitationNode] = {}
        self.out_edges: dict[str, dict[str, dict]] = defaultdict(dict)
        self.in_edges: dict[str, dict[str, dict]] = defaultdict(dict)
        self.alias_to_node: dict[str, str] = {}

    def clear(self):
        self.nodes = {}
        self.out_edges = defaultdict(dict)
        self.in_edges = defaultdict(dict)
        self.alias_to_node = {}

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.nodes = {
                node_id: CitationNode(
                    node_id=node_id,
                    kind=node.get("kind", ""),
                    label=node.get("label", ""),
                    metadata=node.get("metadata", {}) or {},
                    chunk_ids=node.get("chunk_ids", []) or [],
                    aliases=node.get("aliases", []) or [],
                )
                for node_id, node in (payload.get("nodes", {}) or {}).items()
            }
            self.out_edges = defaultdict(dict, {
                source: {target: edge for target, edge in targets.items()}
                for source, targets in (payload.get("out_edges", {}) or {}).items()
            })
            self.in_edges = defaultdict(dict, {
                target: {source: edge for source, edge in sources.items()}
                for target, sources in (payload.get("in_edges", {}) or {}).items()
            })
            self.alias_to_node = payload.get("alias_to_node", {}) or {}
            return True
        except Exception as exc:
            logger.warning(f"Could not load citation graph: {exc}")
            self.clear()
            return False

    def save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        payload = {
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "label": node.label,
                    "metadata": node.metadata,
                    "chunk_ids": node.chunk_ids,
                    "aliases": node.aliases,
                }
                for node_id, node in self.nodes.items()
            },
            "out_edges": {source: targets for source, targets in self.out_edges.items()},
            "in_edges": {target: sources for target, sources in self.in_edges.items()},
            "alias_to_node": self.alias_to_node,
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _register_alias(self, alias: str, node_id: str):
        cleaned = _normalize_alias(alias)
        if cleaned and cleaned not in self.alias_to_node:
            self.alias_to_node[cleaned] = node_id

    def add_node(self, node: CitationNode):
        existing = self.nodes.get(node.node_id)
        if existing:
            existing.metadata.update({k: v for k, v in node.metadata.items() if v not in (None, "", [])})
            existing.chunk_ids = list(dict.fromkeys(existing.chunk_ids + node.chunk_ids))
            existing.aliases = list(dict.fromkeys(existing.aliases + node.aliases))
            node = existing
        else:
            self.nodes[node.node_id] = node

        for alias in [node.node_id, node.label, *node.aliases]:
            self._register_alias(alias, node.node_id)

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0):
        if not source_id or not target_id or source_id == target_id:
            return
        edge = {"relation": relation, "weight": float(weight)}
        current = self.out_edges[source_id].get(target_id)
        if current is None or edge["weight"] >= float(current.get("weight", 0.0)):
            self.out_edges[source_id][target_id] = edge
            self.in_edges[target_id][source_id] = edge

    def _load_cases(self, cases_parquet: str | None = None, raw_dir: str | None = None):
        cases_df = pd.DataFrame()
        raw_map: dict[str, dict] = {}

        if cases_parquet and os.path.exists(cases_parquet):
            cases_df = pd.read_parquet(cases_parquet)

        raw_dir = raw_dir or os.path.join(os.getcwd(), "data", "raw", "courtlistener")
        if os.path.exists(raw_dir):
            for fname in tqdm(sorted(os.listdir(raw_dir)), desc="Loading case raws", unit="file"):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(raw_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                except Exception:
                    continue
                raw_map[_clean(data.get("cluster_id") or data.get("id") or os.path.splitext(fname)[0])] = data

        return cases_df, raw_map

    def _case_aliases(self, raw: dict, row: pd.Series | None = None) -> list[str]:
        aliases = []
        for value in [
            raw.get("caseName"),
            raw.get("caseNameFull"),
            raw.get("neutralCite"),
            raw.get("lexisCite"),
            raw.get("court_citation_string"),
            raw.get("citation"),
        ]:
            if isinstance(value, list):
                aliases.extend([str(item) for item in value if _clean(item)])
            elif _clean(value):
                aliases.append(str(value))
        if row is not None:
            for value in [row.get("bluebook_cite"), row.get("case_name")]:
                if _clean(value):
                    aliases.append(str(value))
        return aliases

    def _statute_aliases(self, row: pd.Series) -> list[str]:
        aliases = [
            row.get("usc_citation"),
            f"{_clean(row.get('title_number'))} U.S.C. § {_clean(row.get('section_number'))}",
            f"{_clean(row.get('title_number'))} USC {_clean(row.get('section_number'))}",
        ]
        return [str(alias) for alias in aliases if _clean(alias)]

    def _regulation_aliases(self, row: pd.Series) -> list[str]:
        aliases = [
            row.get("cfr_citation"),
            f"{_clean(row.get('cfr_title'))} C.F.R. § {_clean(row.get('cfr_part'))}.{_clean(row.get('cfr_section'))}",
            f"{_clean(row.get('cfr_title'))} CFR {_clean(row.get('cfr_part'))}.{_clean(row.get('cfr_section'))}",
        ]
        return [str(alias) for alias in aliases if _clean(alias)]

    def _resolve_alias(self, alias: str) -> str | None:
        return self.alias_to_node.get(_normalize_alias(alias))

    def build_from_data(self, cases_parquet: str | None = None, statutes_parquet: str | None = None,
                        regs_parquet: str | None = None, raw_cases_dir: str | None = None):
        self.clear()

        cases_df, raw_case_map = self._load_cases(cases_parquet=cases_parquet, raw_dir=raw_cases_dir)
        statutes_df = pd.read_parquet(statutes_parquet) if statutes_parquet and os.path.exists(statutes_parquet) else pd.DataFrame()
        regs_df = pd.read_parquet(regs_parquet) if regs_parquet and os.path.exists(regs_parquet) else pd.DataFrame()

        case_groups = {}
        if not cases_df.empty and "parent_opinion_id" in cases_df.columns:
            grouped = cases_df.groupby("parent_opinion_id", dropna=False)
            for parent_id, group in tqdm(grouped, desc="Building case nodes", unit="case"):
                parent_id = _clean(parent_id)
                if not parent_id:
                    continue
                first = group.iloc[0]
                raw = raw_case_map.get(parent_id, {})
                chunk_ids = []
                for _, row in group.iterrows():
                    chunk_ids.append(f"case_{_clean_id(row.get('parent_opinion_id'))}_{_clean_id(row.get('chunk_index'))}")
                node = CitationNode(
                    node_id=_case_node_id(parent_id),
                    kind="case",
                    label=_clean(first.get("case_name") or raw.get("caseName") or raw.get("caseNameFull") or first.get("bluebook_cite") or parent_id),
                    metadata={
                        "parent_opinion_id": parent_id,
                        "case_name": _clean(first.get("case_name") or raw.get("caseName") or raw.get("caseNameFull")),
                        "court": _clean(first.get("court") or raw.get("court")),
                        "date_filed": _clean(first.get("date_filed") or raw.get("dateFiled") or raw.get("date_filed")),
                        "bluebook_cite": _clean(first.get("bluebook_cite") or raw.get("neutralCite") or raw.get("lexisCite") or raw.get("court_citation_string")),
                    },
                    chunk_ids=chunk_ids,
                    aliases=self._case_aliases(raw, first),
                )
                self.add_node(node)
                case_groups[parent_id] = {
                    "node_id": node.node_id,
                    "raw": raw,
                    "chunk_ids": chunk_ids,
                }

        if not statutes_df.empty:
            grouped = statutes_df.groupby(["title_number", "section_number"], dropna=False)
            for key, group in tqdm(grouped, desc="Building statute nodes", unit="statute"):
                title_number, section_number = key
                title_number = _clean(title_number)
                section_number = _clean(section_number)
                if not title_number or not section_number:
                    continue
                first = group.iloc[0]
                chunk_ids = []
                for _, row in group.iterrows():
                    gid = _clean(row.get("granule_id") or row.get("package_id"))
                    if gid:
                        chunk_ids.append(f"stat_{_clean_id(row.get('title_number'))}_{_clean_id(row.get('section_number'))}_{_clean_id(gid)}_{_clean_id(row.get('chunk_index'))}")
                    else:
                        chunk_ids.append(f"stat_{_clean_id(row.get('title_number'))}_{_clean_id(row.get('section_number'))}_{_clean_id(row.get('chunk_index'))}")
                node = CitationNode(
                    node_id=_statute_node_id(title_number, section_number),
                    kind="statute",
                    label=_clean(first.get("section_heading") or first.get("usc_citation") or f"{title_number} U.S.C. {section_number}"),
                    metadata={
                        "title_number": title_number,
                        "section_number": section_number,
                        "section_heading": _clean(first.get("section_heading")),
                        "usc_citation": _clean(first.get("usc_citation")),
                    },
                    chunk_ids=chunk_ids,
                    aliases=self._statute_aliases(first),
                )
                self.add_node(node)

        if not regs_df.empty:
            grouped = regs_df.groupby(["cfr_title", "cfr_part", "cfr_section"], dropna=False)
            for key, group in tqdm(grouped, desc="Building regulation nodes", unit="regulation"):
                cfr_title, cfr_part, cfr_section = key
                cfr_title = _clean(cfr_title)
                cfr_part = _clean(cfr_part)
                cfr_section = _clean(cfr_section)
                if not cfr_title or not cfr_part or not cfr_section:
                    continue
                first = group.iloc[0]
                chunk_ids = [f"reg_{_clean_id(row.get('cfr_title'))}_{_clean_id(row.get('cfr_part'))}_{_clean_id(row.get('cfr_section'))}_{_clean_id(row.get('chunk_index'))}" for _, row in group.iterrows()]
                node = CitationNode(
                    node_id=_regulation_node_id(cfr_title, cfr_part, cfr_section),
                    kind="regulation",
                    label=_clean(first.get("section_heading") or first.get("cfr_citation") or f"{cfr_title} C.F.R. {cfr_part}.{cfr_section}"),
                    metadata={
                        "cfr_title": cfr_title,
                        "cfr_part": cfr_part,
                        "cfr_section": cfr_section,
                        "section_heading": _clean(first.get("section_heading")),
                        "cfr_citation": _clean(first.get("cfr_citation")),
                    },
                    chunk_ids=chunk_ids,
                    aliases=self._regulation_aliases(first),
                )
                self.add_node(node)

        pending_case_edges = []
        for parent_id, payload in case_groups.items():
            raw = payload["raw"] or {}
            cited_ids = []
            opinions = raw.get("opinions") or []
            if opinions and isinstance(opinions, list):
                first_op = opinions[0] or {}
                cited_ids = first_op.get("cites") or []
            for target_id in cited_ids:
                target_node = self.nodes.get(_case_node_id(target_id))
                if target_node:
                    self.add_edge(payload["node_id"], target_node.node_id, relation="case_cites", weight=1.0)
                else:
                    pending_case_edges.append((payload["node_id"], _case_node_id(target_id)))

        text_sources = []
        if not cases_df.empty:
            text_sources.extend(("case", row) for _, row in cases_df.iterrows())
        if not statutes_df.empty:
            text_sources.extend(("statute", row) for _, row in statutes_df.iterrows())
        if not regs_df.empty:
            text_sources.extend(("regulation", row) for _, row in regs_df.iterrows())

        for source_kind, row in tqdm(text_sources, desc="Linking text citations", unit="chunk"):
            if source_kind == "case":
                source_id = _case_node_id(row.get("parent_opinion_id"))
                source_text = row.get("text") or ""
            elif source_kind == "statute":
                source_id = _statute_node_id(row.get("title_number"), row.get("section_number"))
                source_text = row.get("section_text") or ""
            else:
                source_id = _regulation_node_id(row.get("cfr_title"), row.get("cfr_part"), row.get("cfr_section"))
                source_text = row.get("section_text") or ""

            if not source_text:
                continue

            for citation in extract_citations_from_text(str(source_text)):
                target_id = self._resolve_alias(citation)
                if target_id:
                    self.add_edge(source_id, target_id, relation="text_cites", weight=0.7)

        for source_id, target_id in pending_case_edges:
            if target_id in self.nodes:
                self.add_edge(source_id, target_id, relation="case_cites", weight=1.0)

        self.save()
        return {
            "nodes": len(self.nodes),
            "out_edges": sum(len(targets) for targets in self.out_edges.values()),
            "path": self.path,
        }

    def node_for_result(self, result: dict, source_kind: str) -> str | None:
        meta = (result or {}).get("metadata", {}) or {}
        if source_kind == "cases":
            parent_id = _clean_id(meta.get("parent_opinion_id") or result.get("source_id"))
            return _case_node_id(parent_id) if parent_id else None
        if source_kind == "statutes":
            title_number = _clean_id(meta.get("title_number"))
            section_number = _clean_id(meta.get("section_number"))
            if title_number and section_number:
                return _statute_node_id(title_number, section_number)
            cite = _clean(meta.get("usc_citation") or result.get("source_id"))
            return self._resolve_alias(cite) if cite else None
        if source_kind == "regs":
            cfr_title = _clean_id(meta.get("cfr_title"))
            cfr_part = _clean_id(meta.get("cfr_part"))
            cfr_section = _clean_id(meta.get("cfr_section"))
            if cfr_title and cfr_part and cfr_section:
                return _regulation_node_id(cfr_title, cfr_part, cfr_section)
            cite = _clean(meta.get("cfr_citation") or result.get("source_id"))
            return self._resolve_alias(cite) if cite else None
        return None

    def fetch_spec(self, node_id: str) -> dict | None:
        node = self.nodes.get(node_id)
        if not node:
            return None
        return {
            "kind": node.kind,
            "label": node.label,
            "metadata": node.metadata,
            "chunk_ids": node.chunk_ids,
            "node_id": node.node_id,
        }

    def expand(self, seed_node_ids: list[str], max_hops: int = 1, max_nodes: int = 12) -> list[dict]:
        if not seed_node_ids:
            return []

        queue = deque([(seed_id, 0) for seed_id in seed_node_ids if seed_id in self.nodes])
        seen = set(seed_node_ids)
        output: dict[str, dict] = {}

        while queue and len(output) < max_nodes:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue

            neighbors = []
            neighbors.extend((target_id, edge, "out") for target_id, edge in self.out_edges.get(current, {}).items())
            neighbors.extend((source_id, edge, "in") for source_id, edge in self.in_edges.get(current, {}).items())

            for neighbor_id, edge, direction in neighbors:
                if neighbor_id in seen or neighbor_id not in self.nodes:
                    continue
                seen.add(neighbor_id)
                next_depth = depth + 1
                base_score = float(edge.get("weight", 1.0))
                score = base_score / float(next_depth + 1)
                output[neighbor_id] = {
                    "node_id": neighbor_id,
                    "kind": self.nodes[neighbor_id].kind,
                    "label": self.nodes[neighbor_id].label,
                    "metadata": self.nodes[neighbor_id].metadata,
                    "chunk_ids": self.nodes[neighbor_id].chunk_ids,
                    "distance": next_depth,
                    "score": score,
                    "relation": edge.get("relation", "linked"),
                    "direction": direction,
                    "seed_node_id": current,
                }
                if next_depth < max_hops:
                    queue.append((neighbor_id, next_depth))

                if len(output) >= max_nodes:
                    break

        return sorted(output.values(), key=lambda item: (item.get("distance", 99), item.get("score", 0.0)), reverse=False)


def build_citation_graph(cases_parquet: str | None = None, statutes_parquet: str | None = None,
                         regs_parquet: str | None = None, raw_cases_dir: str | None = None,
                         persist_dir: str | None = None) -> dict:
    graph = CitationGraph(persist_dir=persist_dir)
    return graph.build_from_data(
        cases_parquet=cases_parquet,
        statutes_parquet=statutes_parquet,
        regs_parquet=regs_parquet,
        raw_cases_dir=raw_cases_dir,
    )
