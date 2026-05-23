import json

import pandas as pd

from src.rag.citation_graph import CitationGraph


def test_citation_graph_builds_case_and_statute_links(tmp_path):
    cases = pd.DataFrame([
        {
            'parent_opinion_id': 1,
            'chunk_index': 0,
            'case_name': 'Alpha v. Beta',
            'court': 'D.C. Cir.',
            'date_filed': '2025-01-01',
            'bluebook_cite': 'Alpha v. Beta',
            'text': 'See 12 U.S.C. § 5 and 42 C.F.R. § 10.2.',
        },
        {
            'parent_opinion_id': 2,
            'chunk_index': 0,
            'case_name': 'Gamma v. Delta',
            'court': 'D.C. Cir.',
            'date_filed': '2025-01-01',
            'bluebook_cite': 'Gamma v. Delta',
            'text': 'Alpha v. Beta resolves the issue.',
        },
    ])
    statutes = pd.DataFrame([
        {
            'title_number': 12,
            'section_number': 5,
            'section_heading': 'Banking rule',
            'section_text': 'This section references 12 U.S.C. § 5.',
            'usc_citation': '12 U.S.C. § 5',
            'package_id': 'p1',
            'granule_id': 'g1',
            'chunk_index': 0,
        }
    ])
    regs = pd.DataFrame([
        {
            'cfr_title': 42,
            'cfr_part': 10,
            'cfr_section': 2,
            'section_heading': 'Reg rule',
            'section_text': 'See 42 C.F.R. § 10.2.',
            'cfr_citation': '42 C.F.R. § 10.2',
            'chunk_index': 0,
        }
    ])

    cases_path = tmp_path / 'cases.parquet'
    statutes_path = tmp_path / 'statutes.parquet'
    regs_path = tmp_path / 'regs.parquet'
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()

    cases.to_parquet(cases_path)
    statutes.to_parquet(statutes_path)
    regs.to_parquet(regs_path)

    raw_case = {
        'cluster_id': 1,
        'caseName': 'Alpha v. Beta',
        'caseNameFull': 'Alpha v. Beta',
        'neutralCite': 'Alpha v. Beta',
        'lexisCite': '',
        'court_citation_string': 'D.C. Cir.',
        'opinions': [{'cites': [2]}],
    }
    with open(raw_dir / '1.json', 'w', encoding='utf-8') as handle:
        json.dump(raw_case, handle)

    graph = CitationGraph(persist_dir=str(tmp_path))
    stats = graph.build_from_data(
        cases_parquet=str(cases_path),
        statutes_parquet=str(statutes_path),
        regs_parquet=str(regs_path),
        raw_cases_dir=str(raw_dir),
    )

    assert stats['nodes'] >= 4
    assert graph.node_for_result({'metadata': {'parent_opinion_id': 1}}, 'cases') == 'case:1'
    assert graph.node_for_result({'metadata': {'title_number': 12, 'section_number': 5}}, 'statutes') == 'statute:12:5'
    neighbors = graph.expand(['case:1'], max_hops=1, max_nodes=10)
    assert any(item['node_id'] in {'case:2', 'statute:12:5', 'regulation:42:10:2'} for item in neighbors)
