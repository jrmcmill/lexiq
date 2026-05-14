from src.agent.citation import format_bluebook_case, format_bluebook_statute, format_bluebook_regulation, extract_citations_from_text

def test_format_bluebook_case_with_reporter():
    s = format_bluebook_case('Brown v. Board', 'U.S.', '347', '483', 'U.S.', '1954')
    assert 'Brown' in s

def test_format_bluebook_case_without_reporter():
    s = format_bluebook_case('United States v. Jones', None, None, None, 'U.S.', '2012', docket_number='10-1259')
    assert 'No.' in s

def test_format_statute():
    s = format_bluebook_statute('42', '1983', 2018)
    assert 'U.S.C.' in s

def test_format_regulation():
    s = format_bluebook_regulation('12', '226', '1', 2023)
    assert 'C.F.R.' in s

def test_extract_citations():
    t = 'See 347 U.S. 483 and 42 U.S.C. § 1983 and 12 C.F.R. § 226.1'
    c = extract_citations_from_text(t)
    assert any('U.S.C.' in x for x in c)
    assert any('C.F.R.' in x for x in c)
