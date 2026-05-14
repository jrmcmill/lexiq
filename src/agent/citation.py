import re
from datetime import datetime

CASE_RE = re.compile(r"(\d+)\s+([A-Z][A-Za-z\.]+)\s+(\d+)")
USC_RE = re.compile(r"(\d+)\s+U\.S\.C\.?\s+§\s+([\d\w]+)")
CFR_RE = re.compile(r"(\d+)\s+C\.F\.R\.\.?\s+§\s+([\d\.\w]+)")


def format_bluebook_case(case_name: str, reporter: str | None, volume: str | None,
                       page: str | None, court: str = "", year: str = "",
                       docket_number: str | None = None) -> str:
    name = f"*{case_name}*" if case_name else ""
    if reporter and volume and page:
        return f"{name}, {volume} {reporter} {page} ({year})."
    if docket_number:
        return f"{name}, No. {docket_number} ({court} {year})."
    return f"{name}, ({court} {year})."


def format_bluebook_statute(title: str, section: str, year: int | None = None) -> str:
    y = f" ({year})." if year else ""
    return f"{title} U.S.C. § {section}{y}"


def format_bluebook_regulation(cfr_title: str, cfr_part: str, cfr_section: str, year: int | None = None) -> str:
    y = f" ({year})." if year else ""
    return f"{cfr_title} C.F.R. § {cfr_part}.{cfr_section}{y}"


def extract_citations_from_text(text: str) -> list[str]:
    out = []
    out += [m.group(0) for m in CASE_RE.finditer(text)]
    out += [m.group(0) for m in USC_RE.finditer(text)]
    out += [m.group(0) for m in CFR_RE.finditer(text)]
    return out


def parse_citation(raw: str) -> dict | None:
    if USC_RE.search(raw):
        m = USC_RE.search(raw)
        return {"type": "statute", "title": m.group(1), "section": m.group(2)}
    if CFR_RE.search(raw):
        m = CFR_RE.search(raw)
        return {"type": "regulation", "title": m.group(1), "section": m.group(2)}
    if CASE_RE.search(raw):
        m = CASE_RE.search(raw)
        return {"type": "case", "volume": m.group(1), "reporter": m.group(2), "page": m.group(3)}
    return None

if __name__ == '__main__':
    print(format_bluebook_case('Brown v. Board of Education', 'U.S.', '347', '483', 'U.S.', '1954'))
