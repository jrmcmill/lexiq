import os
import re
from datetime import date as _date
from typing import List, Optional
from tqdm import tqdm
import requests
from lxml import etree

from src.observability.logger import get_logger

logger = get_logger(__name__)

BASE = "https://www.ecfr.gov/api"

class ECFRClient:
    def __init__(self):
        self.raw_dir = os.path.join(os.getcwd(), "data", "raw", "ecfr")
        os.makedirs(self.raw_dir, exist_ok=True)

    def list_titles(self) -> List[dict]:
        url = f"{BASE}/versioner/v1/titles.json"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("titles", [])

    def fetch_title_xml(self, title_number: int, date: Optional[str] = None) -> str:
        if date is None:
            date = _date.today().isoformat()
        url = f"{BASE}/versioner/v1/full/{date}/title-{title_number}.xml"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        path = os.path.join(self.raw_dir, f"title_{title_number}_{date}.xml")
        # If the exact title/date file already exists, skip re-downloading
        if os.path.exists(path):
            logger.debug(f"ECFR title XML already exists, skipping: {path}")
            return path
        with open(path, "wb") as f:
            f.write(resp.content)
        return path

    def search(self, query: str, per_page: int = 20) -> List[dict]:
        url = f"{BASE}/search/v1/results"
        params = {"query": query, "per_page": per_page}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    def parse_title_xml(self, xml_path: str) -> List[dict]:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        title_number = self._extract_title_from_filename(xml_path)
        results = []

        for section in root.xpath('//*[local-name()="DIV8" and @TYPE="SECTION"]'):
            section_number = section.attrib.get('N', '')
            heading = self._first_text(section, [
                './*[local-name()="HEAD"][1]',
                './/*[local-name()="HEAD"][1]',
            ])
            part = self._first_ancestor_attr(section, 'DIV5', 'N')

            text = " ".join(text.strip() for text in section.xpath('.//text()') if text and text.strip())
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue

            cfr_citation = f"{title_number} C.F.R. § {section_number}" if title_number and section_number else ""
            for idx, chunk in enumerate(self._chunk_text(text)):
                results.append({
                    "cfr_title": title_number,
                    "cfr_part": part,
                    "cfr_section": section_number,
                    "section_heading": heading,
                    "section_text": chunk,
                    "cfr_citation": cfr_citation,
                    "chunk_index": idx,
                })
        return results

    def _first_text(self, node, xpath_expressions: List[str]) -> str:
        for expression in xpath_expressions:
            values = node.xpath(expression)
            if values:
                value = values[0]
                if hasattr(value, 'text'):
                    value = value.text
                if value:
                    value = str(value).strip()
                    if value:
                        return value
        return ""

    def _first_ancestor_text(self, node, local_name: str) -> str:
        values = node.xpath(f'ancestor::*[local-name()="{local_name}"][1]')
        if not values:
            return ""
        ancestor = values[0]
        return self._first_text(ancestor, [
            './/*[local-name()="title"][1]',
            './/*[local-name()="partNum"][1]',
        ])

    def _first_ancestor_attr(self, node, local_name: str, attr_name: str) -> str:
        values = node.xpath(f'ancestor::*[local-name()="{local_name}"][1]')
        if not values:
            return ""
        ancestor = values[0]
        return str(ancestor.attrib.get(attr_name, '')).strip()

    def _chunk_text(self, text: str, chunk_tokens: int = 512, overlap: int = 100) -> List[str]:
        words = text.split()
        token_per_word = 0.75
        chunk_words = max(1, int(chunk_tokens * token_per_word))
        overlap_words = max(0, int(overlap * token_per_word))
        chunks = []
        index = 0
        while index < len(words):
            chunk = words[index:index + chunk_words]
            if not chunk:
                break
            chunks.append(" ".join(chunk))
            index += max(1, chunk_words - overlap_words)
        return chunks

    def _extract_title_from_filename(self, xml_path: str) -> str:
        filename = os.path.basename(xml_path)
        match = re.search(r"title_(\d+)_", filename)
        return match.group(1) if match else ""

if __name__ == "__main__":
    client = ECFRClient()
    titles = client.list_titles()
    if not titles:
        print("No CFR data fetched")
    else:
        saved = 0
        for title in tqdm(titles, desc="eCFR Titles", unit="title"):
            if title.get("reserved"):
                continue
            title_number = title.get("number")
            issue_date = title.get("latest_issue_date") or title.get("up_to_date_as_of")
            if not title_number or not issue_date:
                continue
            try:
                client.fetch_title_xml(title_number, issue_date)
                saved += 1
            except Exception as exc:
                logger.warning(f"Could not fetch title {title_number}: {exc}")
        print(f"Fetched {saved} CFR titles")
