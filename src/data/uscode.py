import json
import os
import re
from typing import List, Optional
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

from src.config import Config
from src.observability.logger import get_logger

logger = get_logger(__name__)


class USCodeClient:
    def __init__(self):
        self.api_key = Config.GOVINFO_API_KEY
        self.raw_dir = os.path.join(os.getcwd(), "data", "raw", "uscode")
        self.page_size = 50
        self.max_pages = 5
        os.makedirs(self.raw_dir, exist_ok=True)

    def _request(self, method: str, url: str, **kwargs):
        params = kwargs.pop("params", {}) or {}
        if self.api_key:
            params["api_key"] = self.api_key
        response = requests.request(method, url, params=params, timeout=kwargs.pop("timeout", 30), **kwargs)
        response.raise_for_status()
        return response

    def fetch_titles(self) -> List[dict]:
        return self.fetch_sections()

    def fetch_sections(self, query: str = "collection:uscode", max_pages: Optional[int] = None,
                       page_size: Optional[int] = None, sort_by: str = "DATE") -> List[dict]:
        if not self.api_key:
            logger.warning("GOVINFO_API_KEY not set. Skipping USC fetch.")
            return []

        page_size = page_size or self.page_size
        max_pages = max_pages or self.max_pages
        results: List[dict] = []
        offset_mark = "*"
        page = 0
        effective_sort = sort_by
        pbar = tqdm(total=max_pages, desc="US Code", unit="page")

        while page < max_pages:
            payload = {
                "query": query,
                "pageSize": page_size,
                "offsetMark": offset_mark,
                "sortBy": effective_sort,
            }
            try:
                response = self._request("POST", "https://api.govinfo.gov/search", json=payload, timeout=30)
                data = response.json()
            except Exception as exc:
                if effective_sort != "RELEVANCE":
                    logger.warning(f"GovInfo sortBy={effective_sort} failed ({exc}); retrying with RELEVANCE")
                    effective_sort = "RELEVANCE"
                    continue
                logger.warning(f"GovInfo search error: {exc}")
                break

            items = data.get("results", []) or []
            if not items:
                break

            for item in tqdm(items, desc="  Processing granules", leave=False):
                html = self._fetch_granule_html(item)
                saved = self._save_granule(item, html)
                results.append(saved)

            next_mark = data.get("offsetMark")
            if not next_mark or next_mark == offset_mark:
                break
            offset_mark = next_mark
            page += 1
            pbar.update(1)
        pbar.close()
        logger.info(f"Fetched {len(results)} USC granules")
        return results

    def _fetch_granule_html(self, item: dict) -> str:
        download = item.get("download") or {}
        txt_link = download.get("txtLink")
        if not txt_link:
            return ""
        try:
            response = self._request("GET", txt_link, timeout=60)
            return response.text
        except Exception as exc:
            logger.warning(f"Could not fetch USC granule HTML for {item.get('granuleId')}: {exc}")
            return ""

    def _save_granule(self, item: dict, html: str) -> dict:
        record = dict(item)
        record["html"] = html
        granule_id = item.get("granuleId") or item.get("packageId") or "unknown"
        path = os.path.join(self.raw_dir, f"{granule_id}.json")
        # If we already have this granule saved, skip rewriting to preserve original
        if os.path.exists(path):
            logger.debug(f"Granule already exists, skipping save: {path}")
            record["raw_path"] = path
            return record
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle)
            record["raw_path"] = path
            return record
        except Exception as exc:
            logger.warning(f"Failed to save granule {granule_id}: {exc}")
            return record

    def parse_saved_granule(self, json_path: str) -> List[dict]:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        html = data.get("html") or ""
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        title_number = self._extract_title_number(data.get("packageId") or json_path)
        section_number = self._extract_section_number(data.get("granuleId") or json_path)
        section_heading = data.get("title") or self._extract_section_heading(soup)
        usc_citation = f"{title_number} U.S.C. § {section_number}" if title_number and section_number else ""

        records = []
        for idx, chunk in enumerate(self._chunk_text(text)):
            records.append({
                "title_number": title_number,
                "section_number": section_number,
                "section_heading": section_heading,
                "section_text": chunk,
                "usc_citation": usc_citation,
                "package_id": data.get("packageId"),
                "granule_id": data.get("granuleId"),
                "date_issued": data.get("dateIssued"),
                "chunk_index": idx,
            })
        return records

    def _extract_title_number(self, source: str) -> str:
        match = re.search(r"title(\d+)", source or "", re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_section_number(self, source: str) -> str:
        match = re.search(r"sec([A-Za-z0-9]+)", source or "", re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"front", source or "", re.IGNORECASE)
        return "front" if match else ""

    def _extract_section_heading(self, soup: BeautifulSoup) -> str:
        title_tag = soup.find("title")
        if title_tag and title_tag.get_text(strip=True):
            return title_tag.get_text(" ", strip=True)
        heading = soup.find(["h1", "h2", "h3", "h4"])
        return heading.get_text(" ", strip=True) if heading else ""

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


if __name__ == "__main__":
    client = USCodeClient()
    if not client.api_key:
        print("GOVINFO_API_KEY not set in .env. Skipping fetch. Register at https://www.govinfo.gov/api/")
    else:
        try:
            granules = client.fetch_sections()
            print(f"Fetched {len(granules)} USC granules")
        except Exception as exc:
            print(f"GovInfo API error (server may be down): {exc}. Skipping USC fetch.")
