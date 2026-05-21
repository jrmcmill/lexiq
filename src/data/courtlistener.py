import requests
import time
import json
import os
import io
from typing import List, Optional
from tqdm import tqdm
from src.config import Config
from src.observability.logger import get_logger
from lxml import etree
import pdfplumber

logger = get_logger(__name__)

BASE = "https://www.courtlistener.com/api/rest/v4"

class CourtListenerClient:
    def __init__(self):
        self.api_key = Config.COURTLISTENER_API_KEY
        self.page_size = Config.COURTLISTENER_PAGE_SIZE
        self.raw_dir = os.path.join(os.getcwd(), "data", "raw", "courtlistener")
        os.makedirs(self.raw_dir, exist_ok=True)

    def _extract_pdf_text(self, pdf_bytes: bytes) -> str:
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts).strip()

    def _download_and_extract_pdf_text(self, url: str) -> str:
        if not url:
            return ""
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return self._extract_pdf_text(resp.content)
        except Exception as exc:
            logger.warning(f"Could not extract PDF text from {url}: {exc}")
            return ""

    def _request_url(self, url: str, params=None, headers=None):
        headers = headers or {}
        if self.api_key:
            headers["Authorization"] = f"Token {self.api_key}"
        attempt = 0
        while attempt < 5:
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                if resp.status_code in (429, 503):
                    backoff = 2 ** attempt
                    logger.info(f"Rate limited, backing off {backoff}s")
                    time.sleep(backoff)
                    attempt += 1
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as e:
                logger.error(str(e))
                raise
            except requests.exceptions.RequestException as e:
                logger.error(str(e))
                raise
        raise RuntimeError("Max retries exceeded")

    def _request(self, path, params=None, headers=None):
        return self._request_url(f"{BASE}{path}", params=params, headers=headers)

    def fetch_opinions(self, query: str, courts: Optional[List[str]] = None,
                       date_after: Optional[str] = None, date_before: Optional[str] = None,
                       precedential_status: Optional[str] = None, max_pages: int = 1,
                       skip_existing: bool = False, stop_after_seen: Optional[int] = None,
                       stop_after_count: Optional[int] = None) -> List[dict]:
        results = []
        seen = 0
        params = {}
        if query:
            params["q"] = query
        pbar = tqdm(total=max_pages, desc="CourtListener", unit="page")
        next_url = f"{BASE}/search/"
        page = 1
        while page <= max_pages and next_url:
            request_params = params.copy() if page == 1 else None
            if request_params is not None:
                request_params["page_size"] = self.page_size
            try:
                data = self._request_url(next_url, params=request_params)
            except Exception as e:
                logger.warning(f"Error fetching opinions: {e}")
                pbar.close()
                break
            for item in tqdm(data.get("results", []), desc="  Processing results", leave=False):
                case_name = item.get("caseName")
                cluster_id = item.get("cluster_id")
                date_filed = item.get("dateFiled")
                docket_number = item.get("docketNumber")
                court_name = item.get("court")
                citations = []
                opinions = item.get("opinions", [])
                download_url = ""
                extracted_from_pdf = False
                if opinions:
                    first_op = opinions[0]
                    for op in opinions:
                        candidate_url = op.get("download_url", "") or ""
                        if candidate_url:
                            download_url = candidate_url
                            break
                    plain_text = self._download_and_extract_pdf_text(download_url)
                    if plain_text:
                        extracted_from_pdf = True
                    if not plain_text:
                        plain_text = first_op.get("plain_text", "") or first_op.get("snippet", "")
                else:
                    plain_text = ""
                if not plain_text or not plain_text.strip():
                    continue
                opinion_id = cluster_id
                raw_path = os.path.join(self.raw_dir, f"{opinion_id}.json")
                if skip_existing and os.path.exists(raw_path):
                    # skip saving/returning existing items
                    logger.debug(f"Skipping existing opinion {opinion_id}")
                    seen += 1
                    if stop_after_seen and seen >= stop_after_seen:
                        logger.info("Encountered consecutive existing items; stopping incremental fetch")
                        return results
                    continue
                # write new raw file
                item_to_save = dict(item)
                item_to_save["plain_text"] = plain_text
                item_to_save["download_url"] = download_url
                item_to_save["text_source"] = "pdf" if extracted_from_pdf else "snippet"
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(item_to_save, f)
                # reset seen counter when we save a new item
                seen = 0
                results.append({
                    "id": opinion_id,
                    "cluster_id": cluster_id,
                    "author_str": "",
                    "court": court_name or "",
                    "date_filed": date_filed,
                    "precedential_status": "",
                    "plain_text": plain_text,
                    "html": "",
                    "download_url": download_url,
                    "citations": citations,
                    "case_name": case_name,
                    "docket_number": docket_number,
                    "judges": "",
                })
                # stop if we reached the requested total count
                if stop_after_count and len(results) >= stop_after_count:
                    logger.info(f"Reached stop_after_count={stop_after_count}; stopping fetch")
                    pbar.close()
                    return results
            pbar.update(1)
            next_url = data.get("next")
            page += 1
        pbar.close()
        logger.info(f"Fetched {len(results)} opinion(s)")
        return results

    def fetch_opinion_by_id(self, opinion_id: int) -> dict:
        return self._request(f"/opinions/{opinion_id}/")

    def list_courts(self) -> list:
        data = self._request("/courts/")
        return data.get("results", [])

if __name__ == "__main__":
    client = CourtListenerClient()
    if client.api_key:
        res = client.fetch_opinions(query="constitutional", max_pages=1)
        print(f"Fetched {len(res)} opinions")
    else:
        print("COURTLISTENER_API_KEY not set in .env. Skipping fetch. Register at https://www.courtlistener.com/help/api/")
