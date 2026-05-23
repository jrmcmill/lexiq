import os
import json
import re
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import pdfplumber
from src.observability.logger import get_logger
from src.agent.citation import format_bluebook_case, extract_citations_from_text
from src.data.uscode import USCodeClient
from src.data.ecfr import ECFRClient

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self):
        self.raw_dir = os.path.join(os.getcwd(), "data", "raw", "courtlistener")
        self.textbooks_dir = os.path.join(os.getcwd(), "data", "raw", "textbooks")
        self.processed_dir = os.path.join(os.getcwd(), "data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

    def _strip_html(self, html):
        return BeautifulSoup(html or "", "html.parser").get_text()

    def _chunk_text(self, text, chunk_tokens=512, overlap=100):
        words = text.split()
        token_per_word = 0.75
        chunk_words = int(chunk_tokens * token_per_word)
        overlap_words = int(overlap * token_per_word)
        chunks = []
        i = 0
        idx = 0
        while i < len(words):
            part = words[i:i+chunk_words]
            chunks.append((idx, " ".join(part)))
            idx += 1
            i += max(1, chunk_words-overlap_words)
        return chunks

    def _chunk_page_text(self, text: str, chunk_tokens: int = 512, overlap: int = 100):
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return []
        return self._chunk_text(cleaned, chunk_tokens=chunk_tokens, overlap=overlap)

    def _normalize_book_value(self, value):
        text = str(value or "").strip()
        if not text or text.lower() == 'nan':
            return ''
        return text

    def _infer_textbook_heading(self, lines, fallback: str = ''):
        heading = ''
        for line in lines[:8]:
            cleaned = re.sub(r"\s+", " ", (line or "")).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered.startswith('chapter '):
                heading = cleaned
                break
            if len(cleaned) <= 120 and (
                cleaned.isupper()
                or re.match(r"^(chapter|part|section|unit|lesson|module)\b", lowered)
                or re.match(r"^\d+(\.\d+)*\s+", cleaned)
            ):
                heading = cleaned
                break
        return heading or fallback

    def _parse_textbook_pdf(self, path: str, filename: str):
        records = []
        try:
            with pdfplumber.open(path) as pdf:
                meta = pdf.metadata or {}
                pdf_title = self._normalize_book_value(meta.get('Title')) or os.path.splitext(filename)[0]
                pdf_author = self._normalize_book_value(meta.get('Author'))
                pdf_subject = self._normalize_book_value(meta.get('Subject'))
                pdf_creator = self._normalize_book_value(meta.get('Creator'))
                pdf_producer = self._normalize_book_value(meta.get('Producer'))
                current_section = pdf_title

                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ''
                    if not page_text.strip():
                        continue
                    lines = [line for line in page_text.splitlines() if line and line.strip()]
                    section_heading = self._infer_textbook_heading(lines, fallback=current_section or pdf_title)
                    if section_heading:
                        current_section = section_heading

                    chunks = self._chunk_page_text(page_text)
                    for chunk_index, chunk_text in chunks:
                        if not chunk_text.strip():
                            continue
                        records.append({
                            'textbook_id': os.path.splitext(filename)[0],
                            'source_filename': filename,
                            'book_title': pdf_title,
                            'book_author': pdf_author,
                            'book_subject': pdf_subject,
                            'book_creator': pdf_creator,
                            'book_producer': pdf_producer,
                            'page_number': page_number,
                            'page_start': page_number,
                            'page_end': page_number,
                            'section_heading': section_heading or pdf_title,
                            'chapter': section_heading or pdf_title,
                            'chunk_index': chunk_index,
                            'text': chunk_text,
                            'word_count': len(chunk_text.split()),
                        })
        except Exception as exc:
            logger.warning(f"Error parsing textbook PDF {filename}: {exc}")
        return records

    def clean_opinions(self, raw_dir: str = None):
        raw_dir = raw_dir or self.raw_dir
        if not os.path.exists(raw_dir):
            logger.warning(f"Raw directory {raw_dir} does not exist")
            return pd.DataFrame()
        records = []
        fnames = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
        for fname in tqdm(fnames, desc="Processing Opinions", unit="file"):
            path = os.path.join(raw_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading {fname}: {e}")
                continue
            
            # Prefer extracted full text persisted during ingestion.
            text = data.get('plain_text') or data.get('extracted_text')
            if not text:
                text = self._strip_html(data.get('html'))
            # Try nested opinions structure (from /search/ endpoint)
            if not text:
                opinions = data.get('opinions', [])
                if opinions and isinstance(opinions, list):
                    first_op = opinions[0]
                    text = first_op.get('snippet') or self._strip_html(first_op.get('plain_text', ''))
            
            if not text or not str(text).strip():
                continue
            # remove boilerplate headers (simple regex)
            text = re.sub(r"^\s*IN THE.*?\n", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            chunks = self._chunk_text(text)
            citations = extract_citations_from_text(text)
            
            # Extract fields with fallbacks for both old and new structures
            case_name = data.get('case_name') or data.get('caseName')
            court = data.get('court')
            date_filed = data.get('date_filed') or data.get('dateFiled')
            docket_number = data.get('docket_number') or data.get('docketNumber')
            opinion_id = data.get('id') or data.get('cluster_id')
            
            bluebook = format_bluebook_case(case_name or '', None, None, None, court or '', date_filed or '')
            for idx, chunk in chunks:
                records.append({
                    'parent_opinion_id': opinion_id,
                    'chunk_index': idx,
                    'case_name': case_name,
                    'court': court,
                    'date_filed': date_filed,
                    'citations': citations,
                    'docket_number': docket_number,
                    'bluebook_cite': bluebook,
                    'text': chunk,
                    'word_count': len(chunk.split()),
                    'has_citations': len(citations) > 0,
                })
        if records:
            df = pd.DataFrame.from_records(records)
            outpath = os.path.join(self.processed_dir, 'cases.parquet')
            df.to_parquet(outpath)
            logger.info(f"Wrote {len(df)} case chunks to {outpath}")
        else:
            df = pd.DataFrame()
            logger.info("No opinion data to process")
            outpath = os.path.join(self.processed_dir, 'cases.parquet')
            df.to_parquet(outpath)
        return df

    def clean_statutes(self, raw_dir: str = None):
        raw_dir = raw_dir or os.path.join(os.getcwd(), 'data', 'raw', 'uscode')
        records = []
        client = USCodeClient()
        if not os.path.exists(raw_dir):
            logger.warning(f"Raw directory {raw_dir} does not exist")
        else:
            fnames = [f for f in os.listdir(raw_dir) if f.endswith('.json')]
            for fname in tqdm(fnames, desc="Processing Statutes", unit="file"):
                path = os.path.join(raw_dir, fname)
                try:
                    records.extend(client.parse_saved_granule(path))
                except Exception as exc:
                    logger.error(f"Error parsing USC file {fname}: {exc}")
        outpath = os.path.join(self.processed_dir, 'statutes.parquet')
        df = pd.DataFrame.from_records(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'title_number', 'section_number', 'section_heading', 'section_text',
                'usc_citation', 'package_id', 'granule_id', 'date_issued', 'chunk_index'
            ])
        df.to_parquet(outpath)
        if records:
            logger.info(f"Wrote {len(df)} statute chunks to {outpath}")
        else:
            logger.info("No statute data to process")
        return df

    def clean_regulations(self, raw_dir: str = None, max_sections_per_title: int = 100):
        raw_dir = raw_dir or os.path.join(os.getcwd(), 'data', 'raw', 'ecfr')
        records = []
        client = ECFRClient()
        if not os.path.exists(raw_dir):
            logger.warning(f"Raw directory {raw_dir} does not exist")
        else:
            fnames = [f for f in os.listdir(raw_dir) if f.endswith('.xml')]
            for fname in tqdm(fnames, desc="Processing Regulations", unit="file"):
                path = os.path.join(raw_dir, fname)
                try:
                    records.extend(client.parse_title_xml(path)[:max_sections_per_title])
                except Exception as exc:
                    logger.error(f"Error parsing CFR file {fname}: {exc}")
        outpath = os.path.join(self.processed_dir, 'regulations.parquet')
        df = pd.DataFrame.from_records(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'cfr_title', 'cfr_part', 'cfr_section', 'section_heading',
                'section_text', 'cfr_citation', 'chunk_index'
            ])
        df.to_parquet(outpath)
        if records:
            logger.info(f"Wrote {len(df)} regulation chunks to {outpath}")
        else:
            logger.info("No regulation data to process")
        return df

    def clean_textbooks(self, raw_dir: str = None):
        raw_dir = raw_dir or self.textbooks_dir
        records = []
        if not os.path.exists(raw_dir):
            logger.warning(f"Raw directory {raw_dir} does not exist")
        else:
            fnames = [f for f in os.listdir(raw_dir) if f.lower().endswith('.pdf')]
            for fname in tqdm(fnames, desc="Processing Textbooks", unit="file"):
                path = os.path.join(raw_dir, fname)
                records.extend(self._parse_textbook_pdf(path, fname))

        outpath = os.path.join(self.processed_dir, 'textbooks.parquet')
        df = pd.DataFrame.from_records(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'textbook_id', 'source_filename', 'book_title', 'book_author', 'book_subject',
                'book_creator', 'book_producer', 'page_number', 'page_start', 'page_end',
                'section_heading', 'chapter', 'chunk_index', 'text', 'word_count'
            ])
        df.to_parquet(outpath)
        if records:
            logger.info(f"Wrote {len(df)} textbook chunks to {outpath}")
        else:
            logger.info("No textbook data to process")
        return df

if __name__ == '__main__':
    p = Preprocessor()
    try:
        cases = p.clean_opinions()
        statutes = p.clean_statutes()
        regulations = p.clean_regulations()
        textbooks = p.clean_textbooks()
        print('Processed', len(cases), 'opinion chunks')
        print('Processed', len(statutes), 'statute chunks')
        print('Processed', len(regulations), 'regulation chunks')
        print('Processed', len(textbooks), 'textbook chunks')
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        print("No opinion data to process")
