import os
import json
import re
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from src.observability.logger import get_logger
from src.agent.citation import format_bluebook_case, extract_citations_from_text
from src.data.uscode import USCodeClient
from src.data.ecfr import ECFRClient

logger = get_logger(__name__)

class Preprocessor:
    def __init__(self):
        self.raw_dir = os.path.join(os.getcwd(), "data", "raw", "courtlistener")
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
            
            # Extract text from /search/ endpoint response structure
            text = data.get('plain_text')
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

if __name__ == '__main__':
    p = Preprocessor()
    try:
        cases = p.clean_opinions()
        statutes = p.clean_statutes()
        regulations = p.clean_regulations()
        print('Processed', len(cases), 'opinion chunks')
        print('Processed', len(statutes), 'statute chunks')
        print('Processed', len(regulations), 'regulation chunks')
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        print("No opinion data to process")
