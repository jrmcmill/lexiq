import os
from dataclasses import dataclass
from dotenv import load_dotenv
import torch
from src.observability.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

@dataclass
class Config:
    COURTLISTENER_API_KEY: str = os.getenv("COURTLISTENER_API_KEY")
    CONGRESS_API_KEY: str = os.getenv("CONGRESS_API_KEY")
    GOVINFO_API_KEY: str = os.getenv("GOVINFO_API_KEY")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_CASES_COLLECTION: str = os.getenv("CHROMA_CASES_COLLECTION", "lexiq_cases")
    CHROMA_STATUTES_COLLECTION: str = os.getenv("CHROMA_STATUTES_COLLECTION", "lexiq_statutes")
    CHROMA_REGULATIONS_COLLECTION: str = os.getenv("CHROMA_REGULATIONS_COLLECTION", "lexiq_regulations")
    CHROMA_TITLES_COLLECTION: str = os.getenv("CHROMA_TITLES_COLLECTION", "lexiq_titles")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_DOCS_PER_TOOL: int = int(os.getenv("MAX_DOCS_PER_TOOL", "8"))
    MAX_NON_CASE_DOCS_PER_TOOL: int = int(os.getenv("MAX_NON_CASE_DOCS_PER_TOOL", "3"))
    MAX_CASE_REPAIR_ATTEMPTS: int = int(os.getenv("MAX_CASE_REPAIR_ATTEMPTS", "2"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
    COURTLISTENER_MAX_PAGES: int = int(os.getenv("COURTLISTENER_MAX_PAGES", "10"))
    COURTLISTENER_PAGE_SIZE: int = int(os.getenv("COURTLISTENER_PAGE_SIZE", "50"))
    
    # Hallucination reduction parameters
    # RETRIEVAL_MIN_DISTANCE: cosine distance threshold (0-2 scale, lower=stricter)
    # Typical range: 0.3 (very strict), 0.5 (moderate), 0.8 (permissive)
    RETRIEVAL_MIN_DISTANCE: float = float(os.getenv("RETRIEVAL_MIN_DISTANCE", "0.8"))
    # RERANK_MIN_SCORE: CrossEncoder relevance score (0-1 scale, higher=stricter)
    # Typical range: 0.1 (permissive), 0.3 (moderate), 0.7 (very strict)
    RERANK_MIN_SCORE: float = float(os.getenv("RERANK_MIN_SCORE", "0.1"))
    # LLM inference parameters (deterministic mode to reduce hallucination)
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    OLLAMA_TOP_P: float = float(os.getenv("OLLAMA_TOP_P", "0.15"))
    OLLAMA_TOP_K: int = int(os.getenv("OLLAMA_TOP_K", "20"))
    # Minimum sources required before attempting answer
    MIN_RETRIEVED_RESULTS: int = int(os.getenv("MIN_RETRIEVED_RESULTS", "1"))
    REQUIRE_SOURCES: bool = os.getenv("REQUIRE_SOURCES", "true").lower() == "true"
    # Enable query expansion (topical synonyms). Set to false to disable.
    ENABLE_QUERY_EXPANSION: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    # When true, retrieval methods may return a debug trace along with results
    RETRIEVAL_DEBUG: bool = os.getenv("RETRIEVAL_DEBUG", "false").lower() == "true"

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Selected device: {device}")
    return device

# expose module-level constants
Config = Config()
get_device = get_device
