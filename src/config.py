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
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_DOCS_PER_TOOL: int = int(os.getenv("MAX_DOCS_PER_TOOL", "8"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))
    COURTLISTENER_MAX_PAGES: int = int(os.getenv("COURTLISTENER_MAX_PAGES", "10"))
    COURTLISTENER_PAGE_SIZE: int = int(os.getenv("COURTLISTENER_PAGE_SIZE", "50"))

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
