from src.rag.retriever import Retriever
from src.observability.logger import get_logger

logger = get_logger(__name__)
retriever = Retriever()


def case_law_search(query: str, court_filter: str | None = None, date_after: str | None = None, date_before: str | None = None, debug: bool = False, aggressive: bool = False) -> list[dict] | dict:
    try:
        return retriever.retrieve_cases(query, court_filter=court_filter, date_after=date_after, date_before=date_before, debug=debug, aggressive=aggressive)
    except Exception as e:
        logger.error(str(e))
        return {'results': [], 'trace': {}} if debug else []


def statute_search(query: str, debug: bool = False, aggressive: bool = False) -> list[dict] | dict:
    try:
        return retriever.retrieve_statutes(query, debug=debug, aggressive=aggressive)
    except Exception as e:
        logger.error(str(e))
        return {'results': [], 'trace': {}} if debug else []


def regulation_search(query: str, debug: bool = False, aggressive: bool = False) -> list[dict] | dict:
    try:
        return retriever.retrieve_regulations(query, debug=debug, aggressive=aggressive)
    except Exception as e:
        logger.error(str(e))
        return {'results': [], 'trace': {}} if debug else []


def textbook_search(query: str, debug: bool = False, aggressive: bool = False) -> list[dict] | dict:
    try:
        return retriever.retrieve_textbooks(query, debug=debug, aggressive=aggressive)
    except Exception as e:
        logger.error(str(e))
        return {'results': [], 'trace': {}} if debug else []


def session_document_search(query: str, session_id: str) -> list[dict]:
    try:
        return retriever.retrieve_session_docs(query, session_id)
    except Exception as e:
        logger.error(str(e))
        return []
