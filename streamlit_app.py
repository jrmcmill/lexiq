import streamlit as st
from datetime import datetime
import uuid
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.observability.logger import get_logger
from src.config import Config, get_device
from src.documents.session_store import SessionDocumentStore
from src.agent.graph import run_query

logger = get_logger(__name__)

st.set_page_config(page_title="LexIQ — Legal Research Assistant", page_icon="⚖️", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "session_store" not in st.session_state:
    st.session_state.session_store = SessionDocumentStore(st.session_state.session_id)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "research_history" not in st.session_state:
    st.session_state.research_history = []

st.sidebar.markdown("# LexIQ")
st.sidebar.write(f"Model: {Config.OLLAMA_MODEL}")
st.sidebar.write(f"Device: {get_device()}")
st.sidebar.divider()

# Initialize filter variables with defaults
court_filter = None
date_after = None
date_before = None

with st.sidebar.expander("⚙️ Search Filters", expanded=False):
    court_filter = st.text_input("Filter by Court (optional):", placeholder="e.g., 'U.S. Supreme Court'", value="") or None
    date_after = st.text_input("Cases after (YYYY-MM-DD):", placeholder="2020-01-01", value="") or None
    date_before = st.text_input("Cases before (YYYY-MM-DD):", placeholder="2024-12-31", value="") or None

st.sidebar.divider()

# Document upload section
st.sidebar.subheader("📄 Upload Document")
uploaded_files = st.sidebar.file_uploader(
    "Upload legal documents (PDF, DOCX, TXT) to search for related sources",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="sidebar_uploads"
)

if uploaded_files:
    from src.documents.parser import DocumentParser
    parser = DocumentParser()
    
    for file in uploaded_files:
        if file.name not in st.session_state.get("uploaded_docs", {}):
            with st.spinner(f"Processing {file.name}..."):
                try:
                    content = parser.parse_file(file)
                    if "uploaded_docs" not in st.session_state:
                        st.session_state.uploaded_docs = {}
                    st.session_state.uploaded_docs[file.name] = {
                        "content": content,
                        "size": file.size
                    }
                    st.sidebar.success(f"✅ {file.name} uploaded")
                except Exception as e:
                    st.sidebar.error(f"Error processing {file.name}: {str(e)}")

st.sidebar.divider()
st.sidebar.write("⚠️ LexIQ is a research tool, not legal advice.")

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.research_history = []
    st.rerun()

st.title("⚖️ LexIQ — Legal Research Assistant")
st.write("Ask questions about U.S. case law, statutes, and regulations. LexIQ will search its knowledge base and provide sourced answers.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources", expanded=False):
                for source in message["sources"]:
                    st.write(f"**{source['type']}**: {source['citation']}")
        if message.get("metrics"):
            metrics = message["metrics"]
            with st.expander("📊 Metrics", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📜 Cases", metrics.get("cases", 0))
                with col2:
                    st.metric("📋 Statutes", metrics.get("statutes", 0))
                with col3:
                    st.metric("⚖️ Regulations", metrics.get("regulations", 0))
                st.caption(f"Total sources: {metrics.get('total', 0)}")

# Chat input
query = st.chat_input("Ask a legal research question...")
if query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Researching legal sources..."):
            try:
                # Run the agent
                result = run_query(
                    query=query,
                    session_id=st.session_state.session_id,
                    history=st.session_state.research_history,
                    court_filter=court_filter,
                    date_after=date_after,
                    date_before=date_before
                )
                
                # Extract answer
                answer = result.get('final_answer', 'No answer generated')
                if not answer or answer.strip() == '':
                    answer = "I could not find relevant information to answer your question. Try rephrasing or searching for related terms."
                
                # Display answer
                st.markdown(answer)
                
                # Show uploaded documents if available
                if st.session_state.get("uploaded_docs"):
                    st.info(f"📄 **Context from uploaded documents:** {len(st.session_state.uploaded_docs)} document(s) included in search")
                
                # Collect sources from results
                sources = []
                try:
                    cases = result.get('retrieved_cases', []) if isinstance(result.get('retrieved_cases'), list) else []
                    for case in cases:
                        if isinstance(case, dict):
                            meta = case.get('metadata', {})
                            case_name = meta.get('case_name', 'Unknown')
                            citation = meta.get('bluebook_cite') or case_name
                            sources.append({
                                "type": "Case Law",
                                "citation": citation,
                                "distance": case.get('distance', 0)
                            })
                except Exception as e:
                    logger.warning(f"Error processing cases: {str(e)}")
                
                try:
                    statutes = result.get('retrieved_statutes', []) if isinstance(result.get('retrieved_statutes'), list) else []
                    for statute in statutes:
                        if isinstance(statute, dict):
                            meta = statute.get('metadata', {})
                            citation = meta.get('usc_citation', 'Unknown')
                            sources.append({
                                "type": "U.S. Code",
                                "citation": citation,
                                "distance": statute.get('distance', 0)
                            })
                except Exception as e:
                    logger.warning(f"Error processing statutes: {str(e)}")
                
                try:
                    regs = result.get('retrieved_regs', []) if isinstance(result.get('retrieved_regs'), list) else []
                    for reg in regs:
                        if isinstance(reg, dict):
                            meta = reg.get('metadata', {})
                            citation = meta.get('cfr_citation', 'Unknown')
                            sources.append({
                                "type": "Regulation",
                                "citation": citation,
                                "distance": reg.get('distance', 0)
                            })
                except Exception as e:
                    logger.warning(f"Error processing regulations: {str(e)}")
                
                # Display sources
                if sources:
                    with st.expander(f"📎 Sources ({len(sources)})", expanded=True):
                        for source in sources:
                            st.write(f"**{source['type']}**: {source['citation']}")
                
                # Calculate and display metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                cases_retrieved = len([s for s in sources if s['type'] == 'Case Law'])
                statutes_retrieved = len([s for s in sources if s['type'] == 'U.S. Code'])
                regs_retrieved = len([s for s in sources if s['type'] == 'Regulation'])
                
                with metrics_col1:
                    st.metric("📜 Cases Retrieved", cases_retrieved)
                
                with metrics_col2:
                    st.metric("📋 Statutes Retrieved", statutes_retrieved)
                
                with metrics_col3:
                    st.metric("⚖️ Regulations Retrieved", regs_retrieved)
                
                # Calculate average relevance scores (1 - distance, normalized)
                if cases_retrieved > 0:
                    cases = [s for s in sources if s['type'] == 'Case Law']
                    avg_case_score = sum(1 - s.get('distance', 0) for s in cases) / cases_retrieved
                    st.write(f"**Case Relevance Score:** {avg_case_score:.1%}")
                
                if statutes_retrieved > 0:
                    statutes = [s for s in sources if s['type'] == 'U.S. Code']
                    avg_stat_score = sum(1 - s.get('distance', 0) for s in statutes) / statutes_retrieved
                    st.write(f"**Statute Relevance Score:** {avg_stat_score:.1%}")
                
                if regs_retrieved > 0:
                    regs = [s for s in sources if s['type'] == 'Regulation']
                    avg_reg_score = sum(1 - s.get('distance', 0) for s in regs) / regs_retrieved
                    st.write(f"**Regulation Relevance Score:** {avg_reg_score:.1%}")
                
                # Overall coverage metric
                total_sources = cases_retrieved + statutes_retrieved + regs_retrieved
                if total_sources > 0:
                    st.divider()
                    coverage_text = f"✓ **Query Coverage:** Found {total_sources} relevant legal sources across {sum([1 for x in [cases_retrieved, statutes_retrieved, regs_retrieved] if x > 0])} source types"
                    st.info(coverage_text)
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "metrics": {
                        "cases": cases_retrieved,
                        "statutes": statutes_retrieved,
                        "regulations": regs_retrieved,
                        "total": total_sources
                    }
                })
                
                # Add to research history for context
                st.session_state.research_history.append({
                    "query": query,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                })
                
            except RuntimeError as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                logger.error(str(e))
            except Exception as e:
                import traceback
                error_msg = f"⚠️ An error occurred: {str(e)}"
                st.error(error_msg)
                logger.error(f"Error: {str(e)}\nTraceback: {traceback.format_exc()}")
                # Try to provide more debugging info
                try:
                    logger.debug(f"Result type: {type(result)}, Result: {result}")
                except:
                    pass
