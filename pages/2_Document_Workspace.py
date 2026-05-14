import streamlit as st
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.documents.parser import DocumentParser
from src.documents.session_store import SessionDocumentStore
from src.documents.comparator import DocumentComparator
from src.rag.retriever import Retriever

st.set_page_config(page_title="Document Workspace", layout="wide")
st.title("📄 Document Workspace")
st.write("Upload, parse, and compare documents with existing legal sources.")

# Initialize components
@st.cache_resource
def load_parser():
    return DocumentParser()

@st.cache_resource
def load_retriever():
    return Retriever()

@st.cache_resource
def load_comparator():
    return DocumentComparator()

parser = load_parser()
retriever = load_retriever()
comparator = load_comparator()

if "session_id" not in st.session_state:
    st.session_state.session_id = "workspace_" + str(os.urandom(4).hex())

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}

st.subheader("1. Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files:",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_docs:
            with st.spinner(f"Parsing {uploaded_file.name}..."):
                try:
                    parsed = parser.parse_file(uploaded_file)
                    st.session_state.uploaded_docs[uploaded_file.name] = {
                        "content": parsed,
                        "size": len(parsed),
                        "chunks": len(parsed.split('\n\n'))
                    }
                    st.success(f"✓ Parsed {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to parse {uploaded_file.name}: {str(e)}")

if st.session_state.uploaded_docs:
    st.divider()
    st.subheader("2. Uploaded Documents")
    
    cols = st.columns(len(st.session_state.uploaded_docs))
    for col, (fname, doc_info) in zip(cols, st.session_state.uploaded_docs.items()):
        with col:
            st.metric(fname, f"{doc_info['chunks']} sections", f"{doc_info['size']} chars")
            if st.button(f"Remove {fname}", key=f"rm_{fname}"):
                del st.session_state.uploaded_docs[fname]
                st.rerun()
    
    st.divider()
    st.subheader("3. Compare with Legal Sources")
    
    selected_doc = st.selectbox(
        "Select a document to analyze:",
        list(st.session_state.uploaded_docs.keys())
    )
    
    if selected_doc:
        doc_content = st.session_state.uploaded_docs[selected_doc]["content"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_cases = st.checkbox("Search Cases", value=True)
        with col2:
            search_statutes = st.checkbox("Search Statutes", value=True)
        with col3:
            search_regs = st.checkbox("Search Regulations", value=True)
        
        if st.button("Find Related Legal Documents"):
            with st.spinner("Searching legal databases..."):
                results_all = []
                
                if search_cases:
                    case_results = retriever.retrieve_cases(doc_content[:500], n_results=3)
                    results_all.extend([("Case", r) for r in case_results])
                
                if search_statutes:
                    stat_results = retriever.retrieve_statutes(doc_content[:500], n_results=3)
                    results_all.extend([("Statute", r) for r in stat_results])
                
                if search_regs:
                    reg_results = retriever.retrieve_regulations(doc_content[:500], n_results=3)
                    results_all.extend([("Regulation", r) for r in reg_results])
                
                if results_all:
                    st.success(f"Found {len(results_all)} related legal sources")
                    
                    for source_type, result in results_all:
                        with st.expander(f"**{source_type}:** {result['metadata'].get('case_name') or result['metadata'].get('usc_citation') or result['metadata'].get('cfr_citation', 'Unknown')}"):
                            st.write(f"**Type:** {source_type}")
                            for k, v in result['metadata'].items():
                                if v:
                                    st.write(f"**{k.replace('_', ' ').title()}:** {v}")
                            st.text_area("Content Preview", value=result['document'][:300], height=100, disabled=True)
                else:
                    st.info("No related legal documents found.")

st.divider()
st.info("💡 Pro Tip: Upload legal documents, contracts, or briefs to find related case law, statutes, and regulations automatically.")
