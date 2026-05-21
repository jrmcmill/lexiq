import streamlit as st
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.rag.retriever import Retriever
from src.ui_helpers import format_distance

st.set_page_config(page_title="Case Law Explorer", layout="wide")
st.title("⚖️ Case Law Explorer")
st.write("Search and explore case law from CourtListener opinions.")

# Initialize retriever
@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()
counts = retriever.indexer.get_entity_counts()

st.metric("Cases stored", counts.get("cases", 0))

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Search case law by keyword, citation, or legal principle:", placeholder="e.g., 'constitutional due process'")

with col2:
    n_results = st.slider("Results to display:", min_value=1, max_value=20, value=5)

if query:
    with st.spinner("Searching case law..."):
        try:
            # Retrieve from cases collection (already reranked by retriever)
            results = retriever.retrieve_cases(query, n_results=n_results)
            
            if results:
                st.success(f"Found {len(results)} matching documents")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"**{i}. {result['metadata'].get('case_name', 'Unknown Case')}** (Distance: {format_distance(result.get('distance'))})", expanded=i==1):
                        col_meta, col_text = st.columns([1, 2])
                        
                        with col_meta:
                            st.subheader("Metadata")
                            if 'court' in result['metadata']:
                                st.write(f"**Court:** {result['metadata']['court']}")
                            if 'date_filed' in result['metadata']:
                                st.write(f"**Date:** {result['metadata']['date_filed']}")
                            if 'docket_number' in result['metadata']:
                                st.write(f"**Docket:** {result['metadata']['docket_number']}")
                            if 'bluebook_cite' in result['metadata'] and result['metadata']['bluebook_cite']:
                                st.write(f"**Citation:** {result['metadata']['bluebook_cite']}")
                            if 'citations' in result['metadata'] and result['metadata']['citations']:
                                st.write(f"**Cited Cases:** {result['metadata']['citations'][:100]}...")
                        
                        with col_text:
                            st.subheader("Opinion Text")
                            st.text_area(
                                label="Text snippet",
                                value=result['text'][:500] + "..." if len(result['text']) > 500 else result['text'],
                                height=200,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"case_text_{i}"
                            )
            else:
                st.info("No matching cases found. Try a different query.")
        except Exception as e:
            st.error(f"Error searching cases: {str(e)}")

st.divider()
st.subheader("Quick Tips")
st.write("""
- Search by keyword (e.g., "First Amendment", "breach of contract")
- Search by legal principle (e.g., "due process", "negligence")
- Search by party name (if available in indexed cases)
- Adjust result count to see more or fewer matches
""")
