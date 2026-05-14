import streamlit as st
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.rag.retriever import Retriever

st.set_page_config(page_title="Statute & Regulation Lookup", layout="wide")
st.title("📋 Statute & Regulation Lookup")
st.write("Search U.S. Code (statutes) and Code of Federal Regulations (CFR).")

@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

tab1, tab2 = st.tabs(["U.S. Code (Statutes)", "Code of Federal Regulations (CFR)"])

# U.S. Code Tab
with tab1:
    st.subheader("Search U.S. Code")
    st.write("Search by keyword, section number, or legal principle.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        statute_query = st.text_input(
            "Search statutes:",
            placeholder="e.g., 'copyright infringement', '17 U.S.C. § 102'",
            key="statute_search"
        )
    with col2:
        statute_results_count = st.slider("Results:", 1, 20, 5, key="statute_results")
    
    if statute_query:
        with st.spinner("Searching U.S. Code..."):
            try:
                results = retriever.retrieve_statutes(statute_query, n_results=statute_results_count)
                
                if results:
                    st.success(f"Found {len(results)} matching statutes")
                    
                    for i, result in enumerate(results, 1):
                        meta = result['metadata']
                        
                        title_str = f"{meta.get('usc_citation', 'Unknown')} - {meta.get('section_heading', '')}"
                        with st.expander(f"**{i}. {title_str}** (Distance: {result.get('distance', 0):.3f})", expanded=i==1):
                            col_meta, col_text = st.columns([1, 2])
                            
                            with col_meta:
                                st.subheader("Citation Info")
                                st.write(f"**Title:** {meta.get('title_number', 'N/A')}")
                                st.write(f"**Section:** {meta.get('section_number', 'N/A')}")
                                st.write(f"**Full Citation:** {meta.get('usc_citation', 'N/A')}")
                                if meta.get('section_heading'):
                                    st.write(f"**Heading:** {meta['section_heading']}")
                            
                            with col_text:
                                st.subheader("Statute Text")
                                st.text_area(
                                    label="Text",
                                    value=result['text'],
                                    height=300,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                else:
                    st.info("No matching statutes found.")
            except Exception as e:
                st.error(f"Error searching statutes: {str(e)}")

# CFR Tab
with tab2:
    st.subheader("Search Code of Federal Regulations")
    st.write("Search by keyword, regulation section, or topic.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        reg_query = st.text_input(
            "Search regulations:",
            placeholder="e.g., 'environmental protection', '40 CFR 112'",
            key="reg_search"
        )
    with col2:
        reg_results_count = st.slider("Results:", 1, 20, 5, key="reg_results")
    
    if reg_query:
        with st.spinner("Searching CFR..."):
            try:
                results = retriever.retrieve_regulations(reg_query, n_results=reg_results_count)
                
                if results:
                    st.success(f"Found {len(results)} matching regulations")
                    
                    for i, result in enumerate(results, 1):
                        meta = result['metadata']
                        
                        title_str = f"{meta.get('cfr_citation', 'Unknown')} - {meta.get('section_heading', '')}"
                        with st.expander(f"**{i}. {title_str}** (Distance: {result.get('distance', 0):.3f})", expanded=i==1):
                            col_meta, col_text = st.columns([1, 2])
                            
                            with col_meta:
                                st.subheader("Citation Info")
                                st.write(f"**Title:** {meta.get('cfr_title', 'N/A')}")
                                st.write(f"**Part:** {meta.get('cfr_part', 'N/A')}")
                                st.write(f"**Section:** {meta.get('cfr_section', 'N/A')}")
                                st.write(f"**Full Citation:** {meta.get('cfr_citation', 'N/A')}")
                                if meta.get('section_heading'):
                                    st.write(f"**Heading:** {meta['section_heading']}")
                            
                            with col_text:
                                st.subheader("Regulation Text")
                                st.text_area(
                                    label="Text",
                                    value=result['text'],
                                    height=300,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )
                else:
                    st.info("No matching regulations found.")
            except Exception as e:
                st.error(f"Error searching regulations: {str(e)}")

st.divider()
st.subheader("Quick Reference")
col1, col2 = st.columns(2)
with col1:
    st.write("**U.S. Code Citation Format:**")
    st.code("17 U.S.C. § 102\n35 U.S.C. § 101")
with col2:
    st.write("**CFR Citation Format:**")
    st.code("40 CFR 112\n29 CFR 1910")
