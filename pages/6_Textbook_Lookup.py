import streamlit as st
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.rag.retriever import Retriever
from src.ui_helpers import format_distance

st.set_page_config(page_title="Textbook Lookup", layout="wide")
st.title("📚 Textbook Lookup")
st.write("Search the local textbook corpus for doctrinal background and explanatory material.")
st.caption("Textbooks are treated as secondary sources. Use them for background, then rely on controlling case law, statutes, or regulations for authority.")

@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()
counts = retriever.indexer.get_entity_counts()

st.metric("Textbooks stored", counts.get("textbooks", 0))

col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input(
        "Search the textbook corpus by keyword, topic, or chapter name:",
        placeholder="e.g., negligence, torts, contract formation"
    )
with col2:
    n_results = st.slider("Results to display:", min_value=1, max_value=20, value=5)

if query:
    with st.spinner("Searching textbooks..."):
        try:
            results = retriever.retrieve_textbooks(query, n_results=n_results)

            if results:
                st.success(f"Found {len(results)} matching textbook chunks")

                for i, result in enumerate(results, 1):
                    meta = result.get('metadata', {}) if isinstance(result, dict) else {}
                    title = meta.get('book_title') or meta.get('source_filename') or 'Unknown Textbook'
                    chapter = meta.get('chapter') or meta.get('section_heading') or 'Unknown Chapter'
                    page_number = meta.get('page_number') or meta.get('page_start') or 'N/A'
                    author = meta.get('book_author') or 'Unknown Author'

                    expander_title = f"**{i}. {title}** - {chapter} (Page {page_number}, Distance: {format_distance(result.get('distance'))})"
                    with st.expander(expander_title, expanded=i == 1):
                        meta_col, text_col = st.columns([1, 2])

                        with meta_col:
                            st.subheader("Metadata")
                            st.write(f"**Title:** {title}")
                            st.write(f"**Author:** {author}")
                            if meta.get('book_subject'):
                                st.write(f"**Subject:** {meta['book_subject']}")
                            if meta.get('chapter'):
                                st.write(f"**Chapter:** {meta['chapter']}")
                            if meta.get('section_heading') and meta.get('section_heading') != meta.get('chapter'):
                                st.write(f"**Section:** {meta['section_heading']}")
                            if meta.get('page_start') is not None:
                                page_end = meta.get('page_end')
                                if page_end and page_end != meta.get('page_start'):
                                    st.write(f"**Pages:** {meta['page_start']} - {page_end}")
                                else:
                                    st.write(f"**Page:** {meta.get('page_start')}")
                            if meta.get('source_filename'):
                                st.write(f"**File:** {meta['source_filename']}")

                        with text_col:
                            st.subheader("Textbook Text")
                            st.text_area(
                                label="Text snippet",
                                value=result.get('text', '')[:800] + "..." if len(result.get('text', '')) > 800 else result.get('text', ''),
                                height=240,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"textbook_text_{i}"
                            )
            else:
                st.info("No matching textbook passages found. Try a different topic or chapter name.")
        except Exception as e:
            st.error(f"Error searching textbooks: {str(e)}")

st.divider()
st.subheader("Quick Tips")
st.write("""
- Search by doctrinal topic, chapter name, or a core legal concept.
- Use textbook results for background and framing, then verify the controlling authorities in case law or statutes.
- If the book uses chapter headings, try those terms directly.
""")
