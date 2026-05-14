import streamlit as st
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.rag.retriever import Retriever

st.set_page_config(page_title="Evaluation Dashboard", layout="wide")
st.title("📊 Evaluation Dashboard")
st.write("Run benchmarks and evaluate LexIQ performance on legal research tasks.")

@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = {}

tab1, tab2 = st.tabs(["Run Benchmarks", "View Results"])

# Run Benchmarks Tab
with tab1:
    st.subheader("Benchmark Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_queries = st.text_area(
            "Test Queries (one per line):",
            value="constitutional due process\npatent infringement\nenvironmental protection\n",
            height=150,
            help="Enter legal research queries to test retrieval quality"
        )
    
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            n_results = st.number_input("Results per query:", 1, 20, 5)
        with col_b:
            eval_metric = st.selectbox("Metric:", ["precision", "recall", "mrr"])
    
    if st.button("Run Benchmarks", type="primary"):
        queries = [q.strip() for q in test_queries.split('\n') if q.strip()]
        
        if queries:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_placeholder = st.empty()
            
            all_results = {}
            
            for i, query in enumerate(queries):
                status_text.text(f"Running benchmark {i+1}/{len(queries)}: {query}")
                progress_bar.progress((i + 1) / len(queries))
                
                try:
                    # Retrieve cases
                    case_results = retriever.retrieve_cases(query, n_results=n_results)
                    stat_results = retriever.retrieve_statutes(query, n_results=n_results)
                    reg_results = retriever.retrieve_regulations(query, n_results=n_results)
                    
                    all_results[query] = {
                        "cases": len(case_results),
                        "statutes": len(stat_results),
                        "regulations": len(reg_results),
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    st.warning(f"Error on query '{query}': {str(e)}")
                    all_results[query] = {"error": str(e)}
            
            st.session_state.benchmark_results[f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = all_results
            status_text.success(f"✓ Completed {len(queries)} benchmark queries")
            
            # Display results
            with results_placeholder.container():
                st.subheader("Benchmark Results")
                for query, result in all_results.items():
                    if "error" not in result:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"Cases - {query[:20]}...", result['cases'])
                        with col2:
                            st.metric("Statutes", result['statutes'])
                        with col3:
                            st.metric("Regulations", result['regulations'])
        else:
            st.warning("Please enter at least one test query.")

# View Results Tab
with tab2:
    st.subheader("Benchmark History")
    
    if st.session_state.benchmark_results:
        for run_name, results in st.session_state.benchmark_results.items():
            with st.expander(f"**{run_name}**", expanded=False):
                st.write(f"**Queries run:** {len(results)}")
                
                # Summary metrics
                total_cases = sum(r.get('cases', 0) for r in results.values())
                total_stats = sum(r.get('statutes', 0) for r in results.values())
                total_regs = sum(r.get('regulations', 0) for r in results.values())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Cases Found", total_cases)
                with col2:
                    st.metric("Total Statutes Found", total_stats)
                with col3:
                    st.metric("Total Regulations Found", total_regs)
                
                st.divider()
                
                # Detailed results
                st.write("**Query-by-Query Results:**")
                for query, result in results.items():
                    if "error" not in result:
                        st.write(f"- **{query}:** Cases={result['cases']}, Statutes={result['statutes']}, Regulations={result['regulations']}")
                    else:
                        st.write(f"- **{query}:** Error - {result['error']}")
    else:
        st.info("No benchmark results yet. Run benchmarks on the 'Run Benchmarks' tab to see results here.")

st.divider()
st.subheader("About Benchmarks")
st.write("""
LexIQ evaluation metrics:
- **Precision**: What fraction of retrieved documents are relevant
- **Recall**: What fraction of relevant documents are retrieved
- **MRR (Mean Reciprocal Rank)**: Average rank position of the first relevant result

Benchmarks help identify:
- Retrieval quality across different legal domains
- Performance on different types of queries
- Reranking effectiveness
""")
