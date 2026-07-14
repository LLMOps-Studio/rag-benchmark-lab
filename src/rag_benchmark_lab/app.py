import os
import json
import streamlit as st
import pandas as pd
from rag_benchmark_lab.benchmark import RAGBenchmarkRunner
from llmops_common.client.ollama_client import OllamaClient
from llmops_common.ui.streamlit_helpers import inject_theme

# Set up global Streamlit page configurations
st.set_page_config(
    page_title="RAG Benchmark Lab",
    page_icon="🧪",
    layout="wide"
)

# Apply common LLMOps UI theme
inject_theme("rag_benchmark")

st.title("🧪 RAG Benchmark & Optimization Laboratory")
st.markdown("""
Evaluate local RAG pipelines to find the optimal balance between **Accuracy (Faithfulness & Relevance)** and **Infrastructure Efficiency (Latency)**.
All historical experiment data is automatically synced to the central **MLflow** registry.
""")

# --- DATASET LOADER ---
DATASET_PATH = "data/benchmark/golden_dataset.json"

@st.cache_data
def load_golden_dataset():
    """Loads the SQuAD v2.0 dataset or falls back to defaults if not generated yet."""
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Combine unique contexts to simulate a real Knowledge Base
        unique_contexts = list(set([item["context"] for item in data]))
        kb_text = "\n\n".join(unique_contexts)
        
        # Extract evaluation queries
        queries = "\n".join([item["query"] for item in data])
        
        dataset_info = f"✅ Loaded SQuAD Golden Dataset ({len(data)} queries, {len(unique_contexts)} distinct contexts)."
        return kb_text, queries, dataset_info
    else:
        fallback_text = "Modern vector databases, such as ChromaDB, utilize HNSW graphs..."
        fallback_queries = "What data structure does ChromaDB use?\nHow does RAG improve LLM reliability?"
        warning_info = "⚠️ Golden dataset not found. Using fallback data. Run `python scripts/prepare_squad.py` first."
        return fallback_text, fallback_queries, warning_info

default_kb_text, default_queries, status_msg = load_golden_dataset()

# Sidebar control layout for combinatorial configuration parameters
st.sidebar.header("🛠️ Pipeline Grid Parameters")

ollama_client = OllamaClient()
available_models = ollama_client.get_available_models()

if not available_models:
    st.sidebar.warning(f"No models found at {ollama_client.host}. Please pull one below.")
    available_models = ["phi3:latest"]

selected_models = st.sidebar.multiselect(
    "Select Local Models to Test",
    options=available_models,
    default=[available_models[0]] if available_models else None
)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Pull New Model")
new_model_name = st.sidebar.text_input("Model Name (e.g., qwen2:0.5b)")
if st.sidebar.button("Pull from Registry"):
    if new_model_name:
        with st.spinner(f"Pulling '{new_model_name}'..."):
            success = ollama_client.pull_model(new_model_name)
            if success:
                st.sidebar.success(f"{new_model_name} downloaded!")
                st.rerun()
            else:
                st.sidebar.error("Failed to pull model.")
st.sidebar.markdown("---")

chunk_sizes_raw = st.sidebar.text_input("Chunk Sizes (comma-separated)", value="150, 300")
chunk_overlaps_raw = st.sidebar.text_input("Chunk Overlaps (comma-separated)", value="15, 30")

try:
    chunk_sizes = [int(x.strip()) for x in chunk_sizes_raw.split(",")]
    chunk_overlaps = [int(x.strip()) for x in chunk_overlaps_raw.split(",")]
except ValueError:
    st.sidebar.error("Error: Ensure chunk settings contain only valid integers.")
    st.stop()

# Main Workspace Input Panels
if "⚠️" in status_msg:
    st.warning(status_msg)
else:
    st.success(status_msg)

col_text, col_queries = st.columns(2)

with col_text:
    st.subheader("📄 Target Knowledge Base Document")
    raw_document = st.text_area("Paste context source text here:", value=default_kb_text, height=300)

with col_queries:
    st.subheader("❓ Golden Query Evaluation Dataset")
    queries_raw = st.text_area("Enter evaluation questions (One per line):", value=default_queries, height=300)
    test_queries = [q.strip() for q in queries_raw.split("\n") if q.strip()]

# Triggering the Grid Execution
if st.button("🚀 Execute Optimization Benchmark", type="primary"):
    if not selected_models:
        st.error("Please select at least one local model to evaluate.")
    elif not test_queries:
        st.error("Evaluation dataset cannot be empty. Provide test queries.")
    else:
        st.info(f"Grid search active. Testing {len(selected_models) * len(chunk_sizes) * len(chunk_overlaps)} configurations...")
        
        runner = RAGBenchmarkRunner(experiment_name="RAG_Optimization_Lab")
        
        with st.spinner("Processing embeddings, evaluating LLM-as-a-Judge, and flushing VRAM dynamically..."):
            summary_data = runner.run_grid_benchmark(
                raw_text=raw_document,
                test_queries=test_queries,
                chunk_sizes=chunk_sizes,
                chunk_overlaps=chunk_overlaps,
                models=selected_models
            )
            
            st.success("Optimization run completed successfully! Metrics synchronized to MLflow.")
            
            df_results = pd.DataFrame(summary_data)
            
            st.subheader("🏆 Interactive RAG Leaderboard")
            st.dataframe(
                df_results.sort_values(by=["avg_faithfulness", "avg_latency"], ascending=[False, True]),
                use_container_width=True
            )
            
            st.subheader("📊 Architectural Trade-Off Analysis")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("**Accuracy (Faithfulness vs Relevance Score)**")
                st.bar_chart(df_results, x="run_name", y=["avg_faithfulness", "avg_relevance"])
                
            with col_chart2:
                st.markdown("**Infrastructure Cost (Average Latency in Seconds)**")
                st.bar_chart(df_results, x="run_name", y="avg_latency")