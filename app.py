import streamlit as st
import logging
import os
from src.ingestion import load_documents, chunk_documents
from src.retrieval import build_index, collection_is_populated, load_index
from src.generation import get_query_engine, generate_response
from src.formatting import format_source_nodes
from src.history import DEFAULT_HISTORY_CAP, push_history

st.set_page_config(page_title="LegalRAG", layout="wide")

st.title("LegalRAG: Multilingual Legal Assistant")
st.markdown("Ask legal questions based on the provided documents. Ensure Anthropic API key is set in your environment.")

COLLECTION_NAME = "legal_docs"
DEFAULT_QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_storage")

# Initialize session state
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None
if "index" not in st.session_state:
    st.session_state.index = None
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None

with st.sidebar:
    similarity_top_k = st.slider(
        "Passages to retrieve (top-k)",
        min_value=1,
        max_value=20,
        value=5,
        help="How many similar passages to retrieve for each query.",
    )

    st.markdown("### History")
    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.selected_history = None
    if not st.session_state.history:
        st.caption("No queries yet.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            label = item["query"]
            if len(label) > 60:
                label = label[:57] + "..."
            # Newest first; key on original index so labels stay unique.
            original_idx = len(st.session_state.history) - 1 - idx
            if st.button(label, key=f"history_{original_idx}"):
                st.session_state.query_input = item["query"]
                st.session_state.selected_history = item

data_dir = st.text_input("Data Directory", value="./data")
persist_path_input = st.text_input(
    "Qdrant Storage Path",
    value=DEFAULT_QDRANT_PATH,
    help="On-disk directory for the vector index. Leave blank to use an "
    "in-memory index that is rebuilt every session.",
)
persist_path = persist_path_input.strip() or None

if st.button("Load & Index Documents"):
    with st.spinner("Loading and indexing..."):
        try:
            # Qdrant's local persistence takes an exclusive lock on
            # persist_path. Release any client from a previous run/re-run
            # before opening a new one on the same directory.
            if st.session_state.qdrant_client is not None:
                st.session_state.qdrant_client.close()
                st.session_state.qdrant_client = None
                st.session_state.index = None

            if collection_is_populated(persist_path, COLLECTION_NAME):
                index = load_index(persist_path=persist_path, collection_name=COLLECTION_NAME)
                st.session_state.qdrant_client = index.vector_store.client
                st.session_state.index = index
                st.session_state.query_engine = get_query_engine(
                    index, similarity_top_k=similarity_top_k
                )
                st.session_state.engine_top_k = similarity_top_k
                st.success("Loaded existing index from persistent storage.")
            else:
                docs = load_documents(data_dir)
                if not docs:
                    st.warning("No documents found.")
                else:
                    nodes = chunk_documents(docs)
                    index = build_index(
                        nodes, persist_path=persist_path, collection_name=COLLECTION_NAME
                    )
                    st.session_state.qdrant_client = index.vector_store.client
                    st.session_state.index = index
                    st.session_state.query_engine = get_query_engine(
                        index, similarity_top_k=similarity_top_k
                    )
                    st.session_state.engine_top_k = similarity_top_k
                    st.success(f"Successfully indexed {len(nodes)} chunks from {len(docs)} documents.")
        except Exception as e:
            logging.error(f"Exception during indexing: {e}", exc_info=True)
            st.error(f"Error loading/indexing documents: {str(e)}")

# Rebuild the query engine when top-k changes; the Qdrant index stays put.
if (
    st.session_state.index is not None
    and st.session_state.get("engine_top_k") != similarity_top_k
):
    st.session_state.query_engine = get_query_engine(
        st.session_state.index, similarity_top_k=similarity_top_k
    )
    st.session_state.engine_top_k = similarity_top_k

query = st.text_input("Enter your legal query:", key="query_input")

if st.button("Submit"):
    if st.session_state.query_engine is None:
        st.error("Please load and index documents first.")
    elif not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = generate_response(st.session_state.query_engine, query)
                entry = {
                    "query": query,
                    "answer": response.response,
                    "sources": response.source_nodes,
                }
                st.session_state.history = push_history(
                    st.session_state.history, entry, cap=DEFAULT_HISTORY_CAP
                )
                st.session_state.selected_history = entry
            except Exception as e:
                logging.error(f"Exception during generation: {e}", exc_info=True)
                st.error(f"Error generating response: {str(e)}")

if st.session_state.selected_history is not None:
    entry = st.session_state.selected_history
    st.markdown("### Answer")
    st.write(entry["answer"])
    st.markdown("### Sources")
    for citation in format_source_nodes(entry["sources"]):
        st.info(citation)
