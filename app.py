import streamlit as st
import logging
import os
from src.ingestion import load_documents, chunk_documents
from src.retrieval import build_index, collection_is_populated, load_index
from src.generation import get_query_engine, generate_response
from src.formatting import format_source_nodes

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

            if collection_is_populated(persist_path, COLLECTION_NAME):
                index = load_index(persist_path=persist_path, collection_name=COLLECTION_NAME)
                st.session_state.qdrant_client = index.vector_store.client
                st.session_state.query_engine = get_query_engine(index)
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
                    st.session_state.query_engine = get_query_engine(index)
                    st.success(f"Successfully indexed {len(nodes)} chunks from {len(docs)} documents.")
        except Exception as e:
            logging.error(f"Exception during indexing: {e}", exc_info=True)
            st.error(f"Error loading/indexing documents: {str(e)}")

query = st.text_input("Enter your legal query:")

if st.button("Submit"):
    if st.session_state.query_engine is None:
        st.error("Please load and index documents first.")
    elif not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating answer..."):
            try:
                response = generate_response(st.session_state.query_engine, query)
                st.markdown("### Answer")
                st.write(response.response)
                
                st.markdown("### Sources")
                for citation in format_source_nodes(response.source_nodes):
                    st.info(citation)
            except Exception as e:
                logging.error(f"Exception during generation: {e}", exc_info=True)
                st.error(f"Error generating response: {str(e)}")
