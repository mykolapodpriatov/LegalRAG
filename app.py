import streamlit as st
import logging
from src.ingestion import load_documents, chunk_documents
from src.retrieval import build_index
from src.generation import get_query_engine, generate_response

st.set_page_config(page_title="LegalRAG", layout="wide")

st.title("LegalRAG: Multilingual Legal Assistant")
st.markdown("Ask legal questions based on the provided documents. Ensure Anthropic API key is set in your environment.")

# Initialize session state
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

data_dir = st.text_input("Data Directory", value="./data")

if st.button("Load & Index Documents"):
    with st.spinner("Loading and indexing..."):
        try:
            docs = load_documents(data_dir)
            if not docs:
                st.warning("No documents found.")
            else:
                nodes = chunk_documents(docs)
                index = build_index(nodes)
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
                for source in response.source_nodes:
                    st.info(f"**Score:** {source.score}\n\n**Text:** {source.node.text[:200]}...")
            except Exception as e:
                logging.error(f"Exception during generation: {e}", exc_info=True)
                st.error(f"Error generating response: {str(e)}")
