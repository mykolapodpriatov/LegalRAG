import qdrant_client
import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model():
    """Loads the E5-multilingual embedding model once and caches it.

    The model is ~2.2 GB; without caching it would be reloaded on every
    index build, adding minutes of latency per run.
    """
    return HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")


def build_index(nodes):
    """Builds Qdrant vector index using E5-multilingual.

    Note: Qdrant runs in-memory (``location=":memory:"``), so the index is
    held only for the lifetime of the process and is rebuilt on restart.
    """
    # Setup Qdrant
    client = qdrant_client.QdrantClient(location=":memory:")  # Use memory for prototyping
    vector_store = QdrantVectorStore(client=client, collection_name="legal_docs")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Setup Embedding Model (cached across reruns)
    embed_model = get_embed_model()

    # Build Index
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    return index
