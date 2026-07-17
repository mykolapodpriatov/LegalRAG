import qdrant_client
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore


def _cache_resource(**kwargs):
    """Return Streamlit's ``cache_resource`` decorator when available.

    Falls back to a no-op decorator so this module can be imported and its
    functions called outside a Streamlit runtime -- and even without Streamlit
    installed (e.g. in CI or tests). Caching the embedding model only matters
    inside the app; CLI/test usage tolerates the plain, uncached call.
    """
    try:
        import streamlit as st
    except ModuleNotFoundError:
        def _identity(func):
            return func

        return _identity
    return st.cache_resource(**kwargs)


@_cache_resource(show_spinner="Loading embedding model...")
def get_embed_model():
    """Loads the E5-multilingual embedding model once and caches it.

    The model is ~2.2 GB; without caching it would be reloaded on every index
    build, adding minutes of latency per run. The heavy ``HuggingFaceEmbedding``
    import (which pulls in torch/transformers) is deferred to call time, so
    merely importing this module stays cheap and dependency-free.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")


def build_index(nodes, embed_model=None):
    """Builds a Qdrant vector index over ``nodes``.

    Args:
        nodes: Parsed document nodes to embed and index.
        embed_model: Embedding model to use. Defaults to the cached
            E5-multilingual model via :func:`get_embed_model`. Injecting a
            lightweight model (e.g. ``MockEmbedding``) lets callers build an
            index without loading the heavyweight HuggingFace model, which is
            what tests rely on.

    Note:
        Qdrant runs in-memory (``location=":memory:"``), so the index is held
        only for the lifetime of the process and is rebuilt on restart.
    """
    if embed_model is None:
        embed_model = get_embed_model()

    # Setup Qdrant (in-memory for prototyping)
    client = qdrant_client.QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="legal_docs")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build Index
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    return index
