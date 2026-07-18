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


def build_index(nodes, embed_model=None, persist_path=None, collection_name="legal_docs"):
    """Builds a Qdrant vector index over ``nodes``.

    Args:
        nodes: Parsed document nodes to embed and index.
        embed_model: Embedding model to use. Defaults to the cached
            E5-multilingual model via :func:`get_embed_model`. Injecting a
            lightweight model (e.g. ``MockEmbedding``) lets callers build an
            index without loading the heavyweight HuggingFace model, which is
            what tests rely on.
        persist_path: Optional on-disk directory for Qdrant's local storage.
            When ``None`` (default) Qdrant runs in-memory and the index lives
            only for the process lifetime. When set, embeddings are written to
            disk so the corpus need not be re-embedded on the next run.
        collection_name: Name of the Qdrant collection to write into.

    Note:
        Qdrant's local persistence takes an exclusive lock on ``persist_path``;
        close the client (``index.vector_store.client.close()``) before opening
        another client on the same directory.
    """
    if embed_model is None:
        embed_model = get_embed_model()

    # In-memory when no path is given; otherwise persist to disk so the corpus
    # survives restarts instead of being re-embedded every time.
    if persist_path is None:
        client = qdrant_client.QdrantClient(location=":memory:")
    else:
        client = qdrant_client.QdrantClient(path=persist_path)

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build Index
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    return index
