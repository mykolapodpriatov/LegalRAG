import os

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


def collection_is_populated(persist_path, collection_name="legal_docs"):
    """Checks whether ``persist_path`` already holds a non-empty ``collection_name``.

    Callers use this to decide between reusing a persisted index (via
    :func:`load_index`) and re-embedding the corpus from scratch (via
    :func:`build_index`). Opens and closes its own client so it does not hold
    the on-disk lock afterwards.

    Args:
        persist_path: On-disk directory to check. ``None`` or a nonexistent
            path is treated as "not populated" rather than an error.
        collection_name: Name of the Qdrant collection to look for.

    Returns:
        True if ``persist_path`` exists and contains a ``collection_name``
        collection with at least one point.
    """
    if not persist_path or not os.path.isdir(persist_path):
        return False

    client = qdrant_client.QdrantClient(path=persist_path)
    try:
        names = {c.name for c in client.get_collections().collections}
        if collection_name not in names:
            return False
        return client.count(collection_name=collection_name).count > 0
    finally:
        client.close()


def load_index(embed_model=None, persist_path=None, collection_name="legal_docs"):
    """Opens an already-populated on-disk Qdrant collection as a queryable index.

    Unlike :func:`build_index`, this does not embed any nodes -- it wraps the
    existing collection so callers can query it directly, avoiding a costly
    re-embed of the whole corpus on every run. Check
    :func:`collection_is_populated` first; opening a missing collection would
    silently create an empty one.

    Args:
        embed_model: Embedding model used to embed incoming queries against
            the existing collection. Defaults to the cached E5-multilingual
            model via :func:`get_embed_model`.
        persist_path: On-disk directory holding the Qdrant collection.
        collection_name: Name of the Qdrant collection to open.

    Returns:
        A ``VectorStoreIndex`` backed by the existing, already-embedded
        collection.

    Note:
        Qdrant's local persistence takes an exclusive lock on ``persist_path``;
        close the client (``index.vector_store.client.close()``) before opening
        another client on the same directory.
    """
    if embed_model is None:
        embed_model = get_embed_model()

    client = qdrant_client.QdrantClient(path=persist_path)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
