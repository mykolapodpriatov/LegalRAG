import logging
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

def load_documents(data_dir: str):
    """Loads documents using unstructured.io / SimpleDirectoryReader."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return []
    try:
        reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
        documents = reader.load_data()
        return documents
    except ValueError as e:
        # SimpleDirectoryReader raises ValueError both for an empty directory
        # (expected -> empty corpus) and for genuine parse/read failures. Log so
        # the latter are not silently reported to the user as "No documents found".
        logging.warning("SimpleDirectoryReader failed for %s: %s", data_dir, e)
        return []

def chunk_documents(documents):
    """Chunks documents."""
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes
