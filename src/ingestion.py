import logging
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


def _directory_has_readable_files(data_dir: str) -> bool:
    """Return True if ``data_dir`` holds at least one non-hidden file.

    Walks ``data_dir`` recursively while skipping hidden files and directories,
    mirroring ``SimpleDirectoryReader``'s default ``exclude_hidden=True``. This
    lets us recognise an empty corpus without invoking (and catching an error
    from) the reader.
    """
    for _root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if any(not name.startswith(".") for name in files):
            return True
    return False


def load_documents(data_dir: str):
    """Load documents from ``data_dir`` with LlamaIndex's SimpleDirectoryReader.

    Behavior:
        * Missing path: the directory is created and ``[]`` is returned.
        * Path is a file: raises :class:`NotADirectoryError` with a clear
          message, instead of the opaque ``ValueError`` the reader would emit.
        * Empty directory: returns ``[]`` -- a valid, empty corpus.
        * Non-empty directory the reader cannot parse: the underlying error is
          logged and re-raised, so genuine parse failures are not silently
          reported to the user as "No documents found".

    Args:
        data_dir: Path to the directory containing source documents.

    Returns:
        A list of loaded ``Document`` objects (possibly empty).

    Raises:
        NotADirectoryError: If ``data_dir`` exists but is not a directory.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return []

    if not os.path.isdir(data_dir):
        raise NotADirectoryError(
            f"data_dir must be a directory, but '{data_dir}' is a file."
        )

    # An empty directory is an expected empty corpus, not a failure. Detecting
    # it up front reserves the exception path below for genuine parse errors.
    if not _directory_has_readable_files(data_dir):
        logging.info("No documents found in %s; treating as an empty corpus.", data_dir)
        return []

    try:
        reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
        return reader.load_data()
    except ValueError as e:
        # The directory is non-empty yet the reader failed: a genuine
        # parse/read failure, distinct from an empty corpus. Surface it rather
        # than masquerading as "no documents found".
        logging.error("Failed to parse documents in %s: %s", data_dir, e)
        raise


def chunk_documents(documents):
    """Chunk documents into overlapping nodes for indexing."""
    parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes
