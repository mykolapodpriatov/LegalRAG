# LegalRAG

LegalRAG is a domain-specific legal assistant built on Retrieval-Augmented Generation (RAG).
It retrieves and aggregates information from legal documents (judicial decisions, laws, contracts,
commentaries) and answers questions with references back to the source text.

> **Status:** Working prototype — document ingestion, multilingual retrieval and answer
> generation are implemented. Items under *Roadmap* are planned, not yet built.

## Features
- **Multilingual retrieval** — multilingual E5 embeddings, so queries and documents may be in different languages.
- **Source-grounded answers** — responses are generated from retrieved passages.
- **Streamlit UI** — upload documents, build the index, and ask questions.

## Architecture
- **Framework:** LlamaIndex
- **LLM:** Anthropic Claude 3.5 Sonnet
- **Embeddings:** `intfloat/multilingual-e5-large` (HuggingFace)
- **Vector store:** Qdrant (persisted to disk by default; `QDRANT_PATH` overrides the location, in-memory in the offline demo script)
- **UI:** Streamlit

## Roadmap (planned)
- Legal entity extraction / term normalization (e.g. Legal-BERT)
- Richer document parsing (unstructured.io / LlamaParse)
- Automated quality evaluation (e.g. Open RAG Eval / Ragas)

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your_key"
streamlit run app.py
```

## Try it offline
Smoke-test the retrieval pipeline with no Anthropic API key, no ~2 GB embedding
model and no torch. The demo indexes a tiny hardcoded legal corpus with a
lightweight `MockEmbedding` and prints the top passages for a sample query:
```bash
pip install llama-index-core llama-index-vector-stores-qdrant qdrant-client
python -m scripts.demo_retrieval
```
