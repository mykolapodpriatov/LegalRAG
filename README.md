# LegalRAG

LegalRAG is a domain-specific legal assistant based on Retrieval-Augmented Generation (RAG).
It is designed to extract and aggregate information from judicial decisions, laws, contracts, and legal commentaries across different languages.

## Features
- **Entity Extraction**: Normalization of legal terms and entity extraction using Legal-BERT.
- **Multilingual Support**: Supports queries and answers in multiple languages.
- **Source Verification**: Responses strictly reference source documents.
- **Evaluation**: Uses Open RAG Eval for quality assessment.

## Architecture
- **Framework**: LlamaIndex
- **Data Parsing**: unstructured.io, LlamaParse
- **Entity Extraction**: Legal-BERT / LLM
- **LLM**: Claude 3.5 Sonnet
- **Embeddings**: E5-multilingual (HuggingFace)
- **Vector DB**: Qdrant
- **UI**: Streamlit

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`
