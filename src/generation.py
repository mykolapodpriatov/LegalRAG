import os
from llama_index.llms.anthropic import Anthropic

def get_query_engine(index):
    """Creates a query engine using Claude 3.5 Sonnet."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        
    llm = Anthropic(model="claude-3-5-sonnet-20241022", api_key=api_key)
    
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        response_mode="compact"
    )
    return query_engine

def generate_response(query_engine, query_text):
    """Generates a response for a query."""
    response = query_engine.query(query_text)
    return response
