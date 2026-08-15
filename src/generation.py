import os


def get_query_engine(index, llm=None):
    """Creates a query engine using Claude 3.5 Sonnet.

    Args:
        index: A LlamaIndex index exposing ``as_query_engine``.
        llm: Language model to use. Defaults to Anthropic Claude 3.5 Sonnet
            when omitted. Injecting a lightweight model (e.g. ``MockLLM``)
            lets callers build an engine without an API key or a network
            call, which is what tests rely on.
    """
    if llm is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        from llama_index.llms.anthropic import Anthropic

        llm = Anthropic(model="claude-3-5-sonnet-20241022", api_key=api_key)

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        response_mode="compact",
    )


def generate_response(query_engine, query_text):
    """Generates a response for a query."""
    return query_engine.query(query_text)
