"""
src/observability/langsmith_tracing.py
──────────────────────────────────────
LangSmith observability configuration and helpers.
"""

from langsmith import Client, traceable

# Initialize LangSmith client
# (Automatically picks up LANGCHAIN_API_KEY, LANGCHAIN_PROJECT from environment)
ls_client = Client()

def get_tracer():
    """Retrieve the LangSmith client instance."""
    return ls_client
