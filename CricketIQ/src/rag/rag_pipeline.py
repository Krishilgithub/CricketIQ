"""
src/rag/rag_pipeline.py
───────────────────────
Orchestrates entity retrieval for the Agent's context.
"""

from langsmith import traceable
from src.rag.retriever import get_con, extract_entities
from src.rag.prompt_builder import build_agent_system_prompt

@traceable(run_type="chain", name="RAG Pipeline Orchestrator")
def gather_query_context(query: str) -> str:
    """Orchestrates DB retrieval to create the system prompt."""
    con = get_con()
    entities = extract_entities(query, con)
    if con is not None:
        try:
            con.close()
        except:
            pass
    
    return build_agent_system_prompt(entities)
