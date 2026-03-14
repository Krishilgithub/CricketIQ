"""
RAG Chatbot — Conversational retrieval over cricket stats using LangChain + ChromaDB + Gemini.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import GEMINI_API_KEY, CHROMA_PERSIST_DIR


def build_knowledge_base(conn) -> None:
    """Index Gold-layer stats into ChromaDB for RAG retrieval."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        print("❌ chromadb not installed. pip install chromadb")
        return

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    # Delete existing collection if it exists
    try:
        client.delete_collection("cricket_stats")
    except Exception:
        pass

    collection = client.create_collection(
        name="cricket_stats",
        metadata={"hnsw:space": "cosine"},
    )

    # Index team stats
    teams = conn.execute("""
        SELECT t.team_name, t.is_icc_full_member,
               COUNT(DISTINCT fi.match_id) AS matches,
               ROUND(AVG(fi.total_runs), 1) AS avg_score,
               ROUND(AVG(fi.run_rate), 2) AS avg_rr,
               SUM(fi.fours) AS total_fours,
               SUM(fi.sixes) AS total_sixes
        FROM gold.dim_team t
        LEFT JOIN gold.fact_innings_summary fi ON t.team_key = fi.team_key
        GROUP BY t.team_name, t.is_icc_full_member
        HAVING COUNT(DISTINCT fi.match_id) > 0
    """).fetchall()

    docs, ids, metas = [], [], []
    for i, t in enumerate(teams):
        doc = (
            f"Team: {t[0]}. ICC Full Member: {t[1]}. "
            f"Matches played: {t[2]}. Average score: {t[3]}. "
            f"Average run rate: {t[4]}. Total fours: {t[5]}. Total sixes: {t[6]}."
        )
        docs.append(doc)
        ids.append(f"team_{i}")
        metas.append({"type": "team", "team": t[0]})

    # Index top player stats
    players = conn.execute("""
        SELECT p.player_name, p.teams_played, p.matches_played,
               SUM(b.runs_scored) AS total_runs,
               ROUND(AVG(b.strike_rate), 1) AS avg_sr,
               SUM(b.fours) AS fours, SUM(b.sixes) AS sixes
        FROM gold.dim_player p
        LEFT JOIN gold.fact_batting_innings b ON p.player_key = b.player_key
        GROUP BY p.player_name, p.teams_played, p.matches_played
        HAVING SUM(b.runs_scored) > 0
        ORDER BY total_runs DESC
        LIMIT 200
    """).fetchall()

    for i, p in enumerate(players):
        doc = (
            f"Player: {p[0]}. Teams: {p[1]}. Matches: {p[2]}. "
            f"Total runs: {p[3]}. Average strike rate: {p[4]}. "
            f"Fours: {p[5]}. Sixes: {p[6]}."
        )
        docs.append(doc)
        ids.append(f"player_{i}")
        metas.append({"type": "player", "player": p[0]})

    # Index venue stats
    venues = conn.execute("""
        SELECT venue_name, city, avg_first_innings_score,
               avg_second_innings_score, matches_hosted
        FROM gold.dim_venue
        WHERE matches_hosted >= 3
        ORDER BY matches_hosted DESC
        LIMIT 100
    """).fetchall()

    for i, v in enumerate(venues):
        doc = (
            f"Venue: {v[0]}, City: {v[1]}. "
            f"Average 1st innings score: {v[2]}. Average 2nd innings score: {v[3]}. "
            f"Matches hosted: {v[4]}."
        )
        docs.append(doc)
        ids.append(f"venue_{i}")
        metas.append({"type": "venue", "venue": v[0]})

    collection.add(documents=docs, ids=ids, metadatas=metas)
    print(f"✅ Indexed {len(docs)} documents into ChromaDB")


def get_rag_response(question: str, chat_history: list = None) -> str:
    """
    Answer a cricket question using RAG (ChromaDB retrieval + Gemini generation).
    """
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY not set. Add it to your .env file."

    try:
        import chromadb
        import google.generativeai as genai
    except ImportError:
        return "❌ Required packages not installed. pip install chromadb google-generativeai"

    # Retrieve relevant context
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        collection = client.get_collection("cricket_stats")
    except Exception:
        return "⚠️ Knowledge base not built. Click 'Build Knowledge Base' first."

    results = collection.query(query_texts=[question], n_results=10)
    context = "\n".join(results["documents"][0]) if results["documents"] else ""

    if not context:
        return "No relevant information found in the knowledge base."

    # Build prompt
    history_str = ""
    if chat_history:
        for msg in chat_history[-5:]:
            history_str += f"\n{msg['role']}: {msg['content']}"

    prompt = f"""You are a cricket statistics expert assistant. Answer the question using ONLY the provided context.
If the answer isn't in the context, say "I don't have enough data to answer that."

Context:
{context}

{f"Chat History:{history_str}" if history_str else ""}

Question: {question}

Answer concisely and accurately with numbers when available:"""

    # Generate with Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text


if __name__ == "__main__":
    from src.warehouse.schema import get_connection

    conn = get_connection()
    build_knowledge_base(conn)
    conn.close()

    # Test query
    answer = get_rag_response("Which team has the highest average score?")
    print(f"\nAnswer: {answer}")
