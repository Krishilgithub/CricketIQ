"""
GenAI Dashboard — Chat interface, NL-to-SQL, document extraction.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_genai_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>💬 GenAI — Cricket Intelligence</h1>
        <p>RAG chatbot, natural language queries, document extraction</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "💬 Stats Chatbot", "🔍 NL-to-SQL", "📄 Document Extractor"
    ])

    with tab1:
        _render_chatbot()
    with tab2:
        _render_nl_to_sql()
    with tab3:
        _render_doc_extractor()


def _render_chatbot():
    st.markdown("### 💬 Cricket Stats Chatbot (RAG)")
    st.info("Ask questions about teams, players, venues, and stats. Powered by ChromaDB + Gemini.")

    # Build knowledge base button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔨 Build Knowledge Base", key="build_kb"):
            with st.spinner("Indexing stats into ChromaDB..."):
                try:
                    from src.warehouse.schema import get_connection
                    from src.genai.rag_chatbot import build_knowledge_base

                    conn = get_connection()
                    build_knowledge_base(conn)
                    conn.close()
                    st.success("✅ Knowledge base built!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    question = st.chat_input("Ask about cricket stats...", key="rag_input")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from src.genai.rag_chatbot import get_rag_response
                    answer = get_rag_response(question, st.session_state.chat_history)
                except Exception as e:
                    answer = f"Error: {e}"

                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})


def _render_nl_to_sql():
    st.markdown("### 🔍 Natural Language to SQL")
    st.info("Type a question in plain English and get SQL + results from the warehouse.")

    question = st.text_input(
        "Your question",
        placeholder="e.g., Which team has the highest win rate in T20 World Cups?",
        key="nl_sql_q",
    )

    example_questions = [
        "Top 10 run scorers in T20 Internationals",
        "Average score at each venue with more than 10 matches",
        "Which bowler has the best economy rate?",
        "Team win percentage when batting first",
        "Most sixes hit by a player in a single innings",
    ]

    st.markdown("**Example questions:**")
    for eq in example_questions:
        if st.button(eq, key=f"eq_{eq[:20]}"):
            question = eq

    if question and st.button("🔍 Query", key="nl_sql_btn"):
        with st.spinner("Generating and executing SQL..."):
            try:
                from src.genai.nl_to_sql import nl_to_sql
                result = nl_to_sql(question)

                st.markdown("**Generated SQL:**")
                st.code(result["sql"], language="sql")

                if result["error"]:
                    st.error(f"Execution error: {result['error']}")
                elif result["result"] is not None:
                    st.markdown("**Results:**")
                    st.dataframe(result["result"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error: {e}")


def _render_doc_extractor():
    st.markdown("### 📄 Document Information Extractor")

    extraction_type = st.selectbox(
        "Extraction type",
        ["summary", "stats", "players", "match_report"],
        key="doc_ext_type",
    )

    input_method = st.radio(
        "Input method", ["Paste Text", "Enter URL", "Upload PDF"],
        key="doc_input_method", horizontal=True,
    )

    if input_method == "Paste Text":
        text = st.text_area("Paste text here", height=200, key="doc_text")
        if text and st.button("Extract", key="doc_ext_btn"):
            with st.spinner("Extracting..."):
                try:
                    from src.genai.document_extractor import extract_from_text
                    result = extract_from_text(text, extraction_type)
                    st.markdown("**Extracted Information:**")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif input_method == "Enter URL":
        url = st.text_input("URL", placeholder="https://...", key="doc_url")
        if url and st.button("Extract from URL", key="doc_url_btn"):
            with st.spinner("Fetching and extracting..."):
                try:
                    from src.genai.document_extractor import extract_from_url
                    result = extract_from_url(url, extraction_type)
                    st.markdown("**Extracted Information:**")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif input_method == "Upload PDF":
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="doc_pdf")
        if uploaded and st.button("Extract from PDF", key="doc_pdf_btn"):
            # Save temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("Extracting from PDF..."):
                try:
                    from src.genai.document_extractor import extract_from_pdf
                    result = extract_from_pdf(tmp_path, extraction_type)
                    st.markdown("**Extracted Information:**")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Error: {e}")
