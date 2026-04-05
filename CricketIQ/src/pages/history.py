"""Chat History page — Browse, search, and re-run past conversations."""
import streamlit as st
import pandas as pd


def render():
    st.title("💬 Chat History")
    st.markdown("Review your previous conversations, generated SQL queries, and AI insights.")

    sessions = st.session_state.get("sessions", {})

    col_new, col_search = st.columns([1, 3])
    with col_new:
        if st.button("➕ New Conversation", type="primary"):
            import uuid
            new_id = str(uuid.uuid4())
            st.session_state["current_session_id"] = new_id
            st.session_state["sessions"][new_id] = {
                "title": "New Chat",
                "timestamp": pd.Timestamp.now(),
                "messages": [],
            }
            st.rerun()
    with col_search:
        search_term = st.text_input("🔍 Search conversations...", placeholder="Search by topic, team, player...")

    st.markdown("---")

    if not sessions:
        st.info("No chat history available. Go to the **AI Chat Bot** page to start asking questions!")
        return

    sorted_sessions = sorted(
        sessions.items(),
        key=lambda x: x[1].get("timestamp", pd.Timestamp.min),
        reverse=True,
    )

    # Filter by search term
    if search_term:
        sorted_sessions = [
            (sid, s) for sid, s in sorted_sessions
            if search_term.lower() in s.get("title", "").lower()
            or any(search_term.lower() in m.get("content", "").lower() for m in s.get("messages", []))
        ]

    if not sorted_sessions:
        st.warning("No conversations match your search.")
        return

    total_chats = sum(1 for _, s in sorted_sessions if s.get("messages"))
    st.caption(f"Showing {total_chats} conversation(s)")

    for sess_id, s_data in sorted_sessions:
        msgs = s_data.get("messages", [])
        if not msgs:
            continue

        is_current = sess_id == st.session_state.get("current_session_id")
        badge = "🟢 **Active**  " if is_current else ""
        ts = s_data.get("timestamp")
        ts_str = ts.strftime("%d %b %Y, %H:%M") if ts else "Unknown"
        user_msgs = [m for m in msgs if m["role"] == "user"]
        tool_msgs = [m for m in msgs if m["role"] == "tool"]

        with st.expander(
            f"{badge}📝 {s_data['title']}  —  {ts_str}  |  {len(user_msgs)} questions, {len(tool_msgs)} SQL queries",
            expanded=is_current,
        ):
            col_load, col_del = st.columns([1, 1])
            with col_load:
                if not is_current:
                    if st.button("📥 Load Chat", key=f"load_{sess_id}"):
                        st.session_state["current_session_id"] = sess_id
                        st.success("Chat loaded! Switch to **AI Chat Bot** to continue.")
                        st.rerun()
                else:
                    st.success("✅ This is your active conversation.")

            st.markdown("---")
            for msg in msgs:
                if msg["role"] == "user":
                    st.markdown(f"**🧑 You:** {msg['content']}")
                elif msg["role"] == "tool":
                    with st.container():
                        st.markdown("**🛠️ SQL Executed:**")
                        st.code(msg["content"], language="sql")
                        if "result_df" in msg and msg["result_df"]:
                            try:
                                df_display = pd.DataFrame(msg["result_df"])
                                st.dataframe(df_display.head(10), use_container_width=True, hide_index=True)
                            except Exception:
                                st.caption(msg.get("result", ""))
                        elif "result" in msg:
                            st.caption(msg["result"][:500] + "..." if len(msg.get("result", "")) > 500 else msg.get("result", ""))
                elif msg["role"] == "assistant":
                    st.markdown(f"**🤖 CricketIQ:** {msg['content']}")
                    st.markdown("---")
