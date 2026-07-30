import uuid
import streamlit as st
from agent.graph import app

# Page Config
st.set_page_config(
    page_title="Pakistan Law AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Pakistan Penal Code 1860 AI Assistant")
st.caption("Ask anything from the Pakistan Law PDF")

# Initialize Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📄 Document")
    st.success("Loaded Successfully")
    st.write("**PDF Name:** pakistanlaw.pdf")
    st.write("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        # Reset thread_id to clear LangGraph agent memory state
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Re-render saved sources for past assistant turns
        if message.get("sources"):
            with st.expander("📚 Sources & References"):
                for src in message["sources"]:
                    st.markdown(
                        f"- **Doc:** `{src.get('source', 'N/A')}` | "
                        f"**Ch. {src.get('chapter_number', 'N/A')}:** {src.get('chapter_name', 'N/A')} | "
                        f"**Page:** {src.get('page', 'N/A')}"
                    )

# User Input
question = st.chat_input("Ask anything about Pakistan Law...")

if question:
    # Append & Display User Message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke(
                {"question": question},
                config={
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }
            )

            answer = result.get("answer", "")
            raw_sources = result.get("sources", [])

            # Deduplicate sources
            shown_keys = set()
            unique_sources = []
            
            for source in raw_sources:
                key = (
                    source.get("source"),
                    source.get("page"),
                    source.get("chapter_number")
                )
                if key not in shown_keys:
                    shown_keys.add(key)
                    unique_sources.append(source)

            # Render Answer
            st.markdown(answer)

            # Render Sources in collapsible expander
            if unique_sources:
                with st.expander("📚 Sources & References"):
                    for src in unique_sources:
                        st.markdown(
                            f"- **Doc:** `{src.get('source', 'N/A')}` | "
                            f"**Ch. {src.get('chapter_number', 'N/A')}:** {src.get('chapter_name', 'N/A')} | "
                            f"**Page:** {src.get('page', 'N/A')}"
                        )

    # Save Assistant Response with Sources to Session State
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": unique_sources
        }
    )