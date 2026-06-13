import streamlit as st

from src.chat_engine import ask_agent

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

question = st.chat_input(
    "Ask me anything..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):

        answer, sources = ask_agent(question)

    with st.chat_message("assistant"):

        st.markdown(answer)

        st.markdown("### Sources")

        for source in sources[:5]:

            st.write(source)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )