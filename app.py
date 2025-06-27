# app.py (Final Production Version)

import streamlit as st
import os
from main import load_conversational_chain, process_conversational_query
from knowledge_base import build_knowledge_base

CHROMA_DB_PATH = "chroma/"

st.set_page_config(page_title="Conversational AI Chatbot", layout="wide")
st.title("📄 Conversational AI Chatbot")
st.subheader("An intelligent chatbot with memory, trained on your documents")

# --- Secrets and API Key Handling ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("GOOGLE_API_KEY not found. Please set it in your Streamlit secrets.")
    st.stop()

# --- Knowledge Base Build ---
# Check if the knowledge base has been built, if not, build it.
if not os.path.exists(CHROMA_DB_PATH):
    with st.spinner("Building knowledge base... This may take a few minutes on first startup."):
        build_knowledge_base(google_api_key)

# --- Session State and Chain Loading ---
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_conversational_chain():
    return load_conversational_chain(google_api_key)

chain = get_conversational_chain()

# --- Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question about your documents:"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = process_conversational_query(
                chain,
                user_question,
                st.session_state.messages[:-1]
            )
            answer = result["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})