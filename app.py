# app.py

import streamlit as st
from main import load_conversational_chain, process_conversational_query

# No longer need dotenv
# from dotenv import load_dotenv
# load_dotenv()

st.set_page_config(page_title="Conversational AI Chatbot", layout="wide")
st.title("📄 Conversational AI Chatbot")
st.subheader("An intelligent chatbot with memory, trained on your documents")

# Use st.secrets to get the API key
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("GOOGLE_API_KEY not found. Please set it in your Streamlit secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_conversational_chain():
    # Pass the API key to the loading function
    return load_conversational_chain(google_api_key)

chain = get_conversational_chain()

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