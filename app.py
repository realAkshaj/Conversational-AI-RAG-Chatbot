# app.py (Final version with Chat History and Memory)

import streamlit as st
from main import load_conversational_chain, process_conversational_query
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Conversational AI Chatbot", layout="wide")
st.title("📄 Conversational AI Chatbot")
st.subheader("An intelligent chatbot with memory, trained on your documents")

# --- Session State Initialization ---
# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Conversational Chain Loading ---
# We now load the new conversational chain
@st.cache_resource
def get_conversational_chain():
    return load_conversational_chain()

chain = get_conversational_chain()

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Handling ---
if user_question := st.chat_input("Ask a question about your documents:"):

    # Add user's question to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Process the query in an 'assistant' context
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # We now pass the existing chat history to the processing function
            result = process_conversational_query(
                chain,
                user_question,
                st.session_state.messages[:-1] # Pass all but the last message (the user's new question)
            )

            # Display the main answer
            answer = result["answer"]
            st.markdown(answer)

            # Add the assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})