# main.py (Upgraded with Conversational Memory)
# --- START: CHROMA DB DEPLOYMENT FIX ---
# This is a workaround for a known issue with ChromaDB on certain environments,
# including Streamlit Cloud. It forces ChromaDB to use the correct sqlite3 library.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END: CHROMA DB DEPLOYMENT FIX --

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from chromadb.config import Settings

# --- Configuration (remains the same) ---
CHROMA_DB_PATH = "chroma/"
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

def load_conversational_chain():
    """
    Loads and initializes a conversational RAG chain with memory.

    Returns:
        ConversationalRetrievalChain: The initialized conversational chain.
    """
    # Load the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the existing vector database
    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )

    # Create a retriever
    retriever = db.as_retriever()

    # Create the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.5  # Slightly more creative for conversation
    )

    # Set up memory
    # "chat_history" is where the memory will store messages.
    # "return_messages=True" ensures the history is a list of message objects.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create and return the Conversational Retrieval Chain
    # This chain is designed to use a retriever and memory to hold a conversation.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return chain

def process_conversational_query(chain, query, chat_history):
    """
    Processes a user query using the conversational chain.

    Args:
        chain (ConversationalRetrievalChain): The conversational chain.
        query (str): The user's question.
        chat_history (list): The list of past messages.

    Returns:
        dict: The result from the conversational chain.
    """
    result = chain.invoke({"question": query, "chat_history": chat_history})
    return result