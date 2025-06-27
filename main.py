# main.py

# --- START: CHROMA DB DEPLOYMENT FIX ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END: CHROMA DB DEPLOYMENT FIX ---

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from chromadb.config import Settings

CHROMA_DB_PATH = "chroma/"
CHROMA_SETTINGS = Settings(anonymized_telemetry=False, is_persistent=True)

def load_conversational_chain(api_key): # <-- Pass API key as an argument
    """Loads and initializes a conversational RAG chain with memory."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key # <-- Use the passed key
    )

    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )

    retriever = db.as_retriever()

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.5,
        google_api_key=api_key # <-- Use the passed key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return chain

def process_conversational_query(chain, query, chat_history):
    """Processes a user query using the conversational chain."""
    result = chain.invoke({"question": query, "chat_history": chat_history})
    return result