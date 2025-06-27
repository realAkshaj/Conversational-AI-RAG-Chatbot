# knowledge_base.py

import os
import sys
from ingest import load_and_split_documents
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.config import Settings

# --- START: CHROMA DB DEPLOYMENT FIX ---
if sys.platform == "linux":
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- END: CHROMA DB DEPLOYMENT FIX ---

CHROMA_DB_PATH = "chroma/"
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

def build_knowledge_base(api_key):
    """Builds the vector database from documents and persists it."""
    print("Building knowledge base...")

    chunks = load_and_split_documents()
    if not chunks:
        print("No documents to process. Shutting down.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DB_PATH,
        client_settings=CHROMA_SETTINGS
    )
    print("Successfully built and persisted the knowledge base.")