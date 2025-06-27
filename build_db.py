# build_db.py

from ingest import load_and_split_documents
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.config import Settings
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()

# --- Configuration ---
CHROMA_DB_PATH = "chroma/"
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)


def build_and_persist_db():
    """Builds the vector database from documents and persists it."""
    if os.path.exists(CHROMA_DB_PATH):
        print("Chroma database already exists. No need to rebuild.")
        return

    print("No existing vector database found. Building a new one...")

    # Load and split documents
    chunks = load_and_split_documents()
    if not chunks:
        print("No documents to process. Shutting down.")
        return

    # Instantiate the embedding model
    # Make sure to provide your API key if it's not in the environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    # Create a new Chroma database from the document chunks
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_DB_PATH,
        client_settings=CHROMA_SETTINGS
    )
    print("Successfully created and persisted the vector database.")


if __name__ == "__main__":
    build_and_persist_db()