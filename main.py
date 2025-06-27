import os
from dotenv import load_dotenv

# Import LangChain components
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import our document ingestion function from the other file
from ingest import load_and_split_documents

from chromadb.config import Settings

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CHROMA_DB_PATH = "chroma/"
DOCUMENT_PATH = "documents/"

# This dictionary contains the settings for ChromaDB.
# By setting anonymized_telemetry to False, we disable the telemetry collection.
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)


def main():
    """
    Main function to set up and run the RAG-based QA chatbot.
    """
    # --- 1. SETUP EMBEDDING MODEL & VECTOR DATABASE ---

    print("Setting up Gemini embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # This part for creating/loading the DB is correct and doesn't need changes.
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"No existing vector database found. Creating a new one at: {CHROMA_DB_PATH}")
        chunks = load_and_split_documents()
        if not chunks:
            print("Shutting down, no documents to process.")
            return
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=CHROMA_DB_PATH,
            client_settings=CHROMA_SETTINGS
        )
        print("Successfully created and persisted the vector database.")
    else:
        print(f"Loading existing vector database from: {CHROMA_DB_PATH}")
        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS
        )

    # --- 2. SETUP RETRIEVER, LLM, AND QA CHAIN ---

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Instantiate the Gemini Pro chat model with the updated name and removed argument
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",  # <--- CHANGE HERE: Use the latest model name
        temperature=0.3
    )  # <--- CHANGE HERE: Removed the deprecated 'convert_system_message_to_human'

    template = """
    You are a helpful assistant. Use the following pieces of context to answer the question at the end.
    If you don't know the answer from the context provided, just say that you don't know. Do not try to make up an answer.
    Keep the answer concise.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # --- 3. INTERACTIVE QUESTION-ANSWERING LOOP ---

    print("\n--- Chatbot is ready! ---")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            break

        # Run the QA chain with the user's query using the modern .invoke() method
        result = qa_chain.invoke(query)  # <--- CHANGE HERE: Use .invoke() instead of .__call__()

        print("\nAnswer:", result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            # Use .get() for safer dictionary access
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            print(f"- {source}, page {page}")


if __name__ == "__main__":
    main()