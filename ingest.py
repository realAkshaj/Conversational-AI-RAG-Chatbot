import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file, specifically the OpenAI API key
load_dotenv()
print("Loaded API Key.")

# Define the path to the directory containing the PDF documents
DATA_PATH = "documents/"


def load_and_split_documents():
    """
    Loads documents from the specified directory and splits them into smaller chunks.

    Returns:
        list: A list of document chunks.
    """
    # Initialize a DirectoryLoader to load PDF documents from the specified path
    # It uses PyPDFLoader for each PDF file found.
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)

    # Load the documents from the directory
    documents = loader.load()
    if not documents:
        print("No documents found. Please add some PDFs to the 'documents' folder.")
        return []

    print(f"Loaded {len(documents)} documents.")

    # Initialize a text splitter.
    # RecursiveCharacterTextSplitter is good for generic text. It tries to split
    # on characters like newlines and spaces to keep related text together.
    # chunk_size is the max number of characters in a chunk.
    # chunk_overlap keeps some text between chunks to maintain context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the loaded documents into smaller chunks
    print("Splitting documents into chunks...")
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    return texts


# Main execution block
if __name__ == "__main__":
    # Call the function to load and split the documents
    chunks = load_and_split_documents()

    # As a check, print the content of the first chunk
    if chunks:
        print("\n--- Sample Chunk ---")
        print(chunks[0].page_content)
        print("--------------------")