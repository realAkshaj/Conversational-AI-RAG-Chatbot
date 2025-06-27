# Conversational AI Document Chatbot with RAG and Gemini

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://conversational-ai-rag-chatbot-fwuwydrqbyapeya8l2ea8u.streamlit.app/)

**[► View the Live Demo](https://conversational-ai-rag-chatbot-fwuwydrqbyapeya8l2ea8u.streamlit.app/)**

---

This project is a sophisticated, full-stack conversational AI chatbot built with Python, LangChain, and Google's Gemini models. It uses a Retrieval-Augmented Generation (RAG) architecture to answer questions about a specific set of documents, ensuring that its responses are accurate, context-aware, and grounded in the provided source material.

The application features a user-friendly web interface built with Streamlit and maintains conversational memory, allowing for natural, multi-turn follow-up questions.

![Chatbot Demo](https://storage.googleapis.com/garden-prod/doc-images/image_a17939.png-c9a3f34b-5e60-46b3-b726-b4e45eed5fa4)

---

## Key Features

- **Retrieval-Augmented Generation (RAG):** The core of the project. The chatbot minimizes "hallucinations" by retrieving relevant information from a specialized vector database before formulating an answer.
- **Powered by Google Gemini:** Leverages the `gemini-1.5-flash-latest` model for high-quality language understanding and generation, and `models/embedding-001` for creating semantic vector representations of text.
- **Conversational Memory:** Remembers the context of previous turns in the conversation, allowing for natural follow-up questions (e.g., "Can you explain that in more detail?").
- **Interactive Web UI:** A clean and intuitive chat interface built with Streamlit, making the AI accessible and easy to use.
- **Vector Database for Semantic Search:** Uses ChromaDB to store document embeddings and perform efficient similarity searches, allowing the chatbot to find the most relevant document chunks for any given query.
- **Cross-Platform Compatibility:** Includes specific configurations to ensure the application runs seamlessly on both local (Windows/Mac) and deployed (Linux) environments.
- **Source Verification:** The chatbot can be configured to provide the source documents and page numbers for the information used in its answers, ensuring transparency and trustworthiness.

---

## How It Works

The application follows a modular, multi-stage process:

1.  **Data Ingestion & Indexing (Offline):**
    * The `knowledge_base.py` script loads documents (e.g., PDFs) from the `documents/` directory.
    * Documents are split into smaller, manageable chunks.
    * Each chunk is converted into a numerical vector (embedding) using the Gemini embedding model.
    * These embeddings are stored and indexed in a local ChromaDB vector database, creating a persistent knowledge base.

2.  **Conversational Loop (Live):**
    * The Streamlit app (`app.py`) presents the chat interface.
    * A user asks a question.
    * The `ConversationalRetrievalChain` takes the user's question and the previous chat history to formulate a new, standalone question.
    * The retriever queries the ChromaDB to find the most semantically relevant document chunks based on this new question.
    * The retrieved chunks, along with the user's question and chat history, are passed as context to the Gemini LLM.
    * The LLM generates a comprehensive, source-grounded answer, which is then displayed to the user.

---

## Technology Stack

- **Backend:** Python
- **AI Framework:** LangChain
- **LLM & Embeddings:** Google Gemini
- **Vector Database:** ChromaDB
- **Web Framework:** Streamlit
- **Document Loading:** PyPDF
- **Environment Management:** `python-dotenv`, `pysqlite3-binary` (for Linux compatibility)

---

## How to Run This Project

### Local Development

#### Prerequisites

-   Python 3.9+
-   Git
-   A Google API key with the Gemini API enabled.

#### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/realAkshaj/Conversational-AI-RAG-Chatbot.git](https://github.com/realAkshaj/Conversational-AI-RAG-Chatbot.git)
    cd Conversational-AI-RAG-Chatbot
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    This command will install all necessary packages. Note that it may show an error for `pysqlite3-binary` on Windows, which is expected and can be ignored.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Local API Keys (Secrets):**
    -   Create a directory named `.streamlit` in the project root.
    -   Inside the `.streamlit` directory, create a file named `secrets.toml`.
    -   Add your Google API key to this file in the following format:
        ```toml
        GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
        ```

5.  **Add Your Documents:**
    -   Place the PDF files you want the chatbot to learn from inside the `documents/` folder.

6.  **Build the Knowledge Base:**
    -   Run the ingestion script once to create your vector database.
    ```bash
    python knowledge_base.py
    ```

7.  **Run the Web Application:**
    ```bash
    streamlit run app.py
    ```
    Your browser should automatically open with the running chat application.

### Deployment on Streamlit Cloud

1.  **Push to GitHub:** Ensure your repository is up-to-date with the latest code.
2.  **Sign up for Streamlit Community Cloud:** Use your GitHub account to sign up.
3.  **Deploy New App:**
    -   Click "New app" from your workspace.
    -   Select your repository and the `main` branch.
    -   Ensure the "Main file path" is `app.py`.
4.  **Add Your Secrets:**
    -   After clicking "Deploy!", immediately go to the app's settings (Manage app -> Settings -> Secrets).
    -   Paste your `GOOGLE_API_KEY` in the secrets manager using the same format as your local `secrets.toml` file:
        ```toml
        GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
        ```
    -   Save the secret. The app will reboot and use this key to connect to the Google API.
