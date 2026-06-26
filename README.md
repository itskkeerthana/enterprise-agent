# DocuMind: Agentic RAG Engine 
# 💼 Enterprise Document Intelligence Agent

A modern RAG (Retrieval-Augmented Generation) application that allows users to upload enterprise text or PDF documents and conduct contextual multi-turn conversations using agentic AI.

## 🚀 Features
* **Multi-Format Processing:** Seamlessly processes both `.txt` and `.pdf` document architectures.
* **Agentic Framework:** Powered by `LangGraph` and `LangChain` to selectively query local document embeddings.
* **Conversational Memory:** Maintains complete conversational state context across multi-turn interactions.
* **Cost-Efficient Stack:** Utilizing Google's free tier `gemini-2.5-flash` model and `ChromaDB` for local vector storage.

## 🛠️ Tech Stack
* **Frontend UI:** Streamlit
* **Orchestration:** LangChain / LangGraph
* **LLM & Embeddings:** Google Gemini API (`gemini-2.5-flash`, `gemini-embedding-2-preview`)
* **Vector Store:** ChromaDB

## 📦 Local Installation & Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install required libraries: `pip install streamlit langchain-google-genai langchain-community langchain-text-splitters langgraph pypdf chromadb python-dotenv`
4. Create a `.env` file in the root directory and add your API key: `GOOGLE_API_KEY=your_key_here`
5. Launch the application: `streamlit run app.py`