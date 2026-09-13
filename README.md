# 🎓 College Policy RAG Chatbot

A fully local **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from college policy documents, academic calendars, examination schedules, and other institutional PDFs.

The project uses **Llama 3.2 3B** for answer generation, **nomic-embed-text** for embeddings, **ChromaDB** as the vector database, **LangChain** for the RAG pipeline, and **Streamlit** for the web interface.

The entire system runs locally using **Ollama**, so no external LLM or embedding API is required.

---

## ✨ Features

- 📄 Supports multiple college PDF documents
- 🔄 Incremental PDF ingestion
- ✂️ Automatic document chunking
- 🧠 Local semantic embeddings
- 🔎 Semantic similarity search with ChromaDB
- 🤖 Local LLM inference using Llama 3.2 3B
- 📚 Displays retrieved document sources and pages
- 💬 Chat-style Streamlit interface
- 🛡️ Answers are generated using retrieved document context
- 📊 Retrieval evaluation using Recall@K and MRR
- 💻 Completely local — no external AI API required

---

## 🏗️ Architecture

```text
                     College PDF Documents
                              │
                              ▼
                     ┌─────────────────┐
                     │   PyPDFLoader   │
                     │  PDF Extraction │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  Text Splitter  │
                     │ RecursiveCharacter
                     │ Text Splitter   │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   Embeddings    │
                     │ nomic-embed-text│
                     │     Ollama      │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │    ChromaDB     │
                     │  Vector Store   │
                     └────────┬────────┘
                              │
                              │
                        User Question
                              │
                              ▼
                     ┌─────────────────┐
                     │    Retriever    │
                     │     Top-K=4     │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   Llama 3.2 3B  │
                     │      Ollama     │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Grounded Answer │
                     │  + Source Info  │
                     └─────────────────┘
```

## 🚨 Before running the project, make sure you have:

- Python 3.10+
- Ollama
- Git
- At least a few GB of free disk space for the models
