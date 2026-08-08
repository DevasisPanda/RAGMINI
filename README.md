# TinyRAG Desktop v2.0 — Local PDF Question Answering & Comparison System

**Author**: Devasis Panda (<devasis.stu.work@gmail.com>)  
**GitHub**: [@DevasisPanda](https://github.com/DevasisPanda)  
**License**: MIT  

A modern, high-performance local Python desktop application built on top of **TinyRAG**, **FastEmbed**, **Qdrant Cloud**, and **CustomTkinter**. It provides intelligent PDF document question answering, page-level citation tracking, automatic LLM failover (OpenRouter $\rightarrow$ Google Gemini), and intelligent multi-document comparison retrieval.

---

## ✨ Features

- 🖥️ **Modern Desktop GUI**: Built using CustomTkinter with native drag-and-drop PDF ingestion, live logging terminal, and interactive citation cards.
- ⚖️ **Intelligent Multi-Document Comparison Retrieval**: Automatically detects comparison intent (e.g., *"Compare D.K. Basu and Ashwani Kumar"*) and retrieves balanced context across multiple cases.
- ⚡ **Local Embeddings**: High-speed ONNX vector embeddings powered by `BAAI/bge-small-en-v1.5` (384 dimensions) via **FastEmbed**.
- 🗄️ **Qdrant Vector Store**: Cloud and local vector indexing with idempotent deterministic point IDs and clean collection lifecycle management.
- 🔄 **Automatic LLM Failover**: Primary completions via **OpenRouter API** with automatic fallback to **Google Gemini API** on rate limits (HTTP 429) or timeouts.
- ☁️ **Google Drive Import**: Paste public Google Drive PDF sharing links to download and index directly.
- 📌 **Page-Level Citations**: Accurate page number tracking with expandable retrieved text snippets.
- 🛑 **Strict Unknown Detection**: Prevents hallucinations by returning unanswerable notices when context is missing or low-scoring ($\text{sim} < 0.35$).

---

## 🏗️ Architecture Overview

```text
                               ┌────────────────────────────────┐
                               │  Desktop GUI (CustomTkinter)   │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │       Backend Controller       │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
 ┌──────────────────────────┐                                    ┌──────────────────────────┐
 │  Multi-Doc Comparison    │                                    │  Standard RAG Pipeline   │
 └─────────────┬────────────┘                                    └─────────────┬────────────┘
               │                                                               │
               ▼                                                               ▼
 ┌──────────────────────────┐                                    ┌──────────────────────────┐
 │  FastEmbed + Qdrant      │                                    │   FastEmbed + Qdrant     │
 └─────────────┬────────────┘                                    └─────────────┬────────────┘
               │                                                               │
               ▼                                                               ▼
 ┌──────────────────────────┐                                    ┌──────────────────────────┐
 │ OpenRouter ➔ Gemini LLM  │                                    │ OpenRouter ➔ Gemini LLM  │
 └──────────────────────────┘                                    └──────────────────────────┘
```

---

## 🛠️ Tech Stack

- **GUI Framework**: CustomTkinter, TkinterDnD2
- **Embeddings**: FastEmbed (`BAAI/bge-small-en-v1.5`), SentenceTransformers fallback
- **Vector Database**: Qdrant Cloud (`qdrant-client`)
- **PDF Extraction**: PyMuPDF (`fitz`), pdfminer.six
- **LLM API Providers**: OpenRouter API, Google Gemini API
- **Configuration**: Pydantic Settings, `python-dotenv`

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/DevasisPanda/TinyRag.git
cd TinyRag
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=pdf_documents
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
TOP_K=10
```

### 3. Launch Desktop Application

Run the GUI application:

```bash
python app.py
```

Or run the interactive CLI interface:

```bash
python main.py
```

---

## 📊 Example Queries

### Ordinary Document Q&A
- *"What guidelines were laid down in D.K. Basu vs State of West Bengal?"*
- *"What is the requirement for preparing an arrest memo?"*
- *"What procedural safeguards exist for arrested persons?"*

### Multi-Document Comparison
- *"Compare D.K. Basu and Ashwani Kumar."*
- *"Compare Githa Hariharan and L. Chandra Kumar."*
- *"Difference between D.K. Basu and Ashwani Kumar."*
- *"Compare constitutional issues in Githa Hariharan and L. Chandra Kumar."*

---

## 👨‍💻 Author & Maintainer

**Devasis Panda**  
- Email: devasis.stu.work@gmail.com  
- GitHub: [@DevasisPanda](https://github.com/DevasisPanda)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
