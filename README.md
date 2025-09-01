
# Project Title

A brief description of what this project does and who it's for
Here’s a **README.md** draft based on your instructions and the provided documentation 👇

---

# Enhanced Multimodal PDF RAG System

## Overview

This system is an advanced Retrieval-Augmented Generation (RAG) implementation designed for comprehensive analysis of PDF documents containing both textual and visual content. It combines multiple AI technologies to provide accurate, context-aware responses based strictly on document content.

---

## Setup Instructions

### 1. Clone and Navigate

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 4. Environment Configuration

* Create a `.env` file in the project root.
* Add your **API key**:

  ```env
  OPENAI_API_KEY=your_api_key_here
  ```
* Add your **PDF path** in `app.py` at **line 99**:

  ```python
  pdf_path = "your_pdf_path_here"
  ```

---

## Running the Server

Start the Flask server on **localhost:5001**:

```bash
python app.py
```

---

## API Endpoints

### Ask a Question

```bash
curl -X POST http://127.0.0.1:5001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key market trends?"}'
```

### System Health Check

```bash
curl http://127.0.0.1:5001/health
```

### Debug Retrieval

```bash
curl -X POST http://127.0.0.1:5001/debug \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare software vs hardware growth rates"}'
```

---

## Architecture Overview

### Core Technologies

* **Document Processing & Extraction**: PyMuPDF (fitz)
* **Embeddings**:

  * Text: `sentence-transformers/msmarco-distilbert-dot-v5`
  * Images: `openai/clip-vit-base-patch32`
* **Vector Storage**: FAISS
* **Reranker**: `BAAI/bge-reranker-base`
* **LLMs**: GPT-4o (primary), GPT-3.5-turbo (fallback)
* **Framework**: Flask REST API

### Key Features

* Document-only responses with **strict source attribution** (page-level).
* **Multimodal understanding** (text + images).
* **Efficient retrieval** with FAISS and reranking.
* **Scalable** and optimized for professional document analysis.

---

## Example Workflow

1. **Initialization**

   * Load API keys and models.
   * Setup FAISS vector store.

2. **Document Processing**

   * Parse PDFs, extract text + images.
   * Chunk text (`chunk_size=800`, `chunk_overlap=200`).
   * Generate embeddings and store them.

3. **Query Handling**

   * Retrieve relevant text and image chunks.
   * Apply reranking for precision.
   * Construct ultra-strict prompt with sources.

4. **Response Generation**

   * LLM generates answer with citations.
   * Fallback mechanisms ensure reliability.

---

## Configuration Parameters

* **Chunk Size**: 800 characters
* **Overlap**: 200 characters
* **Retrieval**: Top 25 candidates → rerank to 5
* **LLM Temperature**: 0.1
* **Image Limit**: 2 per query

---

## Conclusion

The **Enhanced Multimodal PDF RAG System** is a state-of-the-art solution for PDF document analysis. By combining advanced embeddings, multimodal retrieval, reranking, and LLMs, it ensures **highly accurate, source-grounded answers** suitable for professional and enterprise use cases.

---

👉 Do you want me to also **add example `.env` file content** and a **sample `requirements.txt`** so everything is plug-and-play?

