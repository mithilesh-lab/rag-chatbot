# RAG Chatbot — Retrieval-Augmented Generation over Custom Documents

## Overview

This project is a production-ready Retrieval-Augmented Generation (RAG) system that enables users to upload their own documents and interact with them through a conversational AI interface. Unlike standard language models that rely solely on pre-trained knowledge, this system grounds every response in the actual content of the uploaded documents, significantly reducing hallucination and improving factual accuracy.

The application accepts PDF and plain text files, processes them into a searchable vector index, and uses a large language model to generate precise, context-aware answers to user queries. The entire pipeline — from document ingestion to response generation — is built from scratch using industry-standard open-source libraries.

## Live Demo

https://rag-chatbot-mithilesh.streamlit.app/

## How It Works

The system operates in two distinct phases: an ingestion phase and a query phase.

During ingestion, the uploaded document is first parsed to extract raw text. This text is then split into smaller overlapping chunks to ensure that context is not lost at boundaries. Each chunk is converted into a dense numerical vector using a sentence embedding model. These vectors are stored in a FAISS index, which enables millisecond-speed similarity search across thousands of chunks.

During querying, the user's question is embedded using the same model. FAISS searches the index for the most semantically similar chunks — not just keyword matches, but meaning-level matches. The retrieved chunks are injected into a structured prompt and sent to the Groq-hosted LLaMA language model, which reads the context and generates a coherent, grounded answer. The system also maintains conversation history across multiple turns, allowing follow-up questions without losing context.

## Architecture
```
Document Ingestion Pipeline
------------------------------------------------------
PDF / TXT File
    -> Text Extraction        (pdfplumber)
    -> Chunking with Overlap  (custom implementation)
    -> Dense Embedding        (Sentence Transformers)
    -> Vector Indexing        (FAISS)

Query Pipeline
------------------------------------------------------
User Question
    -> Embed Question         (Sentence Transformers)
    -> Similarity Search      (FAISS Top-K Retrieval)
    -> Prompt Construction    (context + history)
    -> LLM Generation         (Groq — LLaMA 3)
    -> Answer with Sources
```

## Features

- Supports PDF and plain text document uploads
- Semantic search using dense vector embeddings rather than keyword matching
- Retrieves the most contextually relevant chunks for each query
- Multi-turn conversation with memory of previous exchanges
- Source attribution showing which document sections informed the answer
- Fully deployable as a public web application

## Technology Stack

**FAISS (Facebook AI Similarity Search)**
An open-source library developed by Meta for efficient similarity search over dense vectors. Used here to store and search document chunk embeddings at high speed.

**Sentence Transformers**
A Python library built on top of HuggingFace Transformers that produces high-quality sentence-level embeddings. The model used is all-MiniLM-L6-v2, which maps text to a 384-dimensional vector space capturing semantic meaning.

**Groq API**
Groq provides ultra-fast inference for open-source large language models using custom LPU hardware. This project uses the LLaMA 3.3 70B model served through Groq's free API tier.

**Streamlit**
An open-source Python framework for building interactive web applications. Used to build the entire frontend without requiring separate HTML, CSS, or JavaScript.

**pdfplumber**
A Python library for extracting text and structured data from PDF files with high accuracy.

## Running Locally

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/mithilesh-lab/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
streamlit run app.py
```

The application will open at http://localhost:8501 in your browser.

## Project Structure
```
rag-chatbot/
├── app.py               # Streamlit UI and application logic
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Key Concepts Demonstrated

- End-to-end RAG pipeline design and implementation
- Vector embeddings and semantic similarity search
- FAISS index construction and querying
- Prompt engineering for grounded language model responses
- Multi-turn conversational memory management
- Cloud deployment with Streamlit Community Cloud