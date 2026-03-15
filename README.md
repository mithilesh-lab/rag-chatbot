\# 🧠 RAG Chatbot — Ask Questions Over Your Documents



A RAG (Retrieval-Augmented Generation) chatbot that lets you upload PDFs or text files and ask questions about them. Built with FAISS, Sentence Transformers, Groq LLM, and Streamlit.



\## 🚀 Live Demo

👉 https://rag-chatbot-fqffmmwjpv9xb3h4ulh4ug.streamlit.app



\## 🛠️ How It Works

1\. Upload a PDF or TXT file

2\. Click \*\*Build Knowledge Base\*\*

3\. Ask any question about your document

4\. Get accurate answers with source references!



\## ⚙️ Tech Stack

\- \*\*FAISS\*\* — Vector similarity search

\- \*\*Sentence Transformers\*\* — Text embeddings

\- \*\*Groq LLM\*\* — Fast AI responses (LLaMA 3)

\- \*\*Streamlit\*\* — Web interface



\## 📐 Architecture

```

Upload Document → Extract Text → Chunk → Embed → FAISS Index

Ask Question → Embed Question → Search FAISS → Send to LLM → Answer

```



\## 🔧 Run Locally

```bash

pip install -r requirements.txt

streamlit run app.py

```

