import streamlit as st
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Page config ──
st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")

st.title("🧠 RAG Chatbot")
st.caption("Upload a document and ask questions about it")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    model_name = "llama-3.3-70b-versatile"

    if st.button("🚀 Build Knowledge Base"):
        if not api_key:
            st.error("Please enter your Groq API key!")
        elif not uploaded_file:
            st.error("Please upload a file!")
        else:
            with st.spinner("Processing document..."):

                # ── Load file ──
                if uploaded_file.name.endswith(".pdf"):
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = "\n".join(
                            p.extract_text() for p in pdf.pages if p.extract_text()
                        )
                else:
                    text = uploaded_file.read().decode("utf-8", errors="ignore")

                # ── Chunk ──
                words = text.split()
                chunk_size, overlap = 100, 20
                chunks, start = [], 0
                while start < len(words):
                    end = min(start + chunk_size, len(words))
                    chunks.append(" ".join(words[start:end]))
                    if end == len(words):
                        break
                    start += chunk_size - overlap

                # ── Embed ──
                embedder = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = embedder.encode(
                    chunks, convert_to_numpy=True
                ).astype(np.float32)

                # ── FAISS index ──
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)

                # ── Save to session state ──
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.embedder = embedder
                st.session_state.client = Groq(api_key=api_key)
                st.session_state.chat_history = []
                st.session_state.ready = True

            st.success(f"✅ Ready! Indexed {len(chunks)} chunks.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Chat UI ──
if "ready" not in st.session_state:
    st.info("👈 Upload a document and click 'Build Knowledge Base' to start!")
else:
    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if question := st.chat_input("Ask something about your document..."):

        # Show user message
        with st.chat_message("user"):
            st.write(question)

        # Retrieve chunks
        q_emb = st.session_state.embedder.encode(
            [question], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = st.session_state.index.search(q_emb, 3)
        context = "\n\n".join([st.session_state.chunks[i] for i in indices[0]])

        # Build messages
        messages = [
            {"role": "system", "content": "Answer using ONLY the context provided. If not in context, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}"},
            {"role": "assistant", "content": "Understood, I'll answer from the context."},
        ]
        messages += st.session_state.chat_history[-6:]
        messages.append({"role": "user", "content": question})

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512
                )
                answer = response.choices[0].message.content.strip()
            st.write(answer)

        # Save to history
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})