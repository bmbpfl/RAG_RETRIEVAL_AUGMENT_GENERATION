import streamlit as st
import requests
import subprocess
import psutil
import tiktoken
import faiss
import numpy as np
import re
import io
import os
import tempfile

from sentence_transformers import SentenceTransformer
from docx import Document
from pypdf import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ================== CONFIG ==================
OLLAMA_URL = "http://localhost:11434/api/chat"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.npy"

st.set_page_config(page_title="Local RAG Assistant", layout="wide")

# ================== SESSION STATE ==================
for key, default in {
    "vector_store": None,
    "doc_chunks": [],
    "messages": [],
    "selected_model": None,
    "last_response": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ================== MODEL DISCOVERY ==================
def get_installed_models():
    try:
        result = subprocess.check_output(["ollama", "list"]).decode()
        models = [line.split()[0] for line in result.splitlines()[1:] if line.strip()]
        return models if models else ["llama3.2:latest"]
    except:
        return ["llama3.2:latest"]

AVAILABLE_MODELS = get_installed_models()
if st.session_state.selected_model is None:
    st.session_state.selected_model = AVAILABLE_MODELS[0]

# ================== UTILITIES ==================
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def count_tokens(text, model="gpt-3.5-turbo"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def get_gpu_usage():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"]
        ).decode("utf-8")
        util, mem_used, mem_total = result.strip().split(", ")
        return int(util), int(mem_used), int(mem_total)
    except:
        return None

# ================== SMART CHUNKING ==================
def chunk_text_smart(text, chunk_size=800, overlap=150):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    final_chunks = []
    for c in chunks:
        for i in range(0, len(c), chunk_size - overlap):
            final_chunks.append(c[i:i+chunk_size])

    return final_chunks

# ================== STREAM PDF ==================
def stream_pdf_pages(file, batch_pages=20):
    pdf = PdfReader(io.BytesIO(file.read()))
    total = len(pdf.pages)

    for i in range(0, total, batch_pages):
        batch_text = []
        for j in range(i, min(i + batch_pages, total)):
            text = pdf.pages[j].extract_text() or ""
            batch_text.append(text)
        yield "\n".join(batch_text)

# ================== BATCH EMBEDDING ==================
def embed_in_batches(text_chunks, batch_size=64):
    all_embeddings = []
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        embs = EMBED_MODEL.encode(batch, show_progress_bar=False)
        all_embeddings.append(embs)
    return np.vstack(all_embeddings).astype("float32")

# ================== FAISS PERSISTENCE ==================
def init_faiss(dim):
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    return faiss.IndexFlatL2(dim)

def save_index(index, chunks):
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))

def load_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        st.session_state.vector_store = faiss.read_index(FAISS_INDEX_PATH)
        st.session_state.doc_chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()

load_index()

# ================== INGESTION PIPELINE ==================
def process_large_pdf(file):
    progress = st.progress(0)
    total_batches = len(PdfReader(io.BytesIO(file.read())).pages) // 20 + 1
    file.seek(0)

    index = st.session_state.vector_store
    all_chunks = st.session_state.doc_chunks

    for i, text_batch in enumerate(stream_pdf_pages(file)):
        cleaned = clean_text(text_batch)
        chunks = chunk_text_smart(cleaned)
        embeddings = embed_in_batches(chunks)

        if index is None:
            index = init_faiss(embeddings.shape[1])

        index.add(embeddings)
        all_chunks.extend(chunks)
        progress.progress((i+1)/total_batches)

    save_index(index, all_chunks)
    st.session_state.vector_store = index
    st.session_state.doc_chunks = all_chunks

# ================== RETRIEVAL ==================
def retrieve_context(query, k=5):
    if st.session_state.vector_store is None:
        return ""

    qvec = EMBED_MODEL.encode([query]).astype("float32")
    D, I = st.session_state.vector_store.search(qvec, k)

    return "\n\n".join(
        st.session_state.doc_chunks[i] for i in I[0]
        if i < len(st.session_state.doc_chunks)
    )

# ================== EXPORT ==================
def save_as_pdf(text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    styles = getSampleStyleSheet()
    story = [Paragraph(line, styles["Normal"]) for line in text.split("\n")]
    doc = SimpleDocTemplate(tmp.name, pagesize=letter)
    doc.build(story)
    return tmp.name

# ================== LLM ==================
def format_messages(prompt, context):
    msgs = [{"role": "system", "content": "Use retrieved context when relevant."}]
    for m in st.session_state.messages[-8:]:
        msgs.append(m)
    if context:
        msgs.append({"role": "system", "content": f"Relevant context:\n{context}"})
    msgs.append({"role": "user", "content": prompt})
    return msgs

def query_ollama(messages):
    payload = {"model": st.session_state.selected_model, "messages": messages, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]

# ================== UI ==================
st.title("🧠 Scalable Local RAG Assistant")

selected = st.sidebar.selectbox("Choose Model", AVAILABLE_MODELS)
st.session_state.selected_model = selected

uploaded_file = st.file_uploader("Upload large PDF", type=["pdf"])

if st.button("Process File") and uploaded_file:
    process_large_pdf(uploaded_file)
    st.success("Large document indexed!")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    context = retrieve_context(prompt)
    msgs = format_messages(prompt, context)

    with st.spinner("Thinking..."):
        response = query_ollama(msgs)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(response)

# ================== MONITOR ==================
st.sidebar.subheader("System Monitor")
gpu = get_gpu_usage()
if gpu:
    st.sidebar.write(f"GPU Util: {gpu[0]}%  Memory: {gpu[1]}/{gpu[2]} MB")
st.sidebar.write(f"CPU: {psutil.cpu_percent()}%")
st.sidebar.write(f"RAM: {psutil.virtual_memory().percent}%")
