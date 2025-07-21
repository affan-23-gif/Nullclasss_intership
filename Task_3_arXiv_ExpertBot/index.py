# build_index.py

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
import os

# --- Config ---
JSONL_PATH = "arxiv-metadata-oai-snapshot.json"
OUTPUT_DIR = "faiss_arxiv_index"
MAX_DOCS = 1000  # Set higher if you want more

# --- Load and Parse ---
docs = []
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= MAX_DOCS:
            break
        data = json.loads(line)
        title = data.get("title", "")
        abstract = data.get("abstract", "")
        paper_id = data.get("id", "")
        if title and abstract:
            full_text = f"{title}\n\n{abstract}"
            metadata = {"id": paper_id, "source": "arxiv"}
            docs.append(Document(page_content=full_text, metadata=metadata))

# --- Chunking ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --- Embedding ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Create FAISS Index ---
db = FAISS.from_documents(chunks, embedding_model)

# --- Save Index ---
db.save_local(OUTPUT_DIR)
print(f"✅ FAISS index saved to '{OUTPUT_DIR}' with {len(chunks)} chunks.")
