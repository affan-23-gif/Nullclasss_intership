from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema import Document
from arxiv_loader import load_arxiv_subset
import os

embedding_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

ARXIV_CSV = "arxiv-metadata.csv"  # Ensure this CSV exists
INDEX_DIR = "faiss_arxiv_index"


def build_arxiv_vectorstore():
    raw_docs = load_arxiv_subset(ARXIV_CSV, category="cs", limit=300)
    docs = [Document(page_content=d["content"]) for d in raw_docs]

    vectordb = FAISS.from_documents(docs, embedding_model)
    vectordb.save_local(INDEX_DIR)
    print(f"Saved vectorstore to {INDEX_DIR}")


if __name__ == "__main__":
    build_arxiv_vectorstore()
