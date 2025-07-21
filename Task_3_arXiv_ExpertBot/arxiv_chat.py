# arxiv_chat.py

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# --- Load environment variables ---
load_dotenv()

# --- Load Embedding Model ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load FAISS Vector Store ---
INDEX_PATH = "faiss_arxiv_index"
db = FAISS.load_local(INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

# --- Load LLM from Hugging Face Endpoint ---


llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4",  # Or "gpt-3.5-turbo"
    temperature=0.7
)

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template("""
You are a scientific research assistant. Answer the question based on the following papers:

{context}

Question: {question}

Answer:
""")

# --- QA Chain Setup ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="arXiv ExpertBot", layout="wide")
st.title("🔬 arXiv Scientific Chatbot")

query = st.text_input("Ask a research question (e.g., about QCD, LHC, NLP papers):")

if query:
    with st.spinner("🔍 Searching and thinking..."):
        result = qa_chain.invoke(query)  # ✅ Use `.invoke()` instead of deprecated `__call__`

    st.subheader("📌 Answer")
    st.write(result["result"])

    st.subheader("📚 Sources")
    for doc in result["source_documents"]:
        st.markdown(f"- **Doc ID**: `{doc.metadata.get('id', 'N/A')}`\n\n> {doc.page_content[:300]}...")
