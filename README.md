# Nullclasss_intership
# 🔬 Scientific Research Chatbot (arXiv ExpertBot)

This project is developed as part of the NullClass Generative AI Internship. It is a domain-specific RAG (Retrieval-Augmented Generation) chatbot that allows users to query scientific papers from the arXiv dataset, with a focus on categories like NLP, QCD, LHC, and more.

## 🚀 Features

- 🧠 LLM-powered chatbot using Hugging Face's Falcon 7B Instruct model.
- 🔎 Semantic search over FAISS vector store.
- 📄 Extractive document retrieval from arXiv metadata.
- 🧾 Source-aware answers with citations from relevant research papers.
- 💻 Interactive UI built with Streamlit.

---

## 📁 Project Structure

```bash
.
├── arxiv_chat.py              # Streamlit app for user interaction
├── arxiv_loader.py           # Loads and filters arXiv metadata
├── faiss_arxiv_index/        # Precomputed FAISS index of arXiv papers
├── requirements.txt          # All required Python packages
├── Internship_Report_Affan_Ansari.pdf  # Final report
└── README.md                 # Project description (this file)



## 🧪 How It Works
Loads a subset of arXiv papers using arxiv_loader.py.

Converts abstracts and titles into vector embeddings using sentence-transformers/all-MiniLM-L6-v2.

Stores them in a FAISS index for fast similarity search.

Uses LangChain’s RetrievalQA with HuggingFaceEndpoint (Falcon-7B-Instruct) to generate answers with citations.
