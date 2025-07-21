import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db, update_vector_db_from_file
import os

st.title("CUSTOMER SERVICE CHATBOT 🤖")

btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()
    st.success("Vector DB created from dataset.csv")

uploaded_file = st.file_uploader("Upload new FAQ CSV to update knowledgebase", type=["csv"])
if uploaded_file:
    temp_path = "temp_uploaded.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    update_vector_db_from_file(temp_path)
    st.success("Knowledgebase updated successfully!")

question = st.text_input("Ask your question:")
if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer")
    st.write(response["result"])
