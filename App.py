import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def main():
    st.title("💬 Simple Chatbot with LangChain")

    # Get HuggingFace Hub API token from environment variables
    huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not huggingface_token:
        st.error("Error: HuggingFace API token not found. Please set it in your .env file.")
        return

    # Load your data (FAQ or any text file)
    loader = TextLoader("faq.txt")
    documents = loader.load()

    # Split large documents into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Create embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings()

    # Build vectorstore index using FAISS
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # Initialize the HuggingFaceHub LLM with your token
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 256},
        huggingfacehub_api_token=huggingface_token
    )

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User input query
    query = st.text_input("Ask a question:", "")

    # Run the query through the QA chain and show response
    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
        st.markdown(f"**Bot:** {response}")

if __name__ == "__main__":
    main()
