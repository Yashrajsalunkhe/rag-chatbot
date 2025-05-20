import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

# Set HuggingFace Hub token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load and process data
loader = TextLoader("faq.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Embeddings and FAISS
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# LLM (Google Flan-T5)
llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.5, "max_length": 256})

# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.title("💬 Simple Chatbot with LangChain")
query = st.text_input("Ask a question:", "")

if query:
    response = qa_chain.run(query)
    st.markdown(f"**Bot:** {response}")
