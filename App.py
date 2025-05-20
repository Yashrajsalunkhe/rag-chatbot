import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# --- Load environment variables from .env file ---
load_dotenv()

# --- Get the Hugging Face API token ---
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ✅ Validate the token was loaded correctly
if not huggingface_token:
    st.error("⚠️ Hugging Face API token is missing. Please set HUGGINGFACEHUB_API_TOKEN in your environment.")
    st.stop()

# --- Load and split the document ---
loader = TextLoader("faq.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# --- Create embeddings and vector store ---
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# ✅ Use a model that supports inference on Hugging Face Hub
# You can replace this with another model like "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",  # Flan-T5-base may not work reliably
    model_kwargs={"temperature": 0.5, "max_length": 256},
    huggingfacehub_api_token=huggingface_token
)

# --- Build the RetrievalQA chain ---
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Streamlit UI ---
st.title("💬 RAG Chatbot with LangChain")
query = st.text_input("Ask a question based on the FAQ:", "")

if query:
    with st.spinner("Generating answer..."):
        response = qa_chain.run(query)
        st.markdown(f"**🤖 Bot:** {response}")
