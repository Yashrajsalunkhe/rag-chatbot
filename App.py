import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# Load and process data
loader = TextLoader("faq.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Use public sentence transformer without token
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# Use public T5 model without token
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def get_answer(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = generator(prompt, max_length=256, do_sample=False)
    return response[0]['generated_text']

# Streamlit UI
st.title("📚 FAQ Chatbot (LangChain + HuggingFace)")

query = st.text_input("Ask a question:")
if query:
    answer = get_answer(query)
    st.write("**Answer:**", answer)
