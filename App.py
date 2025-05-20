import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import os

@st.cache_resource
def load_and_split_docs():
    # Fix path issue:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faq_path = os.path.join(base_dir, "faq.txt")
    loader = TextLoader(faq_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(documents)

@st.cache_resource
def create_faiss_index():
    docs = load_and_split_docs()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

@st.cache_resource
def load_generator_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

def get_answer(query, retriever, generator):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = generator(prompt, max_length=256, do_sample=False)
    return response[0]['generated_text']

def main():
    st.title("📚 FAQ Chatbot (LangChain + HuggingFace)")

    with st.spinner("Loading resources..."):
        db = create_faiss_index()
        generator = load_generator_model()
        retriever = db.as_retriever()

    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Generating answer..."):
            answer = get_answer(query, retriever, generator)
        st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
