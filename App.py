import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# === Load documents ===
@st.cache_resource
def load_and_split_documents():
    loader = TextLoader("faq.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(documents)

# === Create embeddings and FAISS (do not cache this function with unhashable input) ===
@st.cache_resource
def build_vector_store():
    docs = load_and_split_documents()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# === Load HF model ===
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-base")

# === Answer function ===
def get_answer(query, retriever, generator):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = generator(prompt, max_length=256, do_sample=False)
    return response[0]['generated_text']

# === Streamlit UI ===
def main():
    st.title("💬 FAQ Chatbot (LangChain + HuggingFace)")

    with st.spinner("Loading resources..."):
        db = build_vector_store()
        generator = load_generator()
        retriever = db.as_retriever()

    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Generating answer..."):
            answer = get_answer(query, retriever, generator)
        st.markdown(f"**🤖 Answer:** {answer}")

if __name__ == "__main__":
    main()
