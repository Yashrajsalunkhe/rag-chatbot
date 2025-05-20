import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# === Configuration ===
FAQ_FILE = "faq.txt"
GENERATION_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_RESPONSE_LEN = 256

# === Load & process the FAQ data ===
@st.cache_resource
def load_docs():
    loader = TextLoader(FAQ_FILE)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)

# === Create Vector Store ===
@st.cache_resource
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# === Load Model Pipeline ===
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model=GENERATION_MODEL)

# === Generate answer ===
def get_answer(query, retriever, generator):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = generator(prompt, max_length=MAX_RESPONSE_LEN, do_sample=False)
    return response[0]['generated_text']

# === Streamlit App ===
def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
    st.title("💬 FAQ Chatbot (LangChain + HuggingFace)")

    # Load data/model
    with st.spinner("Loading documents and models..."):
        docs = load_docs()
        db = create_vector_store(docs)
        retriever = db.as_retriever()
        generator = load_model()

    # UI for user query
    query = st.text_input("Ask a question based on the FAQ:")
    if query:
        with st.spinner("Generating answer..."):
            answer = get_answer(query, retriever, generator)
        st.markdown(f"**🤖 Answer:** {answer}")

if __name__ == "__main__":
    main()
