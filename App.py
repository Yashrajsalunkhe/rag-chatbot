import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# === Constants ===
FAQ_FILE = "faq.txt"
MODEL_NAME = "google/flan-t5-base"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MAX_RESPONSE_LEN = 256

# === Load & process everything inside cache ===
@st.cache_resource
def load_pipeline_and_db():
    # Load and split the document
    loader = TextLoader(FAQ_FILE)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(documents)

    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # Load local/public HF model (no token needed)
    generator = pipeline("text2text-generation", model=MODEL_NAME)

    return retriever, generator

# === Answer Generation ===
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

    # Load once and reuse
    with st.spinner("Loading model and database..."):
        retriever, generator = load_pipeline_and_db()

    # User input
    query = st.text_input("Ask a question based on the FAQ:")
    if query:
        with st.spinner("Generating answer..."):
            answer = get_answer(query, retriever, generator)
        st.markdown(f"**🤖 Answer:** {answer}")

if __name__ == "__main__":
    main()
