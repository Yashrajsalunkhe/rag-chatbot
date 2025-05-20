# 💬 RAG Chatbot using LangChain and HuggingFace

A Retrieval-Augmented Generation (RAG) based chatbot built with Python, LangChain, HuggingFace, and FAISS. This chatbot can answer questions contextually using your custom dataset.

---

## 🚀 Features

- Loads data from `.txt` file (`faq.txt`)
- Uses LangChain's RAG pipeline for intelligent responses
- Vector similarity search using FAISS
- Embeddings from `sentence-transformers`
- Powered by HuggingFace's `flan-t5-base` model (No OpenAI key required)
- Optional: Deployable on **Streamlit**
- Environment variables managed securely using `.env`

---

## 📁 Project Structure

    rag-chatbot/
    ├── app.py # Main Streamlit app
    ├── chatbot.py # Core chatbot logic (embedding, retriever, LLM)
    ├── faq.txt # Source data (knowledge base)
    ├── responses.txt # Sample Q&A responses
    ├── .env.example # Template for environment variables
    ├── .gitignore # Prevents secret files from being tracked
    ├── requirements.txt # Python dependencies
    └── README.md # Project documentation


---

## ⚙️ Setup Instructions

### 🖥️ Run Locally

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/rag-chatbot.git
   cd rag-chatbot

2. **Create and activate a virtual environment**

    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows

3. **Install dependencies**

    pip install -r requirements.txt

4. **Create a .env file**

    HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

5. **Run the chatbot in terminal**

    python app.py


🌐 Deploy on Streamlit

1. **Install Streamlit and pyngrok (if not already):**

    pip install streamlit pyngrok

2. **Run the app:**

    streamlit run app.py

3. **Use ngrok for public URL in Colab or remote:**

    from pyngrok import ngrok
    ngrok.set_auth_token("your-ngrok-token")
    print(ngrok.connect(8501))



📄 Sample Questions and Responses

See the faq.txt file for example interactions with the chatbot.



🛠️ Tech Stack

    Python

    LangChain

    HuggingFace Transformers

    FAISS (for vector storage)

    Streamlit (UI deployment)

    Pyngrok (optional, for Colab/public URL)


🧠 References

    LangChain Documentation

    HuggingFace Hub

    FAISS Documentation

👨‍💻 Author

Made by Yashraj Salunkhe. Feel free to reach out for any questions!
