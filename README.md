# 💬 RAG Chatbot using LangChain and HuggingFace

A Retrieval-Augmented Generation (RAG) based chatbot built with Python, LangChain, HuggingFace, and FAISS. This chatbot can answer questions contextually using your custom dataset.

---

## 🚀 Features

- Loads data from `.txt` file (`faq.txt`)
- Uses LangChain's RAG pipeline for intelligent responses
- Vector similarity search using FAISS
- Embeddings from `sentence-transformers/all-MiniLM-L6-v2`
- Powered by HuggingFace's `flan-t5-base` model (No OpenAI key required)
- Resource caching with Streamlit's `@st.cache_resource` for faster loading
- Handles file paths dynamically for portability
- Optional: Deployable on **Streamlit**

---

## 🌐 Hosted Demo

Try the chatbot live at:  
**https://rag-chatbot-2cub3ctpot7xzimtgdrimr.streamlit.app/**

---

## 📁 Project Structure

    rag-chatbot/
    ├── app.py            # Main Streamlit app with caching and improved file handling
    ├── faq.txt           # Source data (knowledge base)
    ├── requirements.txt  # Python dependencies
    ├── .gitignore        # Prevents secret files from being tracked
    └── README.md         # Project documentation

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

4. **Run the chatbot**

        python app.py


🌐 Deploy on Streamlit

1. **Install Streamlit (and optionally pyngrok)**

        pip install streamlit pyngrok

2. **Run the app**

        streamlit run app.py

3. **Use ngrok for public URL in Colab or remote environments**

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


---

Let me know if you'd like me to generate a `requirements.txt` or anything else!
