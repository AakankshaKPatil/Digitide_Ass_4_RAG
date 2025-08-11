📄 Simple RAG Q&A App (Streamlit + Hugging Face)
This is a simple Retrieval-Augmented Generation (RAG) web application built with Streamlit, Sentence-Transformers, and Hugging Face Transformers.
It lets you upload .txt files, indexes them using embeddings, retrieves relevant context for a user query, and generates an AI-based answer.

✨ Features
Upload one or more .txt documents

Build a knowledge base with all-MiniLM-L6-v2 embeddings

Retrieve the most relevant document for a question

Generate an answer using distilgpt2 (Hugging Face model)

Simple, clean browser-based UI

No API keys required


📂 Project Structure
bash
.
├── rag_app.py          # Main Streamlit app
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── example_doc.txt     # Sample document for testing

**⚙ Installation & Setup**
1.**Clone the repository**
  bash
  
  git clone https://github.com/YOUR_USERNAME/simple-rag-app.git
  cd simple-rag-app

2.**(Optional) Create a virtual environment**
  bash
  
  python -m venv venv
  venv\Scripts\activate      # Windows
  source venv/bin/activate   # Mac/Linux

3.**Install dependencies**
  bash
  
  pip install -r requirements.txt

4.**Run the app**
  bash
  
  streamlit run rag_app.py

5.**Open in your browser**
Streamlit will automatically open:

arduino

http://localhost:8501
