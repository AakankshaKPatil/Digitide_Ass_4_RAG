import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline
import torch

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=0 if torch.cuda.is_available() else -1
    )
    return embedder, generator

embedder, generator = load_models()

st.title("📄 Simple RAG Q&A App")
st.write("Upload text files, ask a question, and get an AI-generated answer using document context.")

# Upload files
uploaded_files = st.file_uploader("Upload .txt documents", type="txt", accept_multiple_files=True)
documents = []
if uploaded_files:
    for file in uploaded_files:
        documents.append(file.read().decode("utf-8"))

# Build embeddings
if st.button("Build Knowledge Base"):
    if not documents:
        st.warning("Please upload at least one document.")
    else:
        st.session_state.chunks = documents
        st.session_state.embeddings = embedder.encode(documents)
        st.success("Knowledge base built!")

# Ask a question
question = st.text_input("Ask a question based on the uploaded documents:")

if st.button("Get Answer"):
    if "embeddings" not in st.session_state:
        st.warning("Please build the knowledge base first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        # Retrieve most relevant document
        q_emb = embedder.encode([question])
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(st.session_state.embeddings)
        _, idxs = nn.kneighbors(q_emb)
        context = st.session_state.chunks[idxs[0][0]]

        # Generate answer
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = generator(prompt, max_length=150, do_sample=True, temperature=0.7)[0]['generated_text']
        st.subheader("Answer")
        st.write(answer)
