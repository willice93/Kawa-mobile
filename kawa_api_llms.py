import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from llama_cpp import Llama

CHUNKS_DIR = "chunks"
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
MAPPING_FILE = os.path.join(EMBEDDINGS_DIR, "embedding_mapping.txt")
MODEL_NAME_EMBED = "all-MiniLM-L6-v2"
GGUF_MODEL_PATH = "models/tinyllama-1.1b-chat.Q3_K_M.gguf"

# Chargement des chunks
def load_chunks():
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f]
    chunks = []
    for name in filenames:
        path = os.path.join(CHUNKS_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            chunks.append(f.read().strip())
    return chunks

# Chargement des embeddings
def load_embeddings():
    return np.load(EMBEDDINGS_FILE)

# Recherche sémantique
def retrieve_context(query, top_k=3):
    query_vec = embedder.encode([query])
    scores, indices = index.search(np.array(query_vec), top_k)
    return "\n\n".join([chunks[i] for i in indices[0]])

# Initialisation
print("[KAWA] Chargement du RAG local...")
chunks = load_chunks()
vectors = load_embeddings()
embedder = SentenceTransformer(MODEL_NAME_EMBED)

import faiss
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

print("[KAWA] Chargement du modèle GGUF...")
llm = Llama(model_path=GGUF_MODEL_PATH, n_ctx=512)

# Flask API
app = Flask(__name__)

from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    context = retrieve_context(question)
    prompt = f"{context}\n\nQuestion: {question}\nRéponse:"

    output = llm(prompt, max_tokens=100)
    answer = output["choices"][0]["text"].strip()

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
