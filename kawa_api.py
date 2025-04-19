import os
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

CHUNKS_DIR = "chunks"
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
MAPPING_FILE = os.path.join(EMBEDDINGS_DIR, "embedding_mapping.txt")
MODEL_NAME_EMBED = "all-MiniLM-L6-v2"
MODEL_NAME_GEN = "distilgpt2"

# Chargement des chunks
def load_chunks():
    filenames = []
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        filenames = [line.strip() for line in f]
    chunks = []
    for name in filenames:
        path = os.path.join(CHUNKS_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            chunks.append(f.read().strip())
    return chunks

# Chargement de l'index depuis les vecteurs npy (verrouillé)
def load_index():
    vectors = np.load(EMBEDDINGS_FILE)
    vectors.setflags(write=False)  # lecture seule
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

# Recherche dans l'index
def retrieve_context(query, top_k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return "\n\n".join([chunks[i] for i in I[0]])

# Initialisations
print("[KAWA] Chargement des ressources...")
chunks = load_chunks()
index = load_index()
embedder = SentenceTransformer(MODEL_NAME_EMBED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_GEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_GEN)

# Flask API
app = Flask(__name__)

from flask import render_template

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    context = retrieve_context(question)

    prompt = f"{context}\n\nQuestion: {question}\nRéponse:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
