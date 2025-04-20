

# KAWA-Mobile

**Agent LLM local avec interface web mobile**, optimisé pour Android (UserLAnd) ou tout environnement Debian.  
Utilise TinyLLaMA + RAG local (Faiss + SentenceTransformers) + interface Flask responsive.

---

## Fonctionnalités

- Modèle : TinyLLaMA 1.1B Chat (GGUF - Q3_K_M)
- Serveur Flask (port 5000)
- RAG minimal avec `.txt` et `.npy`
- Interface web responsive (mobile friendly)
- 100% offline, CPU only
- Compatible Android / Termux / Linux / Raspberry Pi

---

## Installation

### Cloner le projet

```bash
git clone https://github.com/willice93/Kawa-mobile.git
cd Kawa-mobile

Créer l’environnement virtuel

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Installer les dépendances système (Debian)

apt install libopenblas-dev ffmpeg git curl


---

Télécharger le modèle LLM (GGUF)

Télécharge manuellement le modèle TinyLLaMA ici :
TinyLLaMA 1.1B Chat Q3_K_M (GGUF)

Puis place-le dans le dossier models/ :

mv tinyllama-1.1b-chat.Q3_K_M.gguf models/


---

Lancer l’API locale

python kawa_api_llms.py

Ouvrir dans un navigateur mobile ou desktop :

http://localhost:5000


---

Arborescence du projet

Kawa-mobile/
├── kawa_api_llms.py
├── models/
│   └── tinyllama-1.1b-chat.Q3_K_M.gguf
├── chunks/
│   └── *.txt
├── embeddings/
│   ├── embeddings.npy
│   └── embedding_mapping.txt
├── templates/
│   └── index.html
├── requirements.txt
└── README.md


---

Auteur

willice93
Projet libre, reproductible, et conçu pour renforcer l’autonomie numérique avec des outils locaux.


--
