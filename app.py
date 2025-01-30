import streamlit as st
import requests
import re
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ---------------------------- CONFIGURATION ----------------------------

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

st.set_page_config(
    page_title="DeepSeek-R1 Chat avec RAG",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------- INIT QDRANT ----------------------------

qdrant = QdrantClient("localhost", port=6333)
COLLECTION_NAME = "knowledge_base"

# Initialiser Sentence-Transformers pour générer les embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Vérifier si la collection existe, sinon la créer
try:
    qdrant.get_collection(COLLECTION_NAME)
except:
    qdrant.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# ---------------------------- MODE SOMBRE ----------------------------

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

if st.button("🌙 Mode sombre" if not st.session_state["dark_mode"] else "☀️ Mode clair"):
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]
    st.rerun()

light_theme_css = """<style> body { background-color: white; color: black; } </style>"""
dark_theme_css = """<style> body { background-color: #1e1e1e; color: white; } </style>"""
st.markdown(dark_theme_css if st.session_state["dark_mode"] else light_theme_css, unsafe_allow_html=True)

# ---------------------------- FONCTIONS ----------------------------

def clean_response(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9.,!?()€$£% ]", "", text)
    return text.strip()

def query_vllm(messages):
    """Envoie une requête au serveur vLLM et récupère la réponse."""
    try:
        response = requests.post(
            VLLM_URL,
            headers={"Content-Type": "application/json"},
            json={"model": MODEL_NAME, "messages": messages, "temperature": 0.7, "max_tokens": 300}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Erreur lors de la requête : {e}")
        return None

def add_document(text):
    """Ajoute un document à Qdrant avec son embedding."""
    vector = embedding_model.encode(text).tolist()
    point = PointStruct(id=int(hash(text) % 1e6), vector=vector, payload={"text": text})
    qdrant.upsert(COLLECTION_NAME, points=[point])

def search_documents(query):
    """Recherche les documents pertinents dans Qdrant."""
    vector = embedding_model.encode(query).tolist()  # Fonction qui transforme le texte en vecteur

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,  # ✅ `query_vector` fonctionne ici
        limit=3  # Nombre de résultats
    )

    # Extraction du texte des documents
    docs = [hit.payload["text"] for hit in search_result]

    return docs


# ---------------------------- HEADER ----------------------------

st.title("🤖 DeepSeek-R1 Chat avec RAG")
st.markdown("💬 **Posez vos questions et obtenez des réponses augmentées par la récupération de documents.**")

# ---------------------------- GESTION DU CHAT ----------------------------

if st.button("🔄 Réinitialiser la conversation"):
    st.session_state.messages = [{"role": "system", "content": "Posez une question !"}]
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "Tu es une IA utile et bienveillante."}]

# Ajout de documents (interface admin)
if st.checkbox("📄 Ajouter un document à la base de connaissances"):
    doc_text = st.text_area("Ajoutez du contenu textuel pour l'enrichir dans Qdrant")
    if st.button("📥 Ajouter"):
        add_document(doc_text)
        st.success("Document ajouté !")

# Affichage des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.write(msg["content"])

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")

if user_input:
    # Recherche des documents pertinents
    retrieved_docs = search_documents(user_input)

    # Création du contexte à injecter
    context = "\n\n".join(retrieved_docs) if retrieved_docs else "Aucun document trouvé."

    # Ajout du message utilisateur avec le contexte
    query = f"Contexte:\n{context}\n\nQuestion: {user_input}"

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    # Réponse du modèle
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Réflexion en cours..."):
            response = query_vllm(st.session_state.messages)

        if response and "choices" in response:
            answer = clean_response(response["choices"][0]["message"]["content"])
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)
        else:
            st.error("❌ Erreur : aucune réponse reçue du modèle.")
