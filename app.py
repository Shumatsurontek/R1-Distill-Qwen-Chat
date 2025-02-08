import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List
import uuid
import os
import time

# ---------------------------- CONFIGURATION ----------------------------

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

st.set_page_config(
    page_title="DeepSeek-R1 Chat avec RAG",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------- INIT QDRANT ----------------------------

def init_qdrant():
    """Initialise la connexion Qdrant et crée la collection si nécessaire"""
    retries = 5
    while retries > 0:
        try:
            api_key = os.getenv("QDRANT_API_KEY")
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            
            client = QdrantClient(
                url=qdrant_url,
                api_key=api_key
            )
            
            collections = client.get_collections().collections
            exists = any(col.name == "knowledge_base" for col in collections)
            
            if not exists:
                client.create_collection(
                    collection_name="knowledge_base",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                st.success("✅ Collection créée avec succès!")
            else:
                st.info("ℹ️ Collection existante utilisée")
            
            return client
        except Exception as e:
            retries -= 1
            if retries == 0:
                st.error(f"❌ Erreur de connexion à Qdrant après plusieurs tentatives: {str(e)}")
                raise e
            time.sleep(2)

def init_vllm_client():
    """Initialise la connexion vLLM avec retry"""
    retries = 5
    while retries > 0:
        try:
            # Test de connexion
            response = requests.get(f"{VLLM_URL.rsplit('/', 1)[0]}/health")
            if response.status_code == 200:
                return
            raise Exception(f"Status code: {response.status_code}")
        except Exception as e:
            retries -= 1
            if retries == 0:
                st.error(f"❌ Impossible de se connecter à vLLM après plusieurs tentatives: {str(e)}")
                raise e
            st.warning(f"⚠️ Tentative de connexion à vLLM ({retries} essais restants)...")
            time.sleep(5)  # Attendre plus longtemps pour vLLM

# ---------------------------- WEB SCRAPING ----------------------------

def scrape_website(url: str) -> List[str]:
    """Scrape content from a website and split it into chunks."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and split into paragraphs
        text = soup.get_text()
        chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
        
        # Filter out short chunks and combine into reasonable sizes
        meaningful_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < 1000:
                current_chunk += " " + chunk
            else:
                if current_chunk:
                    meaningful_chunks.append(current_chunk.strip())
                current_chunk = chunk
                
        if current_chunk:
            meaningful_chunks.append(current_chunk.strip())
            
        return meaningful_chunks
        
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return []

# ---------------------------- UI COMPONENTS ----------------------------

def sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Dark mode toggle
        if "dark_mode" not in st.session_state:
            st.session_state["dark_mode"] = False
        
        if st.toggle("🌙 Mode sombre", st.session_state["dark_mode"]):
            st.session_state["dark_mode"] = not st.session_state["dark_mode"]
            st.rerun()
            
        # Knowledge Base Management
        st.subheader("📚 Base de connaissances")
        
        # Add single document
        with st.expander("📄 Ajouter un document"):
            doc_text = st.text_area("Contenu du document")
            if st.button("📥 Ajouter le document"):
                if doc_text:
                    add_document(doc_text)
                    st.success("Document ajouté avec succès!")
                    
        # Web scraping
        with st.expander("🌐 Scraper un site web"):
            url = st.text_input("URL du site web")
            if st.button("🔍 Scraper et ajouter"):
                if url:
                    with st.spinner("Scraping en cours..."):
                        chunks = scrape_website(url)
                        for chunk in chunks:
                            add_document(chunk)
                        st.success(f"✅ {len(chunks)} segments ajoutés à la base de connaissances!")

# ---------------------------- CHAT INTERFACE ----------------------------

def chat_interface():
    st.title("🤖 DeepSeek-R1 Chat avec RAG")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Je suis une IA assistante utile et bienveillante."}
        ]
    
    # Chat reset button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.messages = [
                {"role": "system", "content": "Je suis une IA assistante utile et bienveillante."}
            ]
            st.rerun()
    
    # Display chat messages
    for msg in st.session_state.messages[1:]:  # Skip system message
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Posez votre question..."):
        # Search relevant documents
        docs = search_documents(prompt)
        context = "\n\n".join(docs) if docs else "Aucun document pertinent trouvé."
        
        # Prepare query with context
        query = f"Contexte:\n{context}\n\nQuestion: {prompt}"
        
        # Add user message and get response
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Réflexion..."):
                response = query_vllm(st.session_state.messages[-2:])  # Send last 2 messages
                
            if response and "choices" in response:
                answer = clean_response(response["choices"][0]["message"]["content"])
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.write(answer)
            else:
                st.error("❌ Erreur: Pas de réponse du modèle.")

# ---------------------------- MAIN APP ----------------------------

def main():
    try:
        # Initialiser les connexions
        global qdrant
        qdrant = init_qdrant()
        init_vllm_client()
        
        # Afficher l'interface
        sidebar()
        chat_interface()
        
        # Appliquer le thème
        theme = dark_theme_css if st.session_state["dark_mode"] else light_theme_css
        st.markdown(theme, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation: {str(e)}")
        st.info("🔄 Rechargez la page dans quelques instants...")

# ---------------------------- UTILITY FUNCTIONS ----------------------------

def clean_response(text):
    """Nettoie la réponse du modèle."""
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
    vector = embedding_model.encode(query).tolist()
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=3
    )
    return [hit.payload["text"] for hit in search_result]

# Définition des styles pour le mode sombre/clair
light_theme_css = """<style> body { background-color: white; color: black; } </style>"""
dark_theme_css = """<style> body { background-color: #1e1e1e; color: white; } </style>"""

if __name__ == "__main__":
    main()
