import streamlit as st
import requests
import re  # Pour nettoyer les réponses

# ---------------------------- CONFIGURATION ----------------------------

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

st.set_page_config(
    page_title="DeepSeek-R1 Chat",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------- MODE SOMBRE ----------------------------

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

col1, col2 = st.columns([5, 1])
with col2:
    if st.button("🌙 Mode sombre" if not st.session_state["dark_mode"] else "☀️ Mode clair"):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()

# CSS dynamique en fonction du mode sombre
light_theme_css = """
    <style>
        body { background-color: white; color: black; }
        .stChatMessage[data-testid="stChatMessage-user"] { background-color: #d1e7fd; color: black; }
        .stChatMessage[data-testid="stChatMessage-assistant"] { background-color: #e8f5e9; color: black; }
    </style>
"""

dark_theme_css = """
    <style>
        body { background-color: #1e1e1e; color: white; }
        .stChatMessage[data-testid="stChatMessage-user"] { background-color: #333; color: white; }
        .stChatMessage[data-testid="stChatMessage-assistant"] { background-color: #555; color: white; }
    </style>
"""

st.markdown(dark_theme_css if st.session_state["dark_mode"] else light_theme_css, unsafe_allow_html=True)

# ---------------------------- FONCTIONS ----------------------------

def clean_response(text):
    """Nettoie les réponses du modèle pour enlever les tags inutiles."""
    text = re.sub(r"<.*?>", "", text)  # Supprime les balises HTML et XML
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9.,!?()€$£% ]", "", text)  # Supprime les caractères spéciaux non standards
    return text.strip()

def query_vllm(messages):
    """Envoie une requête au serveur vLLM et récupère la réponse."""
    try:
        response = requests.post(
            VLLM_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.7,  # Contrôle de la créativité
                "max_tokens": 300,  # Limite de la réponse
                "stop": ["</think>", "<RE>", "<success>", "<error>"],  # Stop tokens
            },
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Erreur lors de la requête : {e}")
        return None

# ---------------------------- HEADER ----------------------------

st.title("🤖 DeepSeek-R1 Chat")
st.markdown(
    "💬 **Explorez la puissance du modèle DeepSeek-R1 avec Chain-of-Thought (COT).**"
)

# ---------------------------- GESTION DU CHAT ----------------------------

# Bouton pour réinitialiser la conversation
if st.button("🔄 Réinitialiser la conversation"):
    st.session_state.messages = [{"role": "system", "content": "Posez une question !"}]
    st.rerun()

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Tu es une IA utile et bienveillante qui répond de manière claire et concise."}
    ]

# Affichage des messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg["content"])

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")

if user_input:
    # Ajout du message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)

    # Réponse du modèle avec animation de chargement
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Réflexion en cours..."):
            response = query_vllm(st.session_state.messages)

        if response and "choices" in response:
            answer = clean_response(response["choices"][0]["message"]["content"])
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)
        else:
            st.error("❌ Erreur : aucune réponse reçue du modèle.")
