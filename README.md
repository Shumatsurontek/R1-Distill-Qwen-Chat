# 🚀 DeepSeek-R1 Chat Application

Application de chat IA avec RAG (Retrieval Augmented Generation) utilisant DeepSeek-R1, Qdrant et React.

## 🏗️ Architecture

```
R1-Distill-Qwen-Chat/
├── backend/                 # API FastAPI
│   ├── app/
│   │   ├── api/            # Endpoints API
│   │   ├── core/           # Configuration
│   │   ├── services/       # Logique métier
│   │   └── models/         # Modèles Pydantic
│   ├── tests/              # Tests unitaires et d'intégration
│   └── Dockerfile
├── frontend/               # Interface React
│   ├── src/
│   │   ├── components/     # Composants React
│   │   ├── services/       # Services API
│   │   └── styles/         # CSS & Thèmes
│   ├── tests/             # Tests React
│   └── Dockerfile
├── docker/                # Configuration Docker
│   ├── vllm/
│   ├── qdrant/
│   └── nginx/
└── docker-compose.yml
```

## 🧪 Tests

### Tests Unitaires Backend
```bash
cd backend
pytest tests/unit -v --cov=app
```

### Tests d'Intégration
```bash
cd backend
pytest tests/integration -v
```

### Tests E2E
```bash
docker compose -f docker-compose.test.yml up --build
```

## 🔍 Stratégie de Test

1. **Tests Unitaires**:
   - Mock des dépendances externes (Qdrant, vLLM)
   - Test des fonctions individuelles
   - Couverture de code > 80%

2. **Tests d'Intégration**:
   - Test des interactions entre services
   - Validation des flux de données
   - Tests avec Docker

3. **Tests E2E**:
   - Validation complète du système
   - Tests UI avec Cypress
   - Scénarios utilisateur

## 🐳 Docker

1. **Services**:
   - `backend`: API FastAPI
   - `frontend`: UI React
   - `vllm`: Service d'inférence
   - `qdrant`: Base vectorielle
   - `nginx`: Reverse proxy

2. **Configuration**:
   ```yaml
   version: '3.8'
   services:
     backend:
       build: ./backend
       environment:
         - QDRANT_URL=http://qdrant:6333
         - VLLM_URL=http://vllm:8000
     
     frontend:
       build: ./frontend
       ports:
         - "3000:3000"
     
     vllm:
       build: ./docker/vllm
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
     
     qdrant:
       image: qdrant/qdrant
       volumes:
         - qdrant_data:/qdrant/storage
   ```

## 🚀 Démarrage

1. **Développement**:
   ```bash
   docker compose up -d
   cd frontend && npm start
   cd backend && uvicorn app.main:app --reload
   ```

2. **Production**:
   ```bash
   docker compose -f docker-compose.prod.yml up -d
   ```

## 📝 Tests à implémenter

1. **Backend**:
   - Tests unitaires des services
   - Tests d'intégration API
   - Tests de performance

2. **Frontend**:
   - Tests des composants React
   - Tests d'intégration UI
   - Tests E2E avec Cypress

3. **Infrastructure**:
   - Tests de charge
   - Tests de résilience
   - Tests de sécurité

## 🔥 DeepSeek-R1 Chatbot

An interactive chatbot leveraging `DeepSeek-R1` and Chain-of-Thought (COT) reasoning.

## 🚀 Features
- **Fast inference** with `vLLM`
- **Interactive UI** powered by `Streamlit`
- **Simple API integration**
- **Docker support** for easy deployment

## 🛠 Installation
Clone this repo and install dependencies:
```bash
git clone https://github.com/ton_repo/deepseek-r1-chat.git
cd deepseek-r1-chat
pip install -r requirements.txt
