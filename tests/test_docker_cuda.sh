#!/bin/bash

# Couleurs pour les messages
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "🔍 Démarrage des tests CUDA pour le conteneur vLLM..."

# Construire l'image
echo "🏗️  Construction de l'image Docker..."
docker compose build vllm

# Vérifier CUDA dans le conteneur
echo "🔬 Vérification de CUDA dans le conteneur..."
docker run --rm --gpus all r1-distill-qwen-chat-vllm python3 -c "
import torch
print(f'CUDA disponible: {torch.cuda.is_available()}')
print(f'Version CUDA: {torch.version.cuda}')
print(f'Nombre de GPUs: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU actif: {torch.cuda.get_device_name(0)}')
    print(f'Mémoire totale: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"

# Vérifier vLLM
echo "🚀 Test du serveur vLLM..."
docker compose up -d vllm
sleep 30  # Attendre le démarrage

# Test de l'API
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
       "messages": [{"role": "user", "content": "Dis bonjour"}]
     }'

# Vérifier le statut
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tests réussis!${NC}"
else
    echo -e "${RED}❌ Échec des tests${NC}"
    docker compose logs vllm
fi

# Nettoyage
docker compose down 