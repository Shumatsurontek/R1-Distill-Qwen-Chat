from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.core.config import settings

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if not any(col.name == "knowledge_base" for col in collections):
            self.client.create_collection(
                collection_name="knowledge_base",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def search(self, vector, limit=3):
        return self.client.search(
            collection_name="knowledge_base",
            query_vector=vector,
            limit=limit
        ) 