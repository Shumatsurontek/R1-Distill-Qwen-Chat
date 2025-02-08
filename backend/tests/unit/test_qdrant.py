import pytest
from unittest.mock import patch, MagicMock
from app.services.qdrant import QdrantService

class TestQdrantService:
    @patch('app.services.qdrant.QdrantClient')
    def test_init_collection(self, MockQdrantClient):
        # Setup
        mock_client = MockQdrantClient.return_value
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        # Test
        service = QdrantService()
        
        # Verify
        MockQdrantClient.assert_called_once_with(
            url='http://localhost:6333',
            api_key='eyJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3MiOiJyIn0.r7g20UrWnGzTUVxundSq6wKvIXEw57iNnCcuAsOXtaQ'
        )
        mock_client.create_collection.assert_called_once()

    @patch('app.services.qdrant.QdrantClient')
    def test_search(self, MockQdrantClient):
        # Setup
        mock_client = MockQdrantClient.return_value
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="knowledge_base")]
        mock_client.get_collections.return_value = mock_collections

        # Test
        service = QdrantService()
        vector = [0.1] * 384
        service.search(vector)

        # Verify
        mock_client.search.assert_called_once_with(
            collection_name="knowledge_base",
            query_vector=vector,
            limit=3
        ) 