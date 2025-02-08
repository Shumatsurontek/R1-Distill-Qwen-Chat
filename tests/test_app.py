import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import clean_response, scrape_website, init_qdrant
import requests
from bs4 import BeautifulSoup
from unittest.mock import patch, MagicMock

class TestChatApp(unittest.TestCase):
    def setUp(self):
        """Setup pour les tests"""
        self.sample_text = "Ceci est un <script>test</script> avec des <style>styles</style> à nettoyer!"
        self.sample_url = "http://example.com"
        
    def test_clean_response(self):
        """Test du nettoyage des réponses"""
        cleaned = clean_response(self.sample_text)
        self.assertNotIn("<script>", cleaned)
        self.assertNotIn("<style>", cleaned)
        self.assertTrue(cleaned.startswith("Ceci"))
        
    @patch('requests.get')
    def test_scrape_website(self, mock_get):
        """Test du web scraping"""
        # Mock de la réponse HTTP
        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <body>
                <p>Paragraphe 1</p>
                <script>JavaScript code</script>
                <p>Paragraphe 2</p>
                <style>.css{color:red;}</style>
                <p>Paragraphe 3</p>
            </body>
        </html>
        """
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        chunks = scrape_website(self.sample_url)
        
        self.assertIsInstance(chunks, list)
        self.assertTrue(any("Paragraphe" in chunk for chunk in chunks))
        self.assertFalse(any("JavaScript" in chunk for chunk in chunks))
        self.assertFalse(any("css" in chunk for chunk in chunks))

class TestQdrantIntegration(unittest.TestCase):
    @patch('app.QdrantClient', autospec=True)
    def test_init_qdrant(self, MockQdrantClient):
        """Test de l'initialisation de Qdrant"""
        # Setup des mocks
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        
        # Configuration du mock client
        mock_client = MockQdrantClient.return_value
        mock_client.get_collections.return_value = mock_collections_response
        
        # Configuration avec le bon token
        with patch.dict('os.environ', {
            'QDRANT_API_KEY': 'eyJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3MiOiJyIn0.r7g20UrWnGzTUVxundSq6wKvIXEw57iNnCcuAsOXtaQ',
            'QDRANT_URL': 'http://localhost:6333'
        }):
            # Premier test : création nouvelle collection
            client = init_qdrant()
            
            # Vérifications
            MockQdrantClient.assert_called_once_with(
                url='http://localhost:6333',
                api_key='eyJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3MiOiJyIn0.r7g20UrWnGzTUVxundSq6wKvIXEw57iNnCcuAsOXtaQ'
            )
            mock_client.get_collections.assert_called_once()
            mock_client.create_collection.assert_called_once()

            # Deuxième test : collection existante
            mock_collections_response.collections = [MagicMock(name="knowledge_base")]
            client = init_qdrant()
            mock_client.create_collection.assert_called_once()

class TestVLLMIntegration(unittest.TestCase):
    def setUp(self):
        """Setup pour les tests VLLM"""
        self.vllm_url = "http://localhost:8000/v1/chat/completions"
        
    @patch('requests.post')
    def test_query_vllm(self, mock_post):
        """Test des requêtes VLLM"""
        # Mock de la réponse VLLM
        mock_response = {
            "choices": [{
                "message": {
                    "content": "Réponse test"
                }
            }]
        }
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        from app import query_vllm
        response = query_vllm([{"role": "user", "content": "Test"}])
        
        self.assertIn("choices", response)
        self.assertEqual(response["choices"][0]["message"]["content"], "Réponse test")

if __name__ == '__main__':
    unittest.main() 