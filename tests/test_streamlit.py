import unittest
import streamlit as st
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sidebar, chat_interface

class TestStreamlitInterface(unittest.TestCase):
    @patch('streamlit.sidebar')
    def test_sidebar(self, mock_sidebar):
        """Test des composants de la sidebar"""
        # Mock des composants Streamlit
        mock_expander = MagicMock()
        mock_sidebar.return_value.__enter__.return_value.expander.return_value.__enter__.return_value = mock_expander
        
        # Test de la sidebar
        sidebar()
        
        # Vérifier que les composants sont créés
        mock_sidebar.return_value.__enter__.return_value.title.assert_called_with("⚙️ Configuration")
        
    @patch('streamlit.chat_input')
    @patch('streamlit.chat_message')
    def test_chat_interface(self, mock_chat_message, mock_chat_input):
        """Test de l'interface de chat"""
        # Mock du chat input
        mock_chat_input.return_value = "Test message"
        
        # Mock du chat message
        mock_message = MagicMock()
        mock_chat_message.return_value.__enter__.return_value = mock_message
        
        # Test de l'interface
        with patch('app.query_vllm') as mock_query:
            mock_query.return_value = {
                "choices": [{
                    "message": {
                        "content": "Test response"
                    }
                }]
            }
            chat_interface()
            
            # Vérifier que le message est traité
            mock_message.write.assert_called()

if __name__ == '__main__':
    unittest.main() 