import pytest
from unittest.mock import patch, MagicMock
from app.services.vllm import VLLMService

class TestVLLMService:
    @pytest.mark.asyncio
    @patch('app.services.vllm.requests.post')
    async def test_generate(self, mock_post):
        # Setup
        mock_response = {
            "choices": [{
                "message": {
                    "content": "Test response"
                }
            }]
        }
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None

        # Test
        service = VLLMService()
        response = await service.generate([{"role": "user", "content": "Test"}])

        # Verify
        assert "choices" in response
        assert response["choices"][0]["message"]["content"] == "Test response" 