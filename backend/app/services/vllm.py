import requests
from app.core.config import settings

class VLLMService:
    def __init__(self):
        self.url = settings.VLLM_URL
        self.model = settings.MODEL_NAME

    async def generate(self, messages, temperature=0.7, max_tokens=300):
        try:
            response = requests.post(
                f"{self.url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"VLLM error: {str(e)}") 