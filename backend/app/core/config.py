from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = "eyJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3MiOiJyIn0.r7g20UrWnGzTUVxundSq6wKvIXEw57iNnCcuAsOXtaQ"
    VLLM_URL: str = "http://localhost:8000"
    MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    class Config:
        env_file = ".env"

settings = Settings() 