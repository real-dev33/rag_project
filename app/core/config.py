# app/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- Qdrant Cloud ----
    qdrant_url: str
    qdrant_api_key: str

    # ---- Embedding Model ----
    # Model name for sentence-transformers (free, local)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Embedding dimension must match the model above (MiniLM = 384)
    embedding_dimension: int = 384
    gemini_api_key: str=""
    groq_api_key: str = ""
    openrouter_api_key: str = ""

    # ---- PostgreSQL (optional, for task logging) ----
    # Leave blank for now if you haven't set up PostgreSQL
    database_url: str = ""

    # ---- General ----
    # App environment: development, production, test
    environment: str = "development"
    debug: bool = True

    class Config:
        # Load from the .env file located in the project root
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance – import this everywhere
settings = Settings()