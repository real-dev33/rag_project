from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    # ... other settings

    class Config:
        env_file = ".env"

settings = Settings()