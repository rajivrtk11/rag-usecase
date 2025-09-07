import os
from typing import Optional
# Import BaseSettings from pydantic_settings instead of pydantic
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Google API Settings
    GOOGLE_API_KEY: Optional[str] = None
    
    # ChromaDB Settings
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    
    # PDF Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()