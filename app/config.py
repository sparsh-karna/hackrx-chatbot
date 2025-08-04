from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Google AI Configuration
    google_api_key: str
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str = "hackrx-documents"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_bearer_token: str
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4096
    
    # Vector Store Configuration
    similarity_threshold: float = 0.7
    top_k_results: int = 10
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
