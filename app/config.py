from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    
    # FAISS Configuration
    faiss_index_path: str = "data/faiss_index"
    faiss_document_store_path: str = "data/document_store.json"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_bearer_token: str
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1500
    chunk_overlap: int = 300
    max_tokens: int = 300
    
    # Vector Store Configuration
    similarity_threshold: float = 0.7
    top_k_results: int = 3
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
