import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken

logger = logging.getLogger(__name__)

class TextChunker:
    """Handles text chunking and embedding generation"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
    
    def chunk_document(self, document_data: Dict[str, Any]) -> List[Document]:
        """
        Split document into chunks
        
        Args:
            document_data: Dictionary containing content and metadata
            
        Returns:
            List of LangChain Document objects
        """
        try:
            content = document_data['content']
            metadata = document_data['metadata']
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks)
                })
                
                # Add token count if tokenizer is available
                if self.tokenizer:
                    doc_metadata['token_count'] = len(self.tokenizer.encode(chunk))
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            logger.info(f"Created {len(documents)} chunks from document")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        return self.embedding_model.get_sentence_embedding_dimension()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: 1 token ≈ 4 characters
            return len(text) // 4
