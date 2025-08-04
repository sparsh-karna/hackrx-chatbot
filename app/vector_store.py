import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
import numpy as np
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector storage and retrieval using Pinecone"""
    
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int = 384):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.pc = None
        self.index = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize Pinecone connection"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Get list of existing indexes
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            # Create index if it doesn't exist
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    async def store_documents(self, documents: List[Document], embeddings: np.ndarray) -> List[str]:
        """
        Store documents and embeddings in Pinecone
        
        Args:
            documents: List of Document objects
            embeddings: NumPy array of embeddings
            
        Returns:
            List of document IDs
        """
        try:
            vectors = []
            document_ids = []
            
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = str(uuid.uuid4())
                document_ids.append(doc_id)
                
                # Prepare metadata for Pinecone
                metadata = {
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', ''),
                    'type': doc.metadata.get('type', ''),
                    'chunk_id': doc.metadata.get('chunk_id', i),
                    'page_number': doc.metadata.get('page_number'),
                    'token_count': doc.metadata.get('token_count', 0)
                }
                
                # Remove None values
                metadata = {k: v for k, v in metadata.items() if v is not None}
                
                vectors.append({
                    'id': doc_id,
                    'values': embedding.tolist(),
                    'metadata': metadata
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda b=batch: self.index.upsert(vectors=b)
                )
            
            logger.info(f"Stored {len(vectors)} documents in Pinecone")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error storing documents in Pinecone: {str(e)}")
            raise
    
    async def similarity_search(self, query_embedding: np.ndarray, top_k: int = 10, 
                              filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results with content and metadata
        """
        try:
            # Perform search
            search_results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.query(
                    vector=query_embedding.tolist(),
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
            )
            
            results = []
            for match in search_results.matches:
                results.append({
                    'id': match.id,
                    'score': float(match.score),
                    'content': match.metadata.get('content', ''),
                    'metadata': match.metadata
                })
            
            logger.info(f"Retrieved {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    async def delete_by_source(self, source: str):
        """Delete all documents from a specific source"""
        try:
            # Note: This is a simplified implementation
            # In production, you might want to implement batch deletion
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.delete(filter={'source': source})
            )
            logger.info(f"Deleted documents from source: {source}")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.describe_index_stats()
            )
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
