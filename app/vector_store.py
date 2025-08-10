import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import numpy as np
import faiss
import json
import uuid
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector storage and retrieval using FAISS"""
    
    def __init__(self, index_path: str, document_store_path: str, dimension: int = 3072):
        self.index_path = index_path
        self.document_store_path = document_store_path
        self.dimension = dimension
        self.index = None
        self.document_store = {}  # Maps IDs to document content and metadata
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize or load FAISS index"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Check if index exists
            if os.path.exists(self.index_path):
                # Load existing index
                self.index = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: faiss.read_index(self.index_path)
                )
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            else:
                # Create new index (using L2 distance by default)
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Created new FAISS index with dimension {self.dimension}")
                
            # Load document store if it exists
            if os.path.exists(self.document_store_path):
                with open(self.document_store_path, 'r') as f:
                    self.document_store = json.load(f)
                logger.info(f"Loaded document store with {len(self.document_store)} documents")
            else:
                self.document_store = {}
                
            logger.info("FAISS vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise
    
    async def store_documents(self, documents: List[Document], embeddings: np.ndarray) -> List[str]:
        """
        Store documents and embeddings in FAISS
        
        Args:
            documents: List of Document objects
            embeddings: NumPy array of embeddings
            
        Returns:
            List of document IDs
        """
        try:
            document_ids = []
            
            # Add embeddings to FAISS index
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.add(embeddings)
            )
            
            # Starting ID for this batch
            start_id = self.index.ntotal - len(embeddings)
            
            # Store documents in document store
            for i, doc in enumerate(documents):
                doc_id = str(start_id + i)
                document_ids.append(doc_id)
                
                # Prepare metadata for storage
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
                
                # Store document with its ID
                self.document_store[doc_id] = metadata
            
            # Save document store to disk
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self._save_document_store()
            )
            
            # Save index to disk
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: faiss.write_index(self.index, self.index_path)
            )
            
            logger.info(f"Stored {len(documents)} documents in FAISS")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error storing documents in FAISS: {str(e)}")
            raise
            
    def _save_document_store(self):
        """Save document store to disk"""
        os.makedirs(os.path.dirname(self.document_store_path), exist_ok=True)
        with open(self.document_store_path, 'w') as f:
            json.dump(self.document_store, f)
    
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
            # Reshape query for FAISS search
            query_embedding_reshaped = query_embedding.reshape(1, -1)
            
            # Perform search
            D, I = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.index.search(query_embedding_reshaped, top_k)
            )
            
            results = []
            for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                if idx < 0:  # Invalid index
                    continue
                    
                doc_id = str(idx)
                
                # Convert distance to similarity score (FAISS uses L2 distance)
                # Lower distance = higher similarity
                # Simple normalization: exp(-distance)
                similarity_score = float(np.exp(-distance))
                
                # Get document from store
                doc_data = self.document_store.get(doc_id, {})
                
                # Apply filter if provided
                if filter_dict and not self._matches_filter(doc_data, filter_dict):
                    continue
                    
                results.append({
                    'id': doc_id,
                    'score': similarity_score,
                    'content': doc_data.get('content', ''),
                    'metadata': doc_data
                })
            
            logger.info(f"Retrieved {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
            
    def _matches_filter(self, doc_data: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if document matches filter criteria"""
        for key, value in filter_dict.items():
            if key not in doc_data or doc_data[key] != value:
                return False
        return True
    
    async def delete_by_source(self, source: str):
        """
        Delete all documents from a specific source
        
        Note: This is a simplified implementation that rebuilds the index
        In a production environment, you would want a more efficient approach
        """
        try:
            # Find documents to delete
            docs_to_delete = []
            for doc_id, doc_data in self.document_store.items():
                if doc_data.get('source') == source:
                    docs_to_delete.append(doc_id)
            
            if not docs_to_delete:
                logger.info(f"No documents found from source: {source}")
                return
            
            # Delete documents from document store
            for doc_id in docs_to_delete:
                self.document_store.pop(doc_id, None)
            
            # To implement deletion in FAISS, we would need to recreate the index
            # This is because FAISS doesn't support direct deletion
            # In a production system, you might want to use a different approach
            
            logger.info(f"Deleted {len(docs_to_delete)} documents from source: {source}")
            logger.warning("Note: FAISS index needs to be rebuilt for changes to take effect")
            
            # Save document store
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self._save_document_store()
            )
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            total_vectors = 0
            if self.index:
                total_vectors = self.index.ntotal
            
            stats = {
                "dimension": self.dimension,
                "total_vectors": total_vectors,
                "total_documents": len(self.document_store),
                "index_path": self.index_path,
                "document_store_path": self.document_store_path,
                "sources": self._get_unique_sources()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def _get_unique_sources(self) -> Dict[str, int]:
        """Get count of documents by source"""
        sources = {}
        for doc_data in self.document_store.values():
            source = doc_data.get('source', 'unknown')
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
        return sources
