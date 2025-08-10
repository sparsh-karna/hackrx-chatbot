import logging
from typing import List, Dict, Any
import asyncio
from app.document_processor import DocumentProcessor
from app.text_chunker import TextChunker
from app.vector_store import VectorStore
from app.query_processor import QueryProcessor
from app.config import settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrating all components"""
    
    def __init__(self):
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.vector_store = VectorStore(
            index_path=settings.faiss_index_path,
            document_store_path=settings.faiss_document_store_path,
            dimension=self.text_chunker.get_embedding_dimension()
        )
        self.query_processor = QueryProcessor(
            api_key=settings.openai_api_key
        )
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components"""
        if not self._initialized:
            await self.vector_store.initialize()
            self._initialized = True
            logger.info("RAG Pipeline initialized successfully")
    
    async def process_document_and_questions(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Main pipeline method: process document and answer questions
        
        Args:
            document_url: URL to the document
            questions: List of questions to answer
            
        Returns:
            List of answers corresponding to questions
        """
        try:
            await self.initialize()
            
            # Step 1: Process document
            logger.info(f"Processing document from: {document_url}")
            document_data = await self.document_processor.process_document_from_url(document_url)
            
            # Step 2: Chunk document
            logger.info("Chunking document")
            chunks = self.text_chunker.chunk_document(document_data)
            
            # Step 3: Generate embeddings
            logger.info("Generating embeddings")
            chunk_texts = [doc.page_content for doc in chunks]
            embeddings = self.text_chunker.generate_embeddings(chunk_texts)
            
            # Step 4: Store in vector database
            logger.info("Storing in vector database")
            doc_ids = await self.vector_store.store_documents(chunks, embeddings)
            
            # Step 5: Process each question
            logger.info(f"Processing {len(questions)} questions")
            answers = await self._process_questions(questions, document_url)
            
            logger.info("Pipeline completed successfully")
            return answers
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            # Return error messages for all questions
            return [f"Error processing question: {str(e)}" for _ in questions]
    
    async def _process_questions(self, questions: List[str], source_filter: str = None) -> List[str]:
        """Process multiple questions"""
        answers = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}: {question}")
            
            try:
                # Generate query embedding
                query_embedding = self.text_chunker.generate_embeddings([question])[0]
                
                # Retrieve relevant chunks
                filter_dict = {'source': source_filter} if source_filter else None
                relevant_chunks = await self.vector_store.similarity_search(
                    query_embedding=query_embedding,
                    top_k=settings.top_k_results,
                    filter_dict=filter_dict
                )
                
                # Filter by similarity threshold
                relevant_chunks = [
                    chunk for chunk in relevant_chunks 
                    if chunk['score'] >= settings.similarity_threshold
                ]
                
                if not relevant_chunks:
                    answers.append("I couldn't find relevant information in the document to answer this question.")
                    continue
                
                # Generate answer
                result = await self.query_processor.process_query(
                    query=question,
                    relevant_chunks=relevant_chunks,
                    max_tokens=settings.max_tokens
                )
                
                answers.append(result['answer'])
                
                # Log processing details
                logger.info(f"Question {i} processed - Confidence: {result['confidence_score']:.3f}, "
                          f"Chunks used: {result['chunks_used']}, Tokens: {result['token_usage']}")
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {str(e)}")
                answers.append(f"Error processing this question: {str(e)}")
        
        return answers
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        try:
            await self.initialize()
            
            health_status = {
                'status': 'healthy',
                'components': {}
            }
            
            # Check vector store
            try:
                stats = await self.vector_store.get_index_stats()
                # Safely convert stats to a serializable format
                if hasattr(stats, 'to_dict'):
                    stats_dict = stats.to_dict()
                elif hasattr(stats, '__dict__'):
                    stats_dict = {k: v for k, v in stats.__dict__.items() if not k.startswith('_')}
                else:
                    stats_dict = str(stats)
                    
                health_status['components']['vector_store'] = {
                    'status': 'healthy',
                    'stats': stats_dict
                }
            except Exception as e:
                health_status['components']['vector_store'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'unhealthy'
            
            # Check embedding model
            try:
                test_embedding = self.text_chunker.generate_embeddings(["test"])
                health_status['components']['embedding_model'] = {
                    'status': 'healthy',
                    'dimension': len(test_embedding[0])
                }
            except Exception as e:
                health_status['components']['embedding_model'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'unhealthy'
            
            # Check query processor
            try:
                # Simple test query
                test_result = await self.query_processor.process_query(
                    query="test query",
                    relevant_chunks=[{
                        'content': 'test content',
                        'score': 0.9,
                        'metadata': {'source': 'test'}
                    }]
                )
                health_status['components']['query_processor'] = {
                    'status': 'healthy'
                }
            except Exception as e:
                health_status['components']['query_processor'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['status'] = 'unhealthy'
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
