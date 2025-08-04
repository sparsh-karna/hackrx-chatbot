import logging
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn

from app.models import QueryRequest, QueryResponse, ErrorResponse
from app.rag_pipeline import RAGPipeline
from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global rag_pipeline
    
    # Startup
    logger.info("Initializing RAG Pipeline...")
    rag_pipeline = RAGPipeline()
    try:
        await rag_pipeline.initialize()
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Pipeline...")

# Create FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="A robust RAG pipeline for processing documents and answering contextual queries",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if credentials.credentials != settings.api_bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if rag_pipeline is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": "RAG pipeline not initialized"}
            )
        
        health_status = await rag_pipeline.health_check()
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/api/v1/hackrx/run", 
          response_model=QueryResponse,
          responses={
              400: {"model": ErrorResponse},
              401: {"model": ErrorResponse},
              500: {"model": ErrorResponse}
          })
async def run_query(
    request: QueryRequest,
    _: str = Depends(verify_token)
):
    """
    Main endpoint for processing documents and answering questions
    
    This endpoint:
    1. Downloads and processes the document from the provided URL
    2. Chunks the document and generates embeddings
    3. Stores embeddings in vector database
    4. Processes each question using semantic search and LLM
    5. Returns structured answers
    """
    try:
        logger.info(f"Received request for document: {request.documents}")
        logger.info(f"Number of questions: {len(request.questions)}")
        
        if rag_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized"
            )
        
        # Validate input
        if not request.documents:
            raise HTTPException(
                status_code=400,
                detail="Document URL is required"
            )
        
        if not request.questions:
            raise HTTPException(
                status_code=400,
                detail="At least one question is required"
            )
        
        # Process document and questions
        answers = await rag_pipeline.process_document_and_questions(
            document_url=request.documents,
            questions=request.questions
        )
        
        # Validate answers length matches questions length
        if len(answers) != len(request.questions):
            logger.warning(f"Mismatch between questions ({len(request.questions)}) and answers ({len(answers)})")
            # Pad with error messages if needed
            while len(answers) < len(request.questions):
                answers.append("Unable to process this question")
        
        logger.info(f"Successfully processed {len(answers)} answers")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/hackrx/test")
async def test_endpoint(
    request: QueryRequest,
    _: str = Depends(verify_token)
):
    """
    Test endpoint for debugging - returns detailed processing information
    """
    try:
        if rag_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not initialized"
            )
        
        # Initialize pipeline
        await rag_pipeline.initialize()
        
        # Process document
        document_data = await rag_pipeline.document_processor.process_document_from_url(request.documents)
        
        # Chunk document
        chunks = rag_pipeline.text_chunker.chunk_document(document_data)
        
        # Process first question as test
        if request.questions:
            query_embedding = rag_pipeline.text_chunker.generate_embeddings([request.questions[0]])[0]
            
            return {
                "document_processed": True,
                "document_type": document_data['metadata'].get('type'),
                "total_chunks": len(chunks),
                "embedding_dimension": len(query_embedding),
                "first_chunk_preview": chunks[0].page_content[:200] if chunks else None,
                "test_query": request.questions[0] if request.questions else None
            }
        
        return {
            "document_processed": True,
            "document_type": document_data['metadata'].get('type'),
            "total_chunks": len(chunks),
            "no_questions_provided": True
        }
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Test failed: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
