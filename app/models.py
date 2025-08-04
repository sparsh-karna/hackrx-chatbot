from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the document to process")
    questions: List[str] = Field(..., description="List of questions to answer")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to questions")
    
class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    source: str
    page_number: Optional[int] = None
    
class RetrievalResult(BaseModel):
    query: str
    relevant_chunks: List[DocumentChunk]
    answer: str
    confidence_score: float
    reasoning: str
    
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
