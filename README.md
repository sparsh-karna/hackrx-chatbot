# RAG Pipeline for Document Analysis

A production-ready Retrieval-Augmented Generation (RAG) pipeline that processes documents and answers questions using advanced AI techniques. Built with FastAPI, Google Gemini LLM, Pinecone vector database, and LangChain framework.

## 🚀 How the Model Works

### Architecture Overview

The RAG pipeline follows a sophisticated multi-stage approach to understand and answer questions from your documents:

```
Document → Text Extraction → Chunking → Embeddings → Vector Storage → Retrieval → LLM Answer Generation
```

### 1. Document Processing Engine
- **Multi-format Support**: Handles PDF, DOCX, and email files seamlessly
- **Intelligent Text Extraction**: Uses specialized parsers for each format to maintain document structure
- **Content Preprocessing**: Cleans and normalizes text while preserving semantic meaning

### 2. Smart Text Chunking
- **Semantic Chunking**: Breaks documents into meaningful 1000-character chunks with 200-character overlap
- **Context Preservation**: Maintains relationships between related content sections
- **Optimized Retrieval**: Ensures each chunk contains sufficient context for accurate question answering

### 3. Advanced Embedding System
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` model for high-quality 384-dimensional embeddings
- **Semantic Understanding**: Captures deep semantic relationships between text segments
- **Efficient Encoding**: Processes chunks in parallel for optimal performance

### 4. Vector Database Operations
- **Pinecone Integration**: Leverages serverless vector database for scalable similarity search
- **Intelligent Indexing**: Stores embeddings with metadata for enhanced retrieval
- **Similarity Search**: Uses cosine similarity with optimized threshold (0.3) for relevant context retrieval

### 5. LLM-Powered Answer Generation
- **Google Gemini Integration**: Uses `gemini-2.0-flash` model for natural language understanding
- **Context-Aware Responses**: Combines retrieved document chunks with user questions
- **Domain Expertise**: Specialized prompts for insurance, legal, and technical document analysis

### 6. Query Processing Flow

When you ask a question, here's what happens:

1. **Question Analysis**: The system analyzes your question for intent and context
2. **Semantic Retrieval**: Converts your question to embeddings and finds relevant document chunks
3. **Context Assembly**: Combines the most relevant chunks (similarity > 0.3) into coherent context
4. **Answer Generation**: Gemini LLM processes the context and generates accurate, detailed answers
5. **Response Formatting**: Returns structured answers with source attribution

### Key Features

- **High Accuracy**: Optimized similarity thresholds ensure relevant context retrieval
- **Scalable**: Handles documents from single pages to hundreds of pages
- **Fast Response**: Parallel processing and efficient vector operations
- **Source Attribution**: Tracks which document sections support each answer
- **RESTful API**: Clean endpoints for document upload and question answering

## 🏃‍♂️ Quick Start

### Prerequisites

1. **Python Environment**: Python 3.8+ required
2. **API Keys**: You'll need Google AI and Pinecone API keys

### Environment Setup

Create a `.env` file in the root directory:

```env
# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Pinecone Configuration  
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-pipeline
```

### Installation & Launch

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python main.py
   ```

   The server will start on `http://localhost:8000`

3. **Verify Setup**:
   ```bash
   curl http://localhost:8000/health
   ```

## 📚 API Usage

### Upload and Process Documents

```bash
curl -X POST "http://localhost:8000/process" 
  -F "file=@your_document.pdf" 
  -F "questions=What is the coverage period?" 
  -F "questions=What are the exclusions?"
```

### Interactive Documentation

Visit `http://localhost:8000/docs` for the complete interactive API documentation with:
- Live API testing interface
- Request/response examples
- Schema definitions
- Authentication details

## 🔧 Architecture Components

### Core Modules

- **`rag_pipeline.py`**: Main orchestrator coordinating all components
- **`document_processor.py`**: Multi-format document parsing and text extraction
- **`text_chunker.py`**: Intelligent semantic text segmentation
- **`embeddings.py`**: Sentence transformer embedding generation
- **`vector_store.py`**: Pinecone vector database operations
- **`query_processor.py`**: Gemini LLM integration and answer generation
- **`main.py`**: FastAPI server and API endpoints

### Technology Stack

- **Backend Framework**: FastAPI for high-performance async API
- **LLM Engine**: Google Gemini 2.0 Flash for advanced language understanding
- **Vector Database**: Pinecone serverless for scalable similarity search
- **ML Framework**: LangChain for document processing workflows
- **Embeddings**: Sentence Transformers for semantic text representation

## 🎯 Performance Characteristics

- **Processing Speed**: Handles 25+ page documents in seconds
- **Accuracy**: Optimized similarity threshold (0.3) for relevant context retrieval
- **Scalability**: Serverless architecture supports high concurrency
- **Memory Efficiency**: Chunked processing prevents memory overflow
- **API Response**: Sub-second response times for most queries

## 🔍 Example Use Cases

- **Insurance Policy Analysis**: Extract coverage details, exclusions, and terms
- **Legal Document Review**: Find specific clauses, obligations, and definitions
- **Technical Documentation**: Answer implementation questions and find procedures
- **Research Papers**: Extract key findings, methodologies, and conclusions

The system is optimized for complex document analysis where traditional search falls short, providing contextual understanding and intelligent answer synthesis from your document corpus.
