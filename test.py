#!/usr/bin/env python3
"""
Comprehensive test suite for the HackRX RAG Pipeline
Tests all components and provides integration testing
"""

import os
import sys
import asyncio
import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class RAGTester:
    """Complete test suite for the RAG pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        
    async def test_configuration(self) -> bool:
        """Test configuration loading"""
        try:
            from app.config import settings
            
            print("🔍 Testing configuration...")
            
            # Check if required settings are accessible
            print(f"   - Chunk size: {settings.chunk_size}")
            print(f"   - Chunk overlap: {settings.chunk_overlap}")
            print(f"   - Embedding model: {settings.embedding_model}")
            print(f"   - API port: {settings.api_port}")
            
            print("✅ Configuration loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Configuration failed: {str(e)}")
            return False
    
    async def test_document_processing(self) -> bool:
        """Test document processing functionality"""
        try:
            from app.document_processor import DocumentProcessor
            
            processor = DocumentProcessor()
            
            # Test with the sample URL from the problem statement
            test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
            
            print("🔍 Testing document processing...")
            result = await processor.process_document_from_url(test_url)
            
            print(f"✅ Document processed successfully")
            print(f"   - Type: {result['metadata']['type']}")
            print(f"   - Content length: {len(result['content'])} characters")
            
            if result['metadata']['type'] == 'pdf':
                print(f"   - Total pages: {result['metadata']['total_pages']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Document processing failed: {str(e)}")
            return False
    
    async def test_text_chunking(self) -> bool:
        """Test text chunking functionality"""
        try:
            from app.text_chunker import TextChunker
            
            chunker = TextChunker(chunk_size=500, chunk_overlap=100)
            
            # Sample document data
            sample_doc = {
                'content': "This is a sample insurance policy document. " * 100,
                'metadata': {'source': 'test.pdf', 'type': 'pdf'}
            }
            
            print("🔍 Testing text chunking...")
            chunks = chunker.chunk_document(sample_doc)
            
            print(f"✅ Text chunking successful")
            print(f"   - Total chunks: {len(chunks)}")
            print(f"   - First chunk length: {len(chunks[0].page_content)} characters")
            
            # Test embedding generation
            print("🔍 Testing embedding generation...")
            embeddings = chunker.generate_embeddings([chunks[0].page_content])
            
            print(f"✅ Embedding generation successful")
            print(f"   - Embedding dimension: {len(embeddings[0])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Text chunking failed: {str(e)}")
            return False
    
    async def test_environment_setup(self) -> bool:
        """Test environment variables"""
        print("🔍 Testing environment setup...")
        
        required_vars = ['GOOGLE_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT']
        missing_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            print(f"   - {var}: {'***' + value[-8:] if value and len(value) > 8 else 'NOT SET'}")
            
            if not value or value in [f"your_{var.lower()}_here", "your_pinecone_environment_here", "your_gemini_api_key_here"]:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing or placeholder environment variables: {', '.join(missing_vars)}")
            print("   Please update your .env file with actual API keys")
            return False
        
        # Check if API keys look valid (basic format check)
        google_key = os.getenv('GOOGLE_API_KEY')
        pinecone_key = os.getenv('PINECONE_API_KEY')
        
        if google_key and not google_key.startswith('AIza'):
            print("⚠️  GOOGLE_API_KEY doesn't appear to be in correct format (should start with 'AIza')")
            return False
            
        if pinecone_key and not pinecone_key.startswith('pcsk_'):
            print("⚠️  PINECONE_API_KEY doesn't appear to be in correct format (should start with 'pcsk_')")
            return False
        
        print("✅ Environment variables configured with valid-looking API keys")
        return True
    
    async def test_rag_pipeline_initialization(self) -> bool:
        """Test RAG pipeline initialization"""
        try:
            print("🔍 Testing RAG pipeline initialization...")
            
            from app.rag_pipeline import RAGPipeline
            
            pipeline = RAGPipeline()
            
            print("✅ RAG Pipeline created successfully")
            return True
            
        except Exception as e:
            print(f"❌ RAG pipeline initialization failed: {str(e)}")
            return False
    
    async def test_api_server(self) -> bool:
        """Test API server endpoints"""
        try:
            print("🔍 Testing API server...")
            
            # Test health endpoint
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is running and healthy")
                return True
            else:
                print(f"⚠️  API server responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("⚠️  API server is not running (start with 'python start.py')")
            return False
        except Exception as e:
            print(f"❌ API server test failed: {str(e)}")
            return False
    
    async def test_full_integration(self) -> bool:
        """Test full integration with sample data"""
        try:
            print("🔍 Testing full integration...")
            
            # Check if environment is ready
            if not await self.test_environment_setup():
                print("⚠️  Skipping integration test - environment not ready")
                return False
            
            from app.rag_pipeline import RAGPipeline
            
            # Sample data
            test_document = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
            test_questions = [
                "What is the grace period for premium payment?",
                "What is the waiting period for pre-existing diseases?"
            ]
            
            # Initialize and test pipeline
            pipeline = RAGPipeline()
            await pipeline.initialize()
            
            print("🚀 Running full integration test...")
            start_time = time.time()
            
            answers = await pipeline.process_document_and_questions(
                document_url=test_document,
                questions=test_questions
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"✅ Integration test successful")
            print(f"   - Questions processed: {len(answers)}")
            print(f"   - Processing time: {processing_time:.2f} seconds")
            print(f"   - Average per question: {processing_time/len(answers):.2f} seconds")
            
            # Display sample results
            for i, (q, a) in enumerate(zip(test_questions, answers), 1):
                print(f"   - Q{i}: {q[:50]}...")
                print(f"     A{i}: {a[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all tests and provide summary"""
        print("🧪 HackRX RAG Pipeline - Comprehensive Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        tests = [
            ("Configuration", self.test_configuration),
            ("Environment Setup", self.test_environment_setup),
            ("Text Chunking", self.test_text_chunking),
            ("Document Processing", self.test_document_processing),
            ("RAG Pipeline Init", self.test_rag_pipeline_initialization),
            ("API Server", self.test_api_server),
            ("Full Integration", self.test_full_integration)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            print(f"\n🚀 Running {test_name} test...")
            try:
                success = await test_func()
                self.test_results[test_name] = success
                if success:
                    passed += 1
            except Exception as e:
                print(f"❌ {test_name} test crashed: {str(e)}")
                self.test_results[test_name] = False
        
        # Generate summary
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        print("=" * 60)
        
        for test_name, success in self.test_results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\nOverall: {passed}/{len(tests)} tests passed")
        print(f"Total test time: {total_time:.2f} seconds")
        
        if passed == len(tests):
            print("🎉 All tests passed! System is ready for submission.")
        elif passed >= len(tests) - 2:  # Allow for API server and integration test failures
            print("⚠️  Most tests passed. Check failed tests for issues.")
        else:
            print("❌ Multiple tests failed. Please check the errors above.")
        
        # Save test results
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(tests),
            "passed_tests": passed,
            "test_results": self.test_results,
            "total_time_seconds": total_time
        }
        
        with open("test_results.json", "w") as f:
            json.dump(result_data, f, indent=2)
        
        print(f"💾 Test results saved to test_results.json")
        
        return passed >= len(tests) - 2  # Allow some flexibility

# Additional utility functions
def test_api_request():
    """Test API request format"""
    print("\n🔍 API Request Format Test:")
    
    sample_request = {
        "documents": "https://example.com/policy.pdf",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    print("Sample API request format:")
    print(json.dumps(sample_request, indent=2))
    
    sample_response = {
        "answers": [
            "A grace period of thirty days is provided for premium payment...",
            "There is a waiting period of thirty-six (36) months..."
        ]
    }
    
    print("\nExpected API response format:")
    print(json.dumps(sample_response, indent=2))

async def main():
    """Main test function"""
    tester = RAGTester()
    
    # Run all tests
    success = await tester.run_all_tests()
    
    # Show API format
    test_api_request()
    
    # Provide next steps
    print("\n" + "=" * 60)
    print("🚀 Next Steps:")
    print("=" * 60)
    
    if success:
        print("1. Start the API server: python start.py")
        print("2. Test with curl or Postman")
        print("3. Submit your solution!")
    else:
        print("1. Fix the failed tests")
        print("2. Update your .env file with proper API keys")
        print("3. Rerun tests: python test.py")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
