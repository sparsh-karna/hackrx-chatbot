import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from langchain.schema import Document
import json
import re

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Handles query processing and answer generation using Gemini"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Define system prompt for better context understanding
        self.system_prompt = """
You are an expert document analysis AI specializing in insurance, legal, HR, and compliance domains.

Your task is to analyze document chunks and provide accurate, contextual answers to specific queries.

Key Guidelines:
1. Always base your answers strictly on the provided document context
2. If information is not available in the context, clearly state "Information not available in the provided documents"
3. For insurance/policy questions, focus on specific terms, conditions, waiting periods, coverage limits
4. For legal documents, emphasize precise language and specific clauses
5. Provide clear, concise answers with specific details when available
6. Include relevant clause references or section numbers when mentioned in the context
7. If multiple conditions apply, list them clearly

Format your response as a clear, direct answer without preambles like "Based on the document" unless specifically relevant to the accuracy of the answer.
"""
    
    async def process_query(self, query: str, relevant_chunks: List[Dict[str, Any]], 
                          max_tokens: int = 4096) -> Dict[str, Any]:
        """
        Process query with relevant document chunks and generate answer
        
        Args:
            query: User query
            relevant_chunks: List of relevant document chunks with metadata
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary containing answer, confidence, reasoning, and used chunks
        """
        try:
            # Prepare context from relevant chunks
            context = self._prepare_context(relevant_chunks)
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Generate response
            response = await self._generate_response(prompt, max_tokens)
            
            # Extract answer and reasoning
            answer, reasoning, confidence = self._parse_response(response, relevant_chunks)
            
            return {
                'answer': answer,
                'confidence_score': confidence,
                'reasoning': reasoning,
                'chunks_used': len(relevant_chunks),
                'token_usage': self._estimate_tokens(prompt + response)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'answer': "I apologize, but I encountered an error while processing your query. Please try again.",
                'confidence_score': 0.0,
                'reasoning': f"Error: {str(e)}",
                'chunks_used': 0,
                'token_usage': 0
            }
    
    def _prepare_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context from relevant document chunks"""
        context_parts = []
        
        for i, chunk in enumerate(relevant_chunks, 1):
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            score = chunk.get('score', 0.0)
            
            # Create context entry
            context_entry = f"[Context {i} - Relevance: {score:.3f}]\n"
            
            # Add source information
            if metadata.get('source'):
                context_entry += f"Source: {metadata['source']}\n"
            if metadata.get('page_number'):
                context_entry += f"Page: {metadata['page_number']}\n"
                
            context_entry += f"Content: {content}\n"
            context_parts.append(context_entry)
        
        return "\n---\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for the model"""
        prompt = f"""
{self.system_prompt}

DOCUMENT CONTEXT:
{context}

QUERY: {query}

INSTRUCTIONS:
Analyze the provided document context and answer the query accurately. Focus on:
1. Specific terms, conditions, and requirements
2. Numerical values (amounts, percentages, time periods)
3. Eligibility criteria and exclusions
4. Procedural requirements

Provide a direct, accurate answer based solely on the document context. If the information is not available, state so clearly.

ANSWER:"""
        
        return prompt
    
    async def _generate_response(self, prompt: str, max_tokens: int) -> str:
        """Generate response using Gemini"""
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,  # Low temperature for factual accuracy
                top_p=0.9,
                top_k=40
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _parse_response(self, response: str, relevant_chunks: List[Dict[str, Any]]) -> tuple:
        """Parse response to extract answer, reasoning, and confidence"""
        try:
            # Clean up the response
            answer = response.strip()
            
            # Calculate confidence based on chunk relevance and response quality
            confidence = self._calculate_confidence(answer, relevant_chunks)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(relevant_chunks, answer)
            
            return answer, reasoning, confidence
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return response.strip(), "Unable to generate reasoning", 0.5
    
    def _calculate_confidence(self, answer: str, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on various factors"""
        try:
            if not relevant_chunks:
                return 0.1
            
            # Base confidence on chunk relevance scores
            avg_relevance = sum(chunk.get('score', 0.0) for chunk in relevant_chunks) / len(relevant_chunks)
            
            # Adjust based on answer characteristics
            confidence = avg_relevance
            
            # Higher confidence for specific answers with numbers/dates
            if re.search(r'\d+', answer):
                confidence *= 1.1
            
            # Lower confidence for "not available" answers
            if "not available" in answer.lower() or "not mentioned" in answer.lower():
                confidence *= 0.7
            
            # Ensure confidence is between 0 and 1
            return min(max(confidence, 0.0), 1.0)
            
        except:
            return 0.5
    
    def _generate_reasoning(self, relevant_chunks: List[Dict[str, Any]], answer: str) -> str:
        """Generate reasoning for the answer"""
        try:
            reasoning_parts = []
            
            # Mention number of sources used
            reasoning_parts.append(f"Answer based on analysis of {len(relevant_chunks)} relevant document sections.")
            
            # Mention source types
            sources = set()
            for chunk in relevant_chunks:
                metadata = chunk.get('metadata', {})
                if metadata.get('source'):
                    sources.add(metadata['source'].split('/')[-1])  # Get filename
            
            if sources:
                reasoning_parts.append(f"Sources analyzed: {', '.join(sources)}")
            
            # Mention relevance quality
            if relevant_chunks:
                avg_score = sum(chunk.get('score', 0.0) for chunk in relevant_chunks) / len(relevant_chunks)
                if avg_score > 0.8:
                    reasoning_parts.append("High relevance match found in documents.")
                elif avg_score > 0.6:
                    reasoning_parts.append("Good relevance match found in documents.")
                else:
                    reasoning_parts.append("Moderate relevance match found in documents.")
            
            return " ".join(reasoning_parts)
            
        except:
            return "Standard document analysis performed."
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
