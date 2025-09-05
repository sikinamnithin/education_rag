import logging
import traceback
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import Config

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        logger.info("Initializing RAGService...")
        
        self.config = Config()
        
        # Setup Azure OpenAI Chat API
        self.chat_endpoint = self.config.AZURE_OPENAI_ENDPOINT
        self.chat_api_key = self.config.AZURE_OPENAI_API_KEY
        self.chat_api_version = self.config.AZURE_OPENAI_API_VERSION
        self.chat_deployment = self.config.AZURE_OPENAI_DEPLOYMENT_NAME
        
        # Setup Azure OpenAI Embeddings API
        self.embedding_endpoint = self.config.AZURE_OPENAI_EMBEDDING_ENDPOINT
        self.embedding_api_key = self.config.AZURE_OPENAI_EMBEDDING_API_KEY
        self.embedding_deployment = self.config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        
        logger.info(f"Chat service configured - Endpoint: {self.chat_endpoint}, "
                   f"Deployment: {self.chat_deployment}")
        logger.info(f"Embeddings service configured - Endpoint: {self.embedding_endpoint}, "
                   f"Deployment: {self.embedding_deployment}")
        
        try:
            self.qdrant_client = QdrantClient(url=self.config.QDRANT_URL)
            logger.info(f"Qdrant client initialized - URL: {self.config.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise
            
        self.collection_name = "documents"
        
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents. 
        
Instructions:
- Use only the information from the provided context to answer questions
- If you cannot find the answer in the context, say "I don't have enough information to answer this question based on the provided documents."
- ellaborate the answer in details. 
"""
        
        logger.info("RAGService initialization completed")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using Azure OpenAI."""
        # Fix URL formatting - remove trailing slash and ensure proper format
        base_endpoint = self.embedding_endpoint.rstrip('/')
        url = f"{base_endpoint}/openai/deployments/{self.embedding_deployment}/embeddings"
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.embedding_api_key
        }
        
        params = {
            'api-version': self.config.AZURE_OPENAI_API_VERSION
        }
        
        data = {
            'input': [text],
            'model': self.embedding_deployment
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data)
            
            # Log the response for debugging if it's an error
            if response.status_code != 200:
                logger.error(f"Embedding API error - Status: {response.status_code}, Response: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            return result['data'][0]['embedding']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API request failed - URL: {url}, Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def _search_relevant_documents(self, query_embedding: List[float], user_id: int = None, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity, filtered by user."""
        search_params = {
            'collection_name': self.collection_name,
            'query_vector': query_embedding,
            'limit': top_k or self.config.RAG_SEARCH_TOP_K,
            'score_threshold': self.config.RAG_SEARCH_SCORE_THRESHOLD
        }
        
        # Add user filter if user_id is provided
        if user_id:
            search_params['query_filter'] = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )
        
        search_result = self.qdrant_client.search(**search_params)
        
        logger.info(f"Qdrant search completed - Found {len(search_result)} points")
        for i, point in enumerate(search_result):
            logger.info(f"Point {i+1}: Score={point.score:.3f}, ID={point.id}, "
                       f"Payload keys={list(point.payload.keys())}")
            content_preview = point.payload.get('content', '')[:100] if point.payload.get('content') else 'NO CONTENT'
            logger.info(f"Point {i+1} content preview: {repr(content_preview)}")
        
        documents = []
        for point in search_result:
            # Handle both old LangChain format and new custom format
            payload = point.payload
            
            # Try to get content from different possible fields
            content = ""
            if 'content' in payload:
                # New format
                content = payload['content']
            elif 'page_content' in payload:
                # Old LangChain format
                content = payload['page_content']
            else:
                # Fallback - check if there's any text-like field
                for key, value in payload.items():
                    if isinstance(value, str) and len(value) > 50:  # Assume text content is longer than 50 chars
                        content = value
                        break
            
            if not content:
                logger.warning(f"No content found in point payload: {payload.keys()}")
                continue
            
            # Handle metadata - extract from nested structure if needed
            metadata = {}
            if 'metadata' in payload and isinstance(payload['metadata'], dict):
                # Old LangChain format with nested metadata
                old_metadata = payload['metadata']
                metadata = {
                    'document_id': old_metadata.get('document_id', payload.get('document_id')),
                    'chunk_id': old_metadata.get('chunk_id', payload.get('chunk_id', 0)),
                    'source': old_metadata.get('source', payload.get('source', 'unknown')),
                    'chunk_size': old_metadata.get('chunk_size', len(content))
                }
            else:
                # New format or flat structure
                metadata = {
                    'document_id': payload.get('document_id'),
                    'chunk_id': payload.get('chunk_id', 0),
                    'source': payload.get('source', 'unknown'),
                    'chunk_size': payload.get('chunk_size', len(content))
                }
            
            documents.append({
                'content': content,
                'score': point.score,
                'metadata': metadata
            })
        
        logger.info(f"Final documents processed: {len(documents)}")
        for i, doc in enumerate(documents):
            logger.info(f"Document {i+1}: Score={doc['score']:.3f}, Content length={len(doc['content'])}, "
                       f"Source={doc['metadata']['source']}")
            content_preview = doc['content'][:150] + "..." if len(doc['content']) > 150 else doc['content']
            logger.info(f"Document {i+1} content: {repr(content_preview)}")
        
        return documents
    
    def _generate_response_with_context(self, query: str, context_documents: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> str:
        # Prepare context from retrieved documents
        context_text = ""
        for i, doc in enumerate(context_documents, 1):
            context_text += f"Document {i} (from {doc['metadata']['source']}):\n{doc['content']}\n\n"
        
        logger.info(f"Context prepared for LLM - Total context length: {len(context_text)} chars")
        context_preview = context_text[:300] + "..." if len(context_text) > 300 else context_text
        logger.info(f"Context preview: {repr(context_preview)}")
        
        if not context_text.strip():
            return "I don't have any relevant documents to answer your question."
        
        # Build conversation messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history for context
        if conversation_history:
            for msg in conversation_history[-8:]:  # Keep last 8 messages for context
                if msg['role'] in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'], 
                        "content": msg['content']
                    })
        
        # Add current query with document context
        current_query = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        messages.append({"role": "user", "content": current_query})
        
        # Fix URL formatting
        base_endpoint = self.chat_endpoint.rstrip('/')
        url = f"{base_endpoint}/openai/deployments/{self.chat_deployment}/chat/completions"
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.chat_api_key
        }
        
        params = {
            'api-version': self.chat_api_version
        }
        
        data = {
            'messages': messages,
            'max_completion_tokens': self.config.AZURE_OPENAI_MAX_COMPLETION_TOKENS
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data)
            
            if response.status_code != 200:
                logger.error(f"Chat API error - Status: {response.status_code}, Response: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            answer_content = result['choices'][0]['message']['content'].strip()
            
            logger.info(f"Azure OpenAI response - Status: {response.status_code}, "
                       f"Answer length: {len(answer_content)}")
            logger.info(f"Raw Azure OpenAI response: {repr(result)}")
            
            return answer_content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat API request failed - URL: {url}, Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            raise

    def _generate_streaming_response_with_context(self, query: str, context_documents: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None):
        """Generate streaming response using Azure OpenAI Chat API."""
        context_text = ""
        for i, doc in enumerate(context_documents, 1):
            context_text += f"Document {i} (from {doc['metadata']['source']}):\n{doc['content']}\n\n"
        
        logger.info(f"Context prepared for streaming LLM - Total context length: {len(context_text)} chars")
        context_preview = context_text[:300] + "..." if len(context_text) > 300 else context_text
        logger.info(f"Context preview: {repr(context_preview)}")
        
        if not context_text.strip():
            yield "I don't have any relevant documents to answer your question."
            return
        
        # Build conversation messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history for context
        if conversation_history:
            for msg in conversation_history[-8:]:  # Keep last 8 messages for context
                if msg['role'] in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'], 
                        "content": msg['content']
                    })
        
        # Add current query with document context
        current_query = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        messages.append({"role": "user", "content": current_query})
        
        # Fix URL formatting
        base_endpoint = self.chat_endpoint.rstrip('/')
        url = f"{base_endpoint}/openai/deployments/{self.chat_deployment}/chat/completions"
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.chat_api_key
        }
        
        params = {
            'api-version': self.chat_api_version
        }
        
        data = {
            'messages': messages,
            'max_completion_tokens': self.config.AZURE_OPENAI_MAX_COMPLETION_TOKENS,
            'stream': True
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data, stream=True)
            
            if response.status_code != 200:
                logger.error(f"Chat API error - Status: {response.status_code}, Response: {response.text}")
                yield f"Error: Unable to process request (Status: {response.status_code})"
                return
            
            logger.info(f"Azure OpenAI streaming response started - Status: {response.status_code}")
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                            
                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get('choices', [])
                            
                            if choices and len(choices) > 0:
                                delta = choices[0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    yield content
                                    
                        except json.JSONDecodeError:
                            # Skip malformed JSON chunks
                            continue
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat API streaming request failed - URL: {url}, Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            yield f"Error: Failed to connect to AI service - {str(e)}"

    def _generate_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate response using Azure OpenAI Chat API."""
        # Prepare context from retrieved documents
        context_text = ""
        for i, doc in enumerate(context_documents, 1):
            context_text += f"Document {i} (from {doc['metadata']['source']}):\n{doc['content']}\n\n"
        
        if not context_text.strip():
            return "I don't have any relevant documents to answer your question."
        
        # Create messages for chat completion
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        # Fix URL formatting - remove trailing slash and ensure proper format
        base_endpoint = self.chat_endpoint.rstrip('/')
        url = f"{base_endpoint}/openai/deployments/{self.chat_deployment}/chat/completions"
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.chat_api_key
        }
        
        params = {
            'api-version': self.chat_api_version
        }
        
        data = {
            'messages': messages,
            'max_completion_tokens': self.config.AZURE_OPENAI_MAX_COMPLETION_TOKENS
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data)
            
            # Log the response for debugging if it's an error
            if response.status_code != 200:
                logger.error(f"Chat API error - Status: {response.status_code}, Response: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat API request failed - URL: {url}, Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def query_with_context(self, question: str, conversation_history: List[Dict[str, str]] = None, user_id: int = None) -> Dict[str, Any]:
        start_time = datetime.now()
        question_length = len(question)
        
        logger.info(f"Starting RAG query with context - User: {user_id}, Length: {question_length} chars, "
                   f"Context messages: {len(conversation_history) if conversation_history else 0}, "
                   f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Generate embedding for the question
            embedding_start = datetime.now()
            query_embedding = self._get_embedding(question)
            embedding_duration = (datetime.now() - embedding_start).total_seconds()
            logger.info(f"Question embedding generated - Duration: {embedding_duration:.3f}s")
            
            # Search for relevant documents
            search_start = datetime.now()
            relevant_docs = self._search_relevant_documents(query_embedding, user_id=user_id)
            search_duration = (datetime.now() - search_start).total_seconds()
            logger.info(f"Document search completed - Found: {len(relevant_docs)} docs, Duration: {search_duration:.3f}s")
            
            # Generate response using chat completion with context
            if not relevant_docs:
                answer = "I don't have any relevant documents to answer your question. Please make sure you have uploaded documents that contain information related to your query."
            else:
                generation_start = datetime.now()
                answer = self._generate_response_with_context(question, relevant_docs, conversation_history)
                generation_duration = (datetime.now() - generation_start).total_seconds()
                logger.info(f"Response generated with context - Duration: {generation_duration:.3f}s")
            
            answer_length = len(answer)
            total_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"RAG query with context completed successfully - Documents found: {len(relevant_docs)}, "
                       f"Answer length: {answer_length} chars, Total time: {total_duration:.3f}s")
            
            return {
                "answer": answer,
                "metadata": {
                    "query_time_ms": round(total_duration * 1000, 2),
                    "embedding_time_ms": round(embedding_duration * 1000, 2) if 'embedding_duration' in locals() else 0,
                    "search_time_ms": round(search_duration * 1000, 2) if 'search_duration' in locals() else 0,
                    "generation_time_ms": round(generation_duration * 1000, 2) if 'generation_duration' in locals() else 0,
                    "answer_length": answer_length
                }
            }
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"RAG query with context failed - Question: {question[:100]}{'...' if len(question) > 100 else ''}, "
                        f"Error: {str(e)}, Duration: {total_duration:.3f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return {
                "answer": "Sorry, I encountered an error while processing your query. Please try again later.",
                "error": str(e),
                "metadata": {
                    "query_time_ms": round(total_duration * 1000, 2),
                    "answer_length": 0
                }
            }

    def query_with_context_streaming(self, question: str, conversation_history: List[Dict[str, str]] = None, user_id: int = None):
        """Stream response with context - yields chunks as they arrive."""
        start_time = datetime.now()
        question_length = len(question)
        
        logger.info(f"Starting streaming RAG query with context - User: {user_id}, Length: {question_length} chars, "
                   f"Context messages: {len(conversation_history) if conversation_history else 0}, "
                   f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Generate embedding for the question
            embedding_start = datetime.now()
            query_embedding = self._get_embedding(question)
            embedding_duration = (datetime.now() - embedding_start).total_seconds()
            logger.info(f"Question embedding generated - Duration: {embedding_duration:.3f}s")
            
            # Search for relevant documents
            search_start = datetime.now()
            relevant_docs = self._search_relevant_documents(query_embedding, user_id=user_id)
            search_duration = (datetime.now() - search_start).total_seconds()
            logger.info(f"Document search completed - Found: {len(relevant_docs)} docs, Duration: {search_duration:.3f}s")
            
            # Generate streaming response
            if not relevant_docs:
                yield "I don't have any relevant documents to answer your question. Please make sure you have uploaded documents that contain information related to your query."
            else:
                generation_start = datetime.now()
                full_answer = ""
                for chunk in self._generate_streaming_response_with_context(question, relevant_docs, conversation_history):
                    full_answer += chunk
                    yield chunk
                
                generation_duration = (datetime.now() - generation_start).total_seconds()
                logger.info(f"Streaming response completed - Duration: {generation_duration:.3f}s, Total length: {len(full_answer)}")
            
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Streaming RAG query with context completed - Documents found: {len(relevant_docs)}, "
                       f"Total time: {total_duration:.3f}s")
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Streaming RAG query with context failed - Question: {question[:100]}{'...' if len(question) > 100 else ''}, "
                        f"Error: {str(e)}, Duration: {total_duration:.3f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            yield "Sorry, I encountered an error while processing your query. Please try again later."

    def query(self, question: str, user_id: int = None) -> Dict[str, Any]:
        start_time = datetime.now()
        question_length = len(question)
        
        logger.info(f"Starting RAG query - Length: {question_length} chars, "
                   f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Generate embedding for the question
            embedding_start = datetime.now()
            query_embedding = self._get_embedding(question)
            embedding_duration = (datetime.now() - embedding_start).total_seconds()
            logger.info(f"Question embedding generated - Duration: {embedding_duration:.3f}s")
            
            # Search for relevant documents
            search_start = datetime.now()
            relevant_docs = self._search_relevant_documents(query_embedding, user_id=user_id)
            search_duration = (datetime.now() - search_start).total_seconds()
            logger.info(f"Document search completed - Found: {len(relevant_docs)} docs, Duration: {search_duration:.3f}s")
            
            # Generate response using chat completion
            if not relevant_docs:
                answer = "I don't have any relevant documents to answer your question. Please make sure you have uploaded documents that contain information related to your query."
            else:
                generation_start = datetime.now()
                answer = self._generate_response(question, relevant_docs)
                generation_duration = (datetime.now() - generation_start).total_seconds()
                logger.info(f"Response generated - Duration: {generation_duration:.3f}s")
            
            answer_length = len(answer)
            total_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"RAG query completed successfully - Documents found: {len(relevant_docs)}, "
                       f"Answer length: {answer_length} chars, Total time: {total_duration:.3f}s")
            
            return {
                "answer": answer,
                "metadata": {
                    "query_time_ms": round(total_duration * 1000, 2),
                    "embedding_time_ms": round(embedding_duration * 1000, 2) if 'embedding_duration' in locals() else 0,
                    "search_time_ms": round(search_duration * 1000, 2) if 'search_duration' in locals() else 0,
                    "generation_time_ms": round(generation_duration * 1000, 2) if 'generation_duration' in locals() else 0,
                    "answer_length": answer_length
                }
            }
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"RAG query failed - Question: {question[:100]}{'...' if len(question) > 100 else ''}, "
                        f"Error: {str(e)}, Duration: {total_duration:.3f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return {
                "answer": "Sorry, I encountered an error while processing your query. Please try again later.",
                "error": str(e),
                "metadata": {
                    "query_time_ms": round(total_duration * 1000, 2),
                    "answer_length": 0
                }
            }

    def query_with_context_streaming(self, question: str, conversation_history: List[Dict[str, str]] = None, user_id: int = None):
        """Stream response with context - yields chunks as they arrive."""
        start_time = datetime.now()
        question_length = len(question)
        
        logger.info(f"Starting streaming RAG query with context - User: {user_id}, Length: {question_length} chars, "
                   f"Context messages: {len(conversation_history) if conversation_history else 0}, "
                   f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        try:
            # Generate embedding for the question
            embedding_start = datetime.now()
            query_embedding = self._get_embedding(question)
            embedding_duration = (datetime.now() - embedding_start).total_seconds()
            logger.info(f"Question embedding generated - Duration: {embedding_duration:.3f}s")
            
            # Search for relevant documents
            search_start = datetime.now()
            relevant_docs = self._search_relevant_documents(query_embedding, user_id=user_id)
            search_duration = (datetime.now() - search_start).total_seconds()
            logger.info(f"Document search completed - Found: {len(relevant_docs)} docs, Duration: {search_duration:.3f}s")
            
            # Generate streaming response
            if not relevant_docs:
                yield "I don't have any relevant documents to answer your question. Please make sure you have uploaded documents that contain information related to your query."
            else:
                generation_start = datetime.now()
                full_answer = ""
                for chunk in self._generate_streaming_response_with_context(question, relevant_docs, conversation_history):
                    full_answer += chunk
                    yield chunk
                
                generation_duration = (datetime.now() - generation_start).total_seconds()
                logger.info(f"Streaming response completed - Duration: {generation_duration:.3f}s, Total length: {len(full_answer)}")
            
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Streaming RAG query with context completed - Documents found: {len(relevant_docs)}, "
                       f"Total time: {total_duration:.3f}s")
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Streaming RAG query with context failed - Question: {question[:100]}{'...' if len(question) > 100 else ''}, "
                        f"Error: {str(e)}, Duration: {total_duration:.3f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            yield "Sorry, I encountered an error while processing your query. Please try again later."
