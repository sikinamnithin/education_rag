import os
import logging
import traceback
import uuid
import requests
from datetime import datetime
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import Config

logger = logging.getLogger(__name__)

try:
    import PyPDF2
except ImportError:
    logger.warning("PyPDF2 not installed, PDF processing will not work")
    PyPDF2 = None

class DocumentProcessor:
    def __init__(self):
        logger.info("Initializing DocumentProcessor...")
        
        self.config = Config()
        
        # Setup Azure OpenAI embeddings API
        self.embedding_endpoint = self.config.AZURE_OPENAI_EMBEDDING_ENDPOINT
        self.embedding_api_key = self.config.AZURE_OPENAI_EMBEDDING_API_KEY
        self.embedding_api_version = self.config.AZURE_OPENAI_API_VERSION
        self.embedding_deployment = self.config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        
        logger.info(f"Azure OpenAI embeddings configured - Endpoint: {self.embedding_endpoint}, "
                   f"Deployment: {self.embedding_deployment}")
        
        try:
            self.qdrant_client = QdrantClient(url=self.config.QDRANT_URL)
            logger.info(f"Qdrant client initialized - URL: {self.config.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise
            
        self.collection_name = "documents"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        logger.info(f"Text chunking configured - Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        self._ensure_collection_exists()
        logger.info("DocumentProcessor initialization completed")
    
    def _ensure_collection_exists(self):
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Qdrant collection '{self.collection_name}' already exists - "
                       f"Points: {collection_info.points_count if hasattr(collection_info, 'points_count') else 'Unknown'}")
        except Exception as e:
            logger.info(f"Qdrant collection '{self.collection_name}' not found, creating new collection...")
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                logger.info(f"Qdrant collection '{self.collection_name}' created successfully - "
                           f"Vector size: 1536, Distance: COSINE")
            except Exception as create_error:
                logger.error(f"Failed to create Qdrant collection '{self.collection_name}': {str(create_error)}")
                raise
    
    def _load_document_content(self, file_path: str) -> str:
        """Load document content from file."""
        if file_path.endswith('.pdf'):
            return self._load_pdf(file_path)
        else:
            return self._load_text_file(file_path)
    
    def _load_pdf(self, file_path: str) -> str:
        """Load PDF content using PyPDF2."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _load_text_file(self, file_path: str) -> str:
        """Load text file content."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence or word boundary
            chunk_text = text[start:end]
            last_sentence = chunk_text.rfind('. ')
            last_word = chunk_text.rfind(' ')
            
            if last_sentence > start + self.chunk_size // 2:
                end = start + last_sentence + 1
            elif last_word > start + self.chunk_size // 2:
                end = start + last_word
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end > self.chunk_overlap else end
        
        return chunks
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using Azure OpenAI."""
        # Fix URL formatting - remove trailing slash and ensure proper format
        base_endpoint = self.embedding_endpoint.rstrip('/')
        url = f"{base_endpoint}/openai/deployments/{self.embedding_deployment}/embeddings"
        
        headers = {
            'Content-Type': 'application/json',
            'api-key': self.embedding_api_key
        }
        
        params = {
            'api-version': self.embedding_api_version
        }
        
        data = {
            'input': texts,
            'model': self.embedding_deployment
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data)
            
            # Log the response for debugging if it's an error
            if response.status_code != 200:
                logger.error(f"Embedding API error - Status: {response.status_code}, Response: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            return [item['embedding'] for item in result['data']]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API request failed - URL: {url}, Error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def process_document(self, file_path: str, document_id: int) -> tuple[bool, any]:
        start_time = datetime.now()
        logger.info(f"Starting document processing - Document ID: {document_id}, File: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(f"Document processing failed - {error_msg}")
                return False, error_msg
            
            file_size = os.path.getsize(file_path)
            logger.info(f"File found - Size: {file_size} bytes, Path: {file_path}")
            
            # Load document content
            load_start = datetime.now()
            text_content = self._load_document_content(file_path)
            load_duration = (datetime.now() - load_start).total_seconds()
            
            logger.info(f"Document loaded - Characters: {len(text_content)}, Load time: {load_duration:.2f}s")
            
            # Split into chunks
            split_start = datetime.now()
            chunks = self._chunk_text(text_content)
            split_duration = (datetime.now() - split_start).total_seconds()
            
            logger.info(f"Document chunked - Chunks created: {len(chunks)}, Split time: {split_duration:.2f}s")
            
            # Generate embeddings
            embedding_start = datetime.now()
            embeddings = self._get_embeddings(chunks)
            embedding_duration = (datetime.now() - embedding_start).total_seconds()
            
            logger.info(f"Embeddings generated - Duration: {embedding_duration:.2f}s")
            
            # Create points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'document_id': document_id,
                        'chunk_id': i,
                        'source': os.path.basename(file_path),
                        'file_path': file_path,
                        'chunk_size': len(chunk),
                        'content': chunk
                    }
                )
                points.append(point)
            
            # Store in Qdrant
            store_start = datetime.now()
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            store_duration = (datetime.now() - store_start).total_seconds()
            
            total_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Document processing completed successfully - Document ID: {document_id}, "
                       f"Chunks: {len(chunks)}, Embedding time: {embedding_duration:.2f}s, "
                       f"Store time: {store_duration:.2f}s, Total time: {total_duration:.2f}s")
            
            return True, len(chunks)
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Document processing failed - Document ID: {document_id}, "
                        f"File: {file_path}, Error: {str(e)}, Duration: {total_duration:.2f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False, str(e)
    
    def delete_document(self, document_id: int) -> bool:
        start_time = datetime.now()
        logger.info(f"Starting vector deletion - Document ID: {document_id}, Collection: {self.collection_name}")
        
        try:
            # First check if there are any vectors to delete
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "document_id", "match": {"value": document_id}}
                    ]
                },
                limit=1
            )
            
            points_found = len(search_result[0]) if search_result and len(search_result) > 0 else 0
            
            if points_found == 0:
                logger.warning(f"No vectors found for document ID {document_id} in collection '{self.collection_name}'")
                return True  # Consider this successful as the desired end state is achieved
            
            logger.info(f"Found vectors for document ID {document_id}, proceeding with deletion...")
            
            delete_result = self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "document_id", "match": {"value": document_id}}
                        ]
                    }
                }
            )
            
            deletion_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Vector deletion completed successfully - Document ID: {document_id}, "
                       f"Duration: {deletion_duration:.2f}s, Delete result: {delete_result}")
            
            return True
            
        except Exception as e:
            deletion_duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Vector deletion failed - Document ID: {document_id}, "
                        f"Error: {str(e)}, Duration: {deletion_duration:.2f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False