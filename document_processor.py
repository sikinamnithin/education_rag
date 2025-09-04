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

try:
    import pymupdf
except ImportError:
    logger.warning("pymupdf not installed, fallback PDF processing will not work")
    pymupdf = None

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
except ImportError:
    logger.warning("pdf2image, pytesseract, or PIL not installed, OCR processing will not work")
    convert_from_path = None
    pytesseract = None
    Image = None

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
        self.batch_size = 100  # Max vectors to upload per batch
        
        logger.info(f"Text chunking configured - Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        logger.info(f"Vector batch processing configured - Batch size: {self.batch_size}")
        
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
        """Load PDF content with page-by-page fallback and image/text detection."""
        
        # Try PyMuPDF first (like youdi-worker approach)
        if pymupdf is not None:
            try:
                logger.info(f"Using PyMuPDF for PDF file: {file_path}")
                text = ""
                
                with pymupdf.open(file_path) as pdf:
                    n_pages = pdf.page_count
                    logger.info(f"PDF has {n_pages} pages")
                    
                    for i in range(n_pages):
                        try:
                            page = pdf[i]
                            
                            # Check if page has images (like youdi-worker)
                            if len(page.get_images()) > 0:
                                logger.debug(f"Page {i+1}/{n_pages} is image-based, using OCR extraction")
                                page_text = self._extract_text_from_image_page(file_path, i)
                            else:
                                logger.debug(f"Page {i+1}/{n_pages} is text-based, using direct text extraction")
                                page_text = page.get_text()
                            
                            if page_text.strip():
                                text += page_text + "\n"
                            
                        except Exception as e:
                            logger.warning(f"PyMuPDF failed on page {i+1}: {e}, trying PyPDF2 fallback")
                            
                            # Page-level fallback to PyPDF2
                            if PyPDF2 is not None:
                                try:
                                    with open(file_path, 'rb') as file:
                                        pdf_reader = PyPDF2.PdfReader(file)
                                        if i < len(pdf_reader.pages):
                                            page_text = pdf_reader.pages[i].extract_text()
                                            if page_text.strip():
                                                text += page_text + "\n"
                                                logger.debug(f"PyPDF2 successfully extracted page {i+1}")
                                except Exception as pdf2_error:
                                    logger.warning(f"PyPDF2 also failed on page {i+1}: {pdf2_error}")
                                    continue
                
                if text.strip():
                    return text.strip()
                else:
                    logger.warning("PyMuPDF extracted no text, falling back to full PyPDF2")
                    
            except (pymupdf.FileDataError, pymupdf.FileNotFoundError) as e:
                logger.warning(f"PyMuPDF failed to open PDF {file_path}: {e}, falling back to PyPDF2")
            except Exception as e:
                logger.warning(f"PyMuPDF failed to process PDF {file_path}: {e}, falling back to PyPDF2")
        
        # Full fallback to PyPDF2
        if PyPDF2 is not None:
            try:
                logger.info(f"Using PyPDF2 fallback for PDF file: {file_path}")
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text.strip():
                                text += page_text + "\n"
                        except Exception as e:
                            logger.warning(f"PyPDF2 failed on a page: {e}")
                            continue
                
                if text.strip():
                    return text.strip()
                    
            except Exception as e:
                logger.warning(f"PyPDF2 failed to read PDF {file_path}: {e}")
        
        raise Exception(f"All PDF readers failed to process {file_path}. The PDF may be corrupted, password-protected, or requires additional libraries.")
    
    def _extract_text_from_image_page(self, file_path: str, page_index: int) -> str:
        """Extract text from image-based PDF page using OCR (like youdi-worker)."""
        
        if convert_from_path is None or pytesseract is None:
            logger.warning("OCR libraries not available, falling back to basic text extraction")
            # Fallback to basic text extraction if OCR not available
            try:
                with pymupdf.open(file_path) as pdf:
                    if page_index < pdf.page_count:
                        return pdf[page_index].get_text()
            except:
                pass
            return ""
        
        try:
            # Convert specific PDF page to image (like youdi-worker approach)
            logger.debug(f"Converting PDF page {page_index + 1} to image for OCR")
            pages = convert_from_path(file_path, first_page=page_index + 1, last_page=page_index + 1)
            
            if not pages:
                logger.warning(f"No image generated for page {page_index + 1}")
                return ""
            
            # Extract text using OCR
            img = pages[0]
            logger.debug(f"Extracting text from image using OCR for page {page_index + 1}")
            text = pytesseract.image_to_string(img)
            
            logger.debug(f"OCR extracted {len(text)} characters from page {page_index + 1}")
            return text.strip()
            
        except Exception as e:
            logger.warning(f"OCR extraction failed for page {page_index + 1}: {e}")
            
            # Fallback to basic text extraction
            try:
                with pymupdf.open(file_path) as pdf:
                    if page_index < pdf.page_count:
                        text = pdf[page_index].get_text()
                        logger.debug(f"Fallback text extraction got {len(text)} characters from page {page_index + 1}")
                        return text
            except Exception as fallback_error:
                logger.warning(f"Fallback text extraction also failed for page {page_index + 1}: {fallback_error}")
            
            return ""
    
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
    
    def _batch_upsert_points(self, points: List[PointStruct]) -> None:
        """Upload points to Qdrant in batches to avoid timeouts."""
        total_points = len(points)
        
        if total_points <= self.batch_size:
            logger.info(f"Uploading {total_points} points in single batch")
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return
        
        logger.info(f"Uploading {total_points} points in batches of {self.batch_size}")
        
        for i in range(0, total_points, self.batch_size):
            batch_end = min(i + self.batch_size, total_points)
            batch_points = points[i:batch_end]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_points + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Uploading batch {batch_num}/{total_batches} - Points {i+1}-{batch_end}")
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=batch_points
            )
        
        logger.info(f"Successfully uploaded all {total_points} points in {total_batches} batches")

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
        
        # Process texts in batches to avoid API limits
        batch_size = 16  # Azure OpenAI embedding API batch limit
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            data = {
                'input': batch_texts,
                'model': self.embedding_deployment
            }
            
            try:
                response = requests.post(url, headers=headers, params=params, json=data)
                
                # Log the response for debugging if it's an error
                if response.status_code != 200:
                    logger.error(f"Embedding API error - Status: {response.status_code}, Response: {response.text}")
                
                response.raise_for_status()
                
                result = response.json()
                batch_embeddings = [item['embedding'] for item in result['data']]
                all_embeddings.extend(batch_embeddings)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Embedding API request failed - URL: {url}, Error: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response body: {e.response.text}")
                raise
        
        return all_embeddings
    
    def process_document(self, file_path: str, document_id: int, user_id: int) -> tuple[bool, any]:
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
                        'user_id': user_id,
                        'document_id': document_id,
                        'chunk_id': i,
                        'source': os.path.basename(file_path),
                        'file_path': file_path,
                        'chunk_size': len(chunk),
                        'content': chunk
                    }
                )
                points.append(point)
            
            # Store in Qdrant using batch processing
            store_start = datetime.now()
            self._batch_upsert_points(points)
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
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # First check if there are any vectors to delete
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1
            )
            
            points_found = len(search_result[0]) if search_result and len(search_result) > 0 else 0
            
            if points_found == 0:
                logger.warning(f"No vectors found for document ID {document_id} in collection '{self.collection_name}'")
                return True  # Consider this successful as the desired end state is achieved
            
            logger.info(f"Found vectors for document ID {document_id}, proceeding with deletion...")
            
            delete_result = self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
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