import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-this-in-production')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/education_app')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
    
    AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
    AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
    AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    
    AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT', os.getenv('AZURE_OPENAI_ENDPOINT'))
    AZURE_OPENAI_EMBEDDING_API_KEY = os.getenv('AZURE_OPENAI_EMBEDDING_API_KEY', os.getenv('AZURE_OPENAI_API_KEY'))
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
    
    # Azure OpenAI Chat Completion Configuration
    AZURE_OPENAI_MAX_COMPLETION_TOKENS = int(os.getenv('AZURE_OPENAI_MAX_COMPLETION_TOKENS', 1000))
    
    # RAG Configuration
    RAG_SEARCH_TOP_K = int(os.getenv('RAG_SEARCH_TOP_K', 5))
    RAG_SEARCH_SCORE_THRESHOLD = float(os.getenv('RAG_SEARCH_SCORE_THRESHOLD', 0.5))
    
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size