# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style Guidelines

- Do what is asked - no over-engineering
- Write clean, modular, senior developer quality code
- No comments in code - code should be self-explanatory

## Architecture Overview

Document processing and RAG application with Flask API and asynchronous worker architecture:

- **Flask API Server** (`app.py`): Document upload, deletion, listing, querying endpoints
- **Background Worker** (`worker.py`): Processes document indexing from Redis queue  
- **Document Processor** (`document_processor.py`): Document chunking and vector storage
- **RAG Service** (`rag_service.py`): Question-answering with vector search
- **Queue Service** (`queue_service.py`): Redis task queue management

## Services Used

- **PostgreSQL**: Primary database with SQLAlchemy ORM
- **Redis**: Message queue for async processing
- **Qdrant**: Vector database for embeddings
- **Azure OpenAI**: Chat completion and embeddings
- **Flask**: Web framework
- **Docker/Docker Compose**: Containerization

## Development Commands

### Local Development
```bash
pip install -r requirements.txt
python app.py
python worker.py
```

### Docker
```bash
docker-compose up
docker-compose up --build flask-app
docker-compose logs -f worker
```

## Environment Variables

Required in `.env` file:
- `DATABASE_URL`, `REDIS_URL`, `QDRANT_URL`
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_NAME`
- `AZURE_OPENAI_EMBEDDING_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_API_KEY`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

Note: If separate embedding credentials are not provided, the system will fall back to using the main Azure OpenAI credentials for embeddings.


