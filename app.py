import os
import uuid
import logging
import sys
import traceback
import asyncio
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import redis
import json
import websockets
from models import db, Document, ProcessingStatus, ChatSession, ChatMessage, User
from config import Config
from rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

try:
    db.init_app(app)
    logger.info("Database initialized successfully")
    
    # Run automatic migrations
    from migration_runner import run_automatic_migrations
    logger.info("Running automatic database migrations...")
    migration_success = run_automatic_migrations(app)
    if migration_success:
        logger.info("Database migrations completed successfully")
    else:
        logger.error("Database migrations failed")
        raise Exception("Migration failure - cannot start application")
        
except Exception as e:
    logger.error(f"Failed to initialize database or run migrations: {str(e)}")
    raise

try:
    redis_client = redis.from_url(app.config['REDIS_URL'])
    redis_client.ping()
    logger.info(f"Redis connection established - URL: {app.config['REDIS_URL']}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    raise

try:
    rag_service = RAGService()
    logger.info("RAG service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG service: {str(e)}")
    raise

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class WebSocketChatHandler:
    def __init__(self):
        self.connected_clients = set()
        logger.info("WebSocket chat handler initialized")
    
    async def register_client(self, websocket):
        self.connected_clients.add(websocket)
        logger.info(f"Client connected - Total clients: {len(self.connected_clients)}")
    
    async def unregister_client(self, websocket):
        self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected - Total clients: {len(self.connected_clients)}")
    
    async def send_message(self, websocket, message_type, data):
        message = {
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        await websocket.send(json.dumps(message))
    
    def get_conversation_context(self, session_id, limit=10):
        try:
            messages = ChatMessage.query.filter_by(session_id=session_id)\
                             .order_by(ChatMessage.created_at.desc())\
                             .limit(limit).all()
            
            context = []
            for msg in reversed(messages):
                context.append({
                    'role': msg.role,
                    'content': msg.content
                })
            
            return context
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}")
            return []
    
    async def handle_chat_message(self, websocket, data):
        try:
            question = data.get('message', '').strip()
            session_id = data.get('session_id')
            user_id = data.get('user_id')
            
            if not user_id:
                await self.send_message(websocket, 'error', {'message': 'User authentication required'})
                return
            
            if not question:
                await self.send_message(websocket, 'error', {'message': 'Question is required'})
                return
            
            if not session_id:
                session_id = str(uuid.uuid4())
                
                chat_session = ChatSession(
                    user_id=user_id,
                    session_id=session_id,
                    title=question[:50] + ('...' if len(question) > 50 else '')
                )
                db.session.add(chat_session)
                db.session.commit()
                
                await self.send_message(websocket, 'session_created', {'session_id': session_id})
                
                # Send updated sessions list to refresh sidebar
                await self.handle_get_sessions(websocket, {'user_id': user_id})
            
            user_message = ChatMessage(
                session_id=session_id,
                role='user',
                content=question
            )
            db.session.add(user_message)
            db.session.commit()
            
            await self.send_message(websocket, 'message_received', {
                'message_id': user_message.id,
                'content': question,
                'role': 'user',
                'timestamp': user_message.created_at.isoformat()
            })
            
            conversation_history = self.get_conversation_context(session_id, limit=10)
            
            await self.send_message(websocket, 'typing', {'status': True})
            
            start_time = datetime.now()
            
            # Stream the response
            full_answer = ""
            async for chunk in self.stream_rag_response(question, conversation_history, user_id):
                full_answer += chunk
                await self.send_message(websocket, 'message_chunk', {
                    'content': chunk,
                    'role': 'assistant'
                })
            
            assistant_message = ChatMessage(
                session_id=session_id,
                role='assistant',
                content=full_answer,
                message_metadata={'streaming': True}
            )
            db.session.add(assistant_message)
            db.session.commit()
            
            await self.send_message(websocket, 'typing', {'status': False})
            await self.send_message(websocket, 'message_complete', {
                'message_id': assistant_message.id,
                'content': full_answer,
                'role': 'assistant',
                'timestamp': assistant_message.created_at.isoformat(),
                'metadata': {'streaming': True}
            })
            
            logger.info(f"Chat message processed - Session: {session_id}, "
                       f"Question length: {len(question)}, Answer length: {len(full_answer)}")
            
        except Exception as e:
            logger.error(f"Chat message handling failed: {str(e)}")
            await self.send_message(websocket, 'error', {'message': 'Failed to process message'})
    
    async def handle_get_history(self, websocket, data):
        try:
            session_id = data.get('session_id')
            user_id = data.get('user_id')
            
            if not user_id:
                await self.send_message(websocket, 'error', {'message': 'User authentication required'})
                return
                
            if not session_id:
                await self.send_message(websocket, 'error', {'message': 'Session ID required'})
                return
            
            # Verify the session belongs to the user
            session = ChatSession.query.filter_by(session_id=session_id, user_id=user_id).first()
            if not session:
                await self.send_message(websocket, 'error', {'message': 'Session not found or unauthorized'})
                return
            
            messages = ChatMessage.query.filter_by(session_id=session_id)\
                             .order_by(ChatMessage.created_at).all()
            
            history = [{
                'message_id': msg.id,
                'content': msg.content,
                'role': msg.role,
                'timestamp': msg.created_at.isoformat(),
                'metadata': msg.message_metadata
            } for msg in messages]
            
            await self.send_message(websocket, 'chat_history', {'history': history})
            
        except Exception as e:
            logger.error(f"Get history failed: {str(e)}")
            await self.send_message(websocket, 'error', {'message': 'Failed to retrieve chat history'})
    
    async def handle_get_sessions(self, websocket, data):
        try:
            user_id = data.get('user_id')
            
            if not user_id:
                await self.send_message(websocket, 'error', {'message': 'User authentication required'})
                return
            
            sessions = ChatSession.query.filter_by(user_id=user_id)\
                                       .order_by(ChatSession.updated_at.desc()).all()
            
            sessions_data = [{
                'session_id': session.session_id,
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'message_count': len(session.messages)
            } for session in sessions]
            
            await self.send_message(websocket, 'sessions_list', {'sessions': sessions_data})
            
        except Exception as e:
            logger.error(f"Get sessions failed: {str(e)}")
            await self.send_message(websocket, 'error', {'message': 'Failed to retrieve chat sessions'})
    
    async def handle_message(self, websocket, message):
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'chat_message':
                await self.handle_chat_message(websocket, data)
            elif message_type == 'get_history':
                await self.handle_get_history(websocket, data)
            elif message_type == 'get_sessions':
                await self.handle_get_sessions(websocket, data)
            else:
                await self.send_message(websocket, 'error', {'message': f'Unknown message type: {message_type}'})
                
        except json.JSONDecodeError:
            await self.send_message(websocket, 'error', {'message': 'Invalid JSON format'})
        except Exception as e:
            logger.error(f"Message handling failed: {str(e)}")
            await self.send_message(websocket, 'error', {'message': 'Internal server error'})
    
    async def stream_rag_response(self, question: str, conversation_history, user_id: int):
        """Convert the synchronous streaming RAG response to async generator."""
        import asyncio
        import concurrent.futures
        
        def get_streaming_response():
            return rag_service.query_with_context_streaming(question, conversation_history, user_id)
        
        # Run the blocking generator in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            
            # Get the generator in a separate thread
            stream_gen = await loop.run_in_executor(executor, get_streaming_response)
            
            # Now yield from the generator in a thread-safe way
            while True:
                try:
                    # Get the next chunk in a separate thread
                    chunk = await loop.run_in_executor(executor, lambda: next(stream_gen, None))
                    if chunk is None:
                        break
                    yield chunk
                except StopIteration:
                    break
                except Exception as e:
                    logger.error(f"Error in streaming response: {str(e)}")
                    yield f"Error: {str(e)}"
                    break
    
    async def handle_client(self, websocket):
        await self.register_client(websocket)
        try:
            await self.send_message(websocket, 'connected', {'message': 'Connected to chat server'})
            
            async for message in websocket:
                await self.handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Client handling error: {str(e)}")
        finally:
            await self.unregister_client(websocket)

websocket_handler = WebSocketChatHandler()

async def websocket_handler_with_context(websocket):
    """Wrapper to run WebSocket handler within Flask app context"""
    with app.app_context():
        await websocket_handler.handle_client(websocket)

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    logger.info(f"Home page accessed by user: {current_user.username}")
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            logger.info(f"User logged in successfully: {username}")
            return jsonify({'message': 'Login successful', 'user': user.to_dict()}), 200
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            return jsonify({'error': 'Invalid username or password'}), 401
    
    return render_template('auth.html', mode='login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not username or not email or not password:
            return jsonify({'error': 'Username, email, and password required'}), 400
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Username already exists'}), 409
        
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 409
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            
            login_user(user)
            logger.info(f"New user registered and logged in: {username}")
            return jsonify({'message': 'Registration successful', 'user': user.to_dict()}), 201
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration failed for {username}: {str(e)}")
            return jsonify({'error': 'Registration failed'}), 500
    
    return render_template('auth.html', mode='register')

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    username = current_user.username
    logout_user()
    logger.info(f"User logged out: {username}")
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/current-user')
@login_required
def current_user_info():
    return jsonify(current_user.to_dict()), 200

@app.route('/health', methods=['GET'])
def health_check():
    start_time = datetime.now()
    
    try:
        db_status = 'healthy'
        with app.app_context():
            db.session.execute('SELECT 1')
    except Exception as e:
        db_status = f'unhealthy: {str(e)}'
        logger.error(f"Database health check failed: {str(e)}")
    
    try:
        redis_status = 'healthy'
        redis_client.ping()
    except Exception as e:
        redis_status = f'unhealthy: {str(e)}'
        logger.error(f"Redis health check failed: {str(e)}")
    
    response_time = (datetime.now() - start_time).total_seconds()
    
    health_status = {
        'status': 'healthy' if 'unhealthy' not in db_status and 'unhealthy' not in redis_status else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'response_time_ms': round(response_time * 1000, 2),
        'services': {
            'database': db_status,
            'redis': redis_status
        }
    }
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    logger.info(f"Health check completed - Status: {health_status['status']}, "
               f"Response time: {response_time:.3f}s")
    
    return jsonify(health_status), status_code

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    start_time = datetime.now()
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'Unknown'))
    
    logger.info(f"File upload request received - User: {current_user.username}, Client IP: {client_ip}")
    
    if 'file' not in request.files:
        logger.warning(f"Upload failed - No file provided - Client IP: {client_ip}")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning(f"Upload failed - No file selected - Client IP: {client_ip}")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"Upload failed - File type not allowed: {file.filename} - Client IP: {client_ip}")
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        logger.info(f"Processing file upload - Original: {original_filename}, "
                   f"Unique: {unique_filename}, Path: {file_path}")
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"File saved successfully - Size: {file_size} bytes, Path: {file_path}")
        
        document = Document(
            user_id=current_user.id,
            filename=unique_filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            status=ProcessingStatus.PENDING
        )
        
        db.session.add(document)
        db.session.commit()
        
        logger.info(f"Document record created - ID: {document.id}, "
                   f"Original filename: {original_filename}, Status: {document.status}")
        
        message = {
            'document_id': document.id,
            'action': 'index',
            'file_path': file_path,
            'filename': unique_filename
        }
        
        queue_result = redis_client.lpush('file_processing_queue', json.dumps(message))
        logger.info(f"Task queued for processing - Document ID: {document.id}, "
                   f"Queue length: {queue_result}, Message: {message}")
        
        upload_duration = (datetime.now() - start_time).total_seconds()
        
        response = {
            'message': 'File uploaded successfully',
            'document_id': document.id,
            'status': 'pending',
            'file_size': file_size,
            'upload_time_ms': round(upload_duration * 1000, 2)
        }
        
        logger.info(f"Upload completed successfully - Document ID: {document.id}, "
                   f"Duration: {upload_duration:.3f}s, Client IP: {client_ip}")
        
        return jsonify(response), 201
        
    except Exception as e:
        upload_duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Upload failed - Error: {str(e)}, Duration: {upload_duration:.3f}s, "
                    f"Client IP: {client_ip}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up failed upload file: {file_path}")
            except:
                logger.warning(f"Failed to clean up file: {file_path}")
        
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<int:document_id>', methods=['DELETE'])
@login_required
def delete_document(document_id):
    start_time = datetime.now()
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'Unknown'))
    
    logger.info(f"Document deletion request - Document ID: {document_id}, User: {current_user.username}, Client IP: {client_ip}")
    
    try:
        document = Document.query.filter_by(id=document_id, user_id=current_user.id).first()
        if not document:
            logger.warning(f"Document not found or unauthorized - Document ID: {document_id}, User: {current_user.username}")
            return jsonify({'error': 'Document not found'}), 404
        
        logger.info(f"Document found for deletion - ID: {document_id}, "
                   f"Filename: {document.original_filename}, Status: {document.status}, "
                   f"File path: {document.file_path}")
        
        message = {
            'document_id': document.id,
            'action': 'delete',
            'collection_name': document.collection_name
        }
        
        queue_result = redis_client.lpush('file_processing_queue', json.dumps(message))
        logger.info(f"Vector deletion task queued - Document ID: {document_id}, "
                   f"Queue length: {queue_result}, Collection: {document.collection_name}")
        
        file_deleted = False
        if os.path.exists(document.file_path):
            try:
                os.remove(document.file_path)
                file_deleted = True
                logger.info(f"File deleted from filesystem - Path: {document.file_path}")
            except Exception as file_error:
                logger.warning(f"Failed to delete file from filesystem - Path: {document.file_path}, "
                              f"Error: {str(file_error)}")
        else:
            logger.warning(f"File not found on filesystem - Path: {document.file_path}")
        
        db.session.delete(document)
        db.session.commit()
        
        deletion_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Document deletion completed successfully - Document ID: {document_id}, "
                   f"File deleted: {file_deleted}, Duration: {deletion_duration:.3f}s, "
                   f"Client IP: {client_ip}")
        
        return jsonify({
            'message': 'Document deleted successfully',
            'document_id': document_id,
            'file_deleted': file_deleted,
            'deletion_time_ms': round(deletion_duration * 1000, 2)
        }), 200
        
    except Exception as e:
        deletion_duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Document deletion failed - Document ID: {document_id}, "
                    f"Error: {str(e)}, Duration: {deletion_duration:.3f}s, "
                    f"Client IP: {client_ip}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        status_code = 404 if 'not found' in str(e).lower() else 500
        return jsonify({'error': str(e)}), status_code

@app.route('/documents/<int:document_id>/view')
@login_required
def view_document(document_id):
    try:
        document = Document.query.filter_by(id=document_id, user_id=current_user.id).first()
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        if not os.path.exists(document.file_path):
            return jsonify({'error': 'Document file not found'}), 404
        
        return send_file(
            document.file_path,
            as_attachment=False,
            download_name=document.original_filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error serving document {document_id}: {str(e)}")
        return jsonify({'error': 'Failed to serve document'}), 500

@app.route('/documents', methods=['GET'])
@login_required
def list_documents():
    start_time = datetime.now()
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'Unknown'))
    
    logger.info(f"Documents list request - User: {current_user.username}, Client IP: {client_ip}")
    
    try:
        documents = Document.query.filter_by(user_id=current_user.id).all()
        document_count = len(documents)
        
        response_data = [doc.to_dict() for doc in documents]
        
        response_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Documents list completed successfully - Count: {document_count}, "
                   f"Duration: {response_duration:.3f}s, Client IP: {client_ip}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        response_duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Documents list failed - Error: {str(e)}, "
                    f"Duration: {response_duration:.3f}s, Client IP: {client_ip}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
@login_required
def query_documents():
    start_time = datetime.now()
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'Unknown'))
    
    logger.info(f"Query request received - Client IP: {client_ip}")
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            logger.warning(f"Query failed - No query provided - Client IP: {client_ip}")
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        query_length = len(query)
        
        logger.info(f"Processing query - Length: {query_length} chars, "
                   f"Query: {query[:100]}{'...' if len(query) > 100 else ''}, "
                   f"Client IP: {client_ip}")
        
        rag_start = datetime.now()
        response = rag_service.query(query)
        rag_duration = (datetime.now() - rag_start).total_seconds()
        
        source_count = len(response.get('sources', []))
        answer_length = len(response.get('answer', ''))
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            'query': query,
            'response': response['answer'],
            'sources': response.get('sources', []),
            'metadata': {
                'query_time_ms': round(total_duration * 1000, 2),
                'rag_time_ms': round(rag_duration * 1000, 2),
                'source_count': source_count,
                'answer_length': answer_length
            }
        }
        
        logger.info(f"Query completed successfully - Sources: {source_count}, "
                   f"Answer length: {answer_length} chars, "
                   f"RAG time: {rag_duration:.3f}s, Total time: {total_duration:.3f}s, "
                   f"Client IP: {client_ip}")
        
        return jsonify(result), 200
        
    except Exception as e:
        total_duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Query failed - Error: {str(e)}, Duration: {total_duration:.3f}s, "
                    f"Client IP: {client_ip}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-sessions/<session_id>', methods=['DELETE'])
@login_required
def delete_chat_session(session_id):
    start_time = datetime.now()
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'Unknown'))
    
    logger.info(f"Chat session deletion request - Session ID: {session_id}, User: {current_user.username}, Client IP: {client_ip}")
    
    try:
        # Find the chat session belonging to the current user
        chat_session = ChatSession.query.filter_by(session_id=session_id, user_id=current_user.id).first()
        
        if not chat_session:
            logger.warning(f"Chat session not found - Session ID: {session_id}, Client IP: {client_ip}")
            return jsonify({'error': 'Chat session not found'}), 404
        
        logger.info(f"Chat session found for deletion - Session ID: {session_id}, "
                   f"Title: {chat_session.title}, Message count: {len(chat_session.messages)}")
        
        # Delete all messages in the session (cascade should handle this, but being explicit)
        ChatMessage.query.filter_by(session_id=session_id).delete()
        
        # Delete the chat session
        db.session.delete(chat_session)
        db.session.commit()
        
        deletion_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Chat session deletion completed successfully - Session ID: {session_id}, "
                   f"Duration: {deletion_duration:.3f}s, Client IP: {client_ip}")
        
        return jsonify({
            'message': 'Chat session deleted successfully',
            'session_id': session_id,
            'deletion_time_ms': round(deletion_duration * 1000, 2)
        }), 200
        
    except Exception as e:
        deletion_duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Chat session deletion failed - Session ID: {session_id}, "
                    f"Error: {str(e)}, Duration: {deletion_duration:.3f}s, "
                    f"Client IP: {client_ip}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        return jsonify({'error': str(e)}), 500

async def start_websocket_server():
    logger.info("Starting WebSocket server on 0.0.0.0:8001")
    
    async with websockets.serve(websocket_handler_with_context, "0.0.0.0", 8001):
        logger.info("WebSocket server started successfully on port 8001 (accessible on all interfaces)")
        await asyncio.Future()  # run forever

def run_websocket_server():
    asyncio.run(start_websocket_server())

if __name__ == '__main__':
    logger.info("="*50)
    logger.info("Starting Flask application with WebSocket support...")
    logger.info(f"Application started at: {datetime.now()}")
    logger.info(f"Configuration - Debug mode: {app.config.get('DEBUG', False)}")
    logger.info(f"Configuration - Upload folder: {app.config.get('UPLOAD_FOLDER', 'uploads')}")
    logger.info(f"Configuration - Max file size: {app.config.get('MAX_CONTENT_LENGTH', '16MB')}")
    logger.info("="*50)
    
    # Database tables are automatically migrated on startup
    logger.info("Database tables automatically migrated on startup")
    
    try:
        # Start WebSocket server in a separate thread
        websocket_thread = threading.Thread(target=run_websocket_server, daemon=True)
        websocket_thread.start()
        logger.info("WebSocket server thread started")
        
        # Start Flask application
        logger.info("Flask application starting on http://0.0.0.0:5000")
        logger.info("WebSocket server available on ws://0.0.0.0:8001")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        exit(1)