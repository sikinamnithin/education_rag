import time
import logging
import sys
import traceback
from datetime import datetime
from flask import Flask
from models import db, Document, ProcessingStatus
from config import Config
from queue_service import QueueService
from document_processor import DocumentProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('worker.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

queue_service = QueueService()
document_processor = DocumentProcessor()

def process_indexing_task(task):
    start_time = datetime.now()
    document_id = task.get('document_id')
    action = task.get('action')
    
    logger.info(f"Processing task - Document ID: {document_id}, Action: {action}, Task: {task}")
    
    if not document_id or not action:
        logger.error(f"Invalid task format - missing document_id or action: {task}")
        return
    
    with app.app_context():
        try:
            document = Document.query.get(document_id)
            if not document:
                logger.error(f"Document not found in database - ID: {document_id}")
                return
            
            logger.info(f"Found document - ID: {document_id}, Filename: {document.original_filename}, "
                       f"Status: {document.status}, File size: {document.file_size} bytes")
            
            if action == 'index':
                logger.info(f"Starting document indexing - Document ID: {document_id}, "
                           f"File: {document.original_filename}, Path: {document.file_path}")
                
                document.status = ProcessingStatus.PROCESSING
                db.session.commit()
                logger.info(f"Updated document status to PROCESSING - Document ID: {document_id}")
                
                process_start = datetime.now()
                success, result = document_processor.process_document(
                    document.file_path, 
                    document.id
                )
                process_duration = (datetime.now() - process_start).total_seconds()
                
                if success:
                    document.status = ProcessingStatus.COMPLETED
                    document.collection_name = document_processor.collection_name
                    logger.info(f"Document indexing completed successfully - Document ID: {document_id}, "
                               f"Chunks created: {result}, Processing time: {process_duration:.2f}s, "
                               f"Collection: {document.collection_name}")
                else:
                    document.status = ProcessingStatus.FAILED
                    logger.error(f"Document indexing failed - Document ID: {document_id}, "
                                f"Error: {result}, Processing time: {process_duration:.2f}s")
                
                db.session.commit()
                logger.info(f"Updated document status in database - Document ID: {document_id}, "
                           f"Final status: {document.status}")
                
            elif action == 'delete':
                logger.info(f"Starting vector deletion - Document ID: {document_id}, "
                           f"Collection: {task.get('collection_name')}")
                
                delete_start = datetime.now()
                success = document_processor.delete_document(document_id)
                delete_duration = (datetime.now() - delete_start).total_seconds()
                
                if success:
                    logger.info(f"Vector deletion completed successfully - Document ID: {document_id}, "
                               f"Processing time: {delete_duration:.2f}s")
                else:
                    logger.error(f"Vector deletion failed - Document ID: {document_id}, "
                                f"Processing time: {delete_duration:.2f}s")
            
            else:
                logger.warning(f"Unknown action received - Document ID: {document_id}, Action: {action}")
                
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Critical error processing task - Document ID: {document_id}, "
                        f"Action: {action}, Error: {str(e)}, Duration: {total_duration:.2f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            if action == 'index':
                try:
                    document = Document.query.get(document_id)
                    if document:
                        document.status = ProcessingStatus.FAILED
                        db.session.commit()
                        logger.info(f"Updated document status to FAILED due to error - Document ID: {document_id}")
                except Exception as db_error:
                    logger.error(f"Failed to update document status to FAILED - Document ID: {document_id}, "
                                f"DB Error: {str(db_error)}")
    
    total_duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Task processing completed - Document ID: {document_id}, Action: {action}, "
               f"Total duration: {total_duration:.2f}s")

def main():
    logger.info("="*50)
    logger.info("Starting document processing worker...")
    logger.info(f"Worker started at: {datetime.now()}")
    logger.info(f"Configuration - Redis URL: {app.config.get('REDIS_URL')}")
    logger.info(f"Configuration - Qdrant URL: {app.config.get('QDRANT_URL')}")
    logger.info(f"Configuration - Database URL: {app.config.get('SQLALCHEMY_DATABASE_URI', '').split('@')[-1] if '@' in app.config.get('SQLALCHEMY_DATABASE_URI', '') else 'Not configured'}")
    logger.info("="*50)
    
    try:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return
    
    task_count = 0
    idle_count = 0
    last_activity_log = datetime.now()
    
    while True:
        try:
            current_time = datetime.now()
            
            task = queue_service.get_next_task()
            if task:
                task_count += 1
                idle_count = 0
                logger.info(f"Retrieved task #{task_count} from queue: {task}")
                
                process_start = datetime.now()
                process_indexing_task(task)
                process_duration = (datetime.now() - process_start).total_seconds()
                
                logger.info(f"Task #{task_count} completed in {process_duration:.2f}s")
                last_activity_log = current_time
                
            else:
                idle_count += 1
                if (current_time - last_activity_log).total_seconds() >= 300:  # Log every 5 minutes when idle
                    logger.info(f"Worker status - Tasks processed: {task_count}, "
                               f"Idle cycles: {idle_count}, Last activity: {last_activity_log}")
                    last_activity_log = current_time
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("="*50)
            logger.info(f"Worker shutdown initiated at: {datetime.now()}")
            logger.info(f"Total tasks processed: {task_count}")
            logger.info("Worker shutting down gracefully...")
            logger.info("="*50)
            break
        except Exception as e:
            logger.error(f"Unexpected error in worker main loop: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Worker will continue after 5 second delay...")
            time.sleep(5)

if __name__ == '__main__':
    main()