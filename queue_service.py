import redis
import json
import logging
import traceback
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

class QueueService:
    def __init__(self):
        logger.info("Initializing QueueService...")
        
        try:
            self.redis_client = redis.from_url(Config.REDIS_URL)
            self.redis_client.ping()
            logger.info(f"Redis connection established - URL: {Config.REDIS_URL}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
            
        self.queue_name = 'file_processing_queue'
        logger.info(f"QueueService initialized - Queue: {self.queue_name}")
    
    def add_indexing_task(self, document_id, action, **kwargs):
        start_time = datetime.now()
        
        message = {
            'document_id': document_id,
            'action': action,
            **kwargs
        }
        
        try:
            message_json = json.dumps(message)
            queue_length = self.redis_client.lpush(self.queue_name, message_json)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Indexing task added to queue - Document ID: {document_id}, "
                       f"Action: {action}, Queue length: {queue_length}, "
                       f"Duration: {duration:.3f}s, Message: {message}")
            
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to add indexing task - Document ID: {document_id}, "
                        f"Action: {action}, Error: {str(e)}, Duration: {duration:.3f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def get_next_task(self, timeout=10):
        try:
            result = self.redis_client.brpop(self.queue_name, timeout=timeout)
            if result:
                _, message = result
                task = json.loads(message.decode('utf-8'))
                
                logger.debug(f"Task retrieved from queue - Queue: {self.queue_name}, "
                           f"Timeout: {timeout}s, Task: {task}")
                
                return task
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse task JSON from queue - Queue: {self.queue_name}, "
                        f"Raw data: {message.decode('utf-8') if 'message' in locals() else 'N/A'}, "
                        f"Error: {str(e)}")
            return None
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error while getting task - Error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting task from queue - Queue: {self.queue_name}, "
                        f"Timeout: {timeout}s, Error: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def retry_task(self, task):
        """Retry a failed task by adding it back to the end of the queue."""
        start_time = datetime.now()
        
        try:
            message_json = json.dumps(task)
            queue_length = self.redis_client.rpush(self.queue_name, message_json)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Task retried and added to end of queue - Document ID: {task.get('document_id')}, "
                       f"Action: {task.get('action')}, Queue length: {queue_length}, "
                       f"Duration: {duration:.3f}s")
            
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to retry task - Document ID: {task.get('document_id')}, "
                        f"Action: {task.get('action')}, Error: {str(e)}, Duration: {duration:.3f}s")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False