#!/usr/bin/env python3
"""
Cleanup script to remove orphaned vectors from Qdrant that don't have corresponding documents in the SQL database.

This script:
1. Connects to both the SQL database and Qdrant
2. Fetches all document IDs that exist in the SQL database
3. Scans all vectors in Qdrant to find orphaned ones
4. Deletes vectors that reference non-existent document IDs
5. Provides detailed logging and statistics

Usage:
    python cleanup_orphaned_vectors.py [--dry-run] [--batch-size 1000]
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Set, List, Dict, Any
from flask import Flask
from models import db, Document
from config import Config
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'cleanup_orphaned_vectors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class OrphanedVectorCleanup:
    def __init__(self, dry_run: bool = False, batch_size: int = 1000):
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.collection_name = "documents"
        
        logger.info("="*60)
        logger.info("Orphaned Vector Cleanup Script Starting")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE CLEANUP'}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Timestamp: {datetime.now()}")
        logger.info("="*60)
        
        # Initialize Flask app and database
        self.app = Flask(__name__)
        self.app.config.from_object(Config)
        db.init_app(self.app)
        
        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(url=self.app.config['QDRANT_URL'])
            logger.info(f"Connected to Qdrant: {self.app.config['QDRANT_URL']}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
            
        # Test database connection
        try:
            with self.app.app_context():
                db.session.execute('SELECT 1')
                logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def get_existing_document_ids(self) -> Set[int]:
        """Get all document IDs that exist in the SQL database."""
        logger.info("Fetching existing document IDs from SQL database...")
        
        with self.app.app_context():
            try:
                documents = Document.query.with_entities(Document.id).all()
                document_ids = {doc.id for doc in documents}
                
                logger.info(f"Found {len(document_ids)} documents in SQL database")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Document IDs: {sorted(document_ids)}")
                
                return document_ids
                
            except Exception as e:
                logger.error(f"Failed to fetch document IDs from database: {str(e)}")
                raise
    
    def scan_all_vectors(self) -> List[Dict[str, Any]]:
        """Scan all vectors in Qdrant collection and return their metadata."""
        logger.info(f"Scanning all vectors in Qdrant collection '{self.collection_name}'...")
        
        try:
            all_vectors = []
            offset = None
            total_scanned = 0
            
            while True:
                # Scroll through all vectors in batches
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=self.batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # We don't need the actual vectors, just metadata
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                # Extract metadata from each point
                for point in points:
                    vector_info = {
                        'id': point.id,
                        'document_id': point.payload.get('document_id'),
                        'user_id': point.payload.get('user_id'),
                        'source': point.payload.get('source', 'unknown'),
                        'chunk_id': point.payload.get('chunk_id', 0)
                    }
                    all_vectors.append(vector_info)
                
                total_scanned += len(points)
                logger.info(f"Scanned {total_scanned} vectors so far...")
                
                offset = next_offset
                if not next_offset:
                    break
            
            logger.info(f"Completed scanning: found {len(all_vectors)} total vectors in Qdrant")
            return all_vectors
            
        except Exception as e:
            logger.error(f"Failed to scan vectors from Qdrant: {str(e)}")
            raise
    
    def find_orphaned_vectors(self, existing_doc_ids: Set[int], all_vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find vectors that reference non-existent document IDs."""
        logger.info("Analyzing vectors to find orphaned ones...")
        
        orphaned_vectors = []
        orphaned_by_doc_id = {}
        
        for vector in all_vectors:
            document_id = vector.get('document_id')
            
            # Skip vectors without document_id (shouldn't happen but be safe)
            if document_id is None:
                logger.warning(f"Vector {vector['id']} has no document_id in payload")
                continue
            
            # Check if document exists in SQL
            if document_id not in existing_doc_ids:
                orphaned_vectors.append(vector)
                
                # Group by document_id for statistics
                if document_id not in orphaned_by_doc_id:
                    orphaned_by_doc_id[document_id] = []
                orphaned_by_doc_id[document_id].append(vector)
        
        logger.info(f"Found {len(orphaned_vectors)} orphaned vectors across {len(orphaned_by_doc_id)} non-existent documents")
        
        # Log statistics for each orphaned document
        for doc_id, vectors in orphaned_by_doc_id.items():
            sources = set(v.get('source', 'unknown') for v in vectors)
            user_ids = set(v.get('user_id') for v in vectors if v.get('user_id'))
            logger.info(f"  Document ID {doc_id}: {len(vectors)} vectors, Sources: {sources}, Users: {user_ids}")
        
        return orphaned_vectors
    
    def delete_orphaned_vectors(self, orphaned_vectors: List[Dict[str, Any]]) -> int:
        """Delete orphaned vectors from Qdrant."""
        if not orphaned_vectors:
            logger.info("No orphaned vectors to delete")
            return 0
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would delete {len(orphaned_vectors)} orphaned vectors")
            return len(orphaned_vectors)
        
        logger.info(f"Deleting {len(orphaned_vectors)} orphaned vectors...")
        
        try:
            deleted_count = 0
            
            # Group vectors by document_id for efficient batch deletion
            vectors_by_doc_id = {}
            for vector in orphaned_vectors:
                doc_id = vector['document_id']
                if doc_id not in vectors_by_doc_id:
                    vectors_by_doc_id[doc_id] = []
                vectors_by_doc_id[doc_id].append(vector)
            
            # Delete by document_id (more efficient than individual vector IDs)
            for doc_id, vectors in vectors_by_doc_id.items():
                try:
                    delete_result = self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="document_id",
                                    match=MatchValue(value=doc_id)
                                )
                            ]
                        )
                    )
                    
                    deleted_count += len(vectors)
                    logger.info(f"Deleted {len(vectors)} vectors for document ID {doc_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to delete vectors for document ID {doc_id}: {str(e)}")
                    continue
            
            logger.info(f"Successfully deleted {deleted_count} orphaned vectors")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete orphaned vectors: {str(e)}")
            raise
    
    def run_cleanup(self) -> Dict[str, int]:
        """Run the complete cleanup process."""
        start_time = datetime.now()
        
        try:
            # Step 1: Get existing document IDs from SQL
            existing_doc_ids = self.get_existing_document_ids()
            
            # Step 2: Scan all vectors in Qdrant
            all_vectors = self.scan_all_vectors()
            
            # Step 3: Find orphaned vectors
            orphaned_vectors = self.find_orphaned_vectors(existing_doc_ids, all_vectors)
            
            # Step 4: Delete orphaned vectors
            deleted_count = self.delete_orphaned_vectors(orphaned_vectors)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            stats = {
                'total_sql_documents': len(existing_doc_ids),
                'total_qdrant_vectors': len(all_vectors),
                'orphaned_vectors_found': len(orphaned_vectors),
                'vectors_deleted': deleted_count,
                'duration_seconds': round(duration, 2)
            }
            
            logger.info("="*60)
            logger.info("Cleanup Summary:")
            logger.info(f"  SQL documents found: {stats['total_sql_documents']}")
            logger.info(f"  Qdrant vectors scanned: {stats['total_qdrant_vectors']}")
            logger.info(f"  Orphaned vectors found: {stats['orphaned_vectors_found']}")
            logger.info(f"  Vectors deleted: {stats['vectors_deleted']}")
            logger.info(f"  Total duration: {stats['duration_seconds']}s")
            logger.info(f"  Mode: {'DRY RUN' if self.dry_run else 'LIVE CLEANUP'}")
            logger.info("="*60)
            
            return stats
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Clean up orphaned vectors in Qdrant')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run in dry-run mode (no actual deletions)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for scanning vectors (default: 1000)')
    
    args = parser.parse_args()
    
    try:
        cleanup = OrphanedVectorCleanup(dry_run=args.dry_run, batch_size=args.batch_size)
        stats = cleanup.run_cleanup()
        
        if args.dry_run:
            print("\n" + "="*60)
            print("DRY RUN COMPLETED - No actual deletions were performed")
            print("Run without --dry-run to perform actual cleanup")
            print("="*60)
        else:
            print(f"\nCleanup completed successfully! Deleted {stats['vectors_deleted']} orphaned vectors.")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Cleanup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Cleanup failed with error: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())