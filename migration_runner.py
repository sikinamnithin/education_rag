"""
Automatic migration runner for the Flask application
Executes all pending migrations in the correct order
"""
import os
import sys
import logging
import importlib.util
from pathlib import Path

logger = logging.getLogger(__name__)

class MigrationRunner:
    def __init__(self, app):
        self.app = app
        self.migrations_dir = Path(__file__).parent / 'migrations'
        self.migration_order = [
            'init_db.py',
            'add_chat_tables.py', 
            'add_authentication.py'
        ]
    
    def run_migrations(self):
        """Run all migrations in order if they haven't been run yet"""
        logger.info("Starting automatic migration check...")
        
        with self.app.app_context():
            from models import db
            
            # Check if we need to run migrations by looking at database state
            try:
                inspector = db.inspect(db.engine)
                existing_tables = set(inspector.get_table_names())
                logger.info(f"Existing database tables: {sorted(existing_tables)}")
                
                # Define expected tables for each migration
                migration_checks = {
                    'init_db.py': {'documents'},
                    'add_chat_tables.py': {'chat_sessions', 'chat_messages'},
                    'add_authentication.py': {'users'}
                }
                
                for migration_file in self.migration_order:
                    required_tables = migration_checks.get(migration_file, set())
                    
                    if required_tables and not required_tables.issubset(existing_tables):
                        logger.info(f"Running migration: {migration_file}")
                        self._run_single_migration(migration_file)
                        
                        # Refresh table list after migration
                        inspector = db.inspect(db.engine)
                        existing_tables = set(inspector.get_table_names())
                    else:
                        logger.info(f"Skipping {migration_file} - tables already exist")
                
                logger.info("Migration check completed successfully")
                return True
                
            except Exception as e:
                logger.error(f"Migration failed: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return False
    
    def _run_single_migration(self, migration_file):
        """Run a single migration file"""
        migration_path = self.migrations_dir / migration_file
        
        if not migration_path.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_path}")
        
        # Load the migration module
        spec = importlib.util.spec_from_file_location(
            migration_file.replace('.py', ''), 
            migration_path
        )
        migration_module = importlib.util.module_from_spec(spec)
        
        # Add current directory to path so migration can import models
        sys.path.insert(0, str(Path(__file__).parent))
        
        try:
            # Suppress print statements from migration scripts during normal execution
            original_stdout = sys.stdout
            if logger.level > logging.DEBUG:
                sys.stdout = open(os.devnull, 'w')
            
            spec.loader.exec_module(migration_module)
            
            # Execute the main function based on migration type
            if migration_file == 'init_db.py':
                result = migration_module.init_database()
            elif migration_file == 'add_chat_tables.py':
                result = migration_module.add_chat_tables()
            elif migration_file == 'add_authentication.py':
                result = migration_module.add_authentication()
            else:
                logger.warning(f"Unknown migration type: {migration_file}")
                result = False
            
            if not result:
                raise Exception(f"Migration {migration_file} reported failure")
                
            logger.info(f"Successfully completed migration: {migration_file}")
            
        finally:
            # Restore stdout
            if logger.level > logging.DEBUG:
                sys.stdout.close()
                sys.stdout = original_stdout
            sys.path.pop(0)

def run_automatic_migrations(app):
    """Main entry point for automatic migrations"""
    runner = MigrationRunner(app)
    return runner.run_migrations()