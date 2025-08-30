"""
Migration script to add chat functionality tables
Run this to add ChatSession and ChatMessage tables to your existing database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from models import db, Document, ChatSession, ChatMessage
from config import Config

def add_chat_tables():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    with app.app_context():
        db.init_app(app)
        
        try:
            # Create all tables (this will only create new ones, not modify existing)
            db.create_all()
            print("Chat tables migration completed successfully!")
            
            # Check if tables exist
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            
            print("Current tables in database:")
            for table in sorted(tables):
                print(f"  ✓ {table}")
                
            if 'chat_sessions' in tables:
                print("\n✓ chat_sessions table created/verified")
            else:
                print("\n✗ chat_sessions table not found")
                
            if 'chat_messages' in tables:
                print("✓ chat_messages table created/verified")
            else:
                print("✗ chat_messages table not found")
                
            if 'documents' in tables:
                print("✓ documents table exists")
            else:
                print("✗ documents table not found - run init_db.py first")
                
        except Exception as e:
            print(f"Error during migration: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    return True

if __name__ == '__main__':
    success = add_chat_tables()
    if success:
        print("\n🎉 Migration completed successfully!")
        print("You can now start the application with chat functionality enabled.")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)