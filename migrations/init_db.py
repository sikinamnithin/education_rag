"""
Database initialization script
Run this to create the initial database schema
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from models import db, Document, ChatSession, ChatMessage
from config import Config

def init_database():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    with app.app_context():
        db.init_app(app)
        
        try:
            # Create all tables
            db.create_all()
            print("Database tables created successfully!")
            
            # Check what tables were created
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            
            print("Created tables:")
            for table in sorted(tables):
                print(f"  ✓ {table}")
                
        except Exception as e:
            print(f"Error creating database tables: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    return True

if __name__ == '__main__':
    success = init_database()
    if success:
        print("\n🎉 Database initialization completed successfully!")
    else:
        print("\n❌ Database initialization failed!")
        sys.exit(1)