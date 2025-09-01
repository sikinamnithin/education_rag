"""
Migration script to add authentication and user isolation
Run this to add User table and user_id foreign keys to existing tables
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from sqlalchemy import text
from config import Config

def add_authentication():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Import db without model dependencies that require flask_login
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy()
    
    with app.app_context():
        db.init_app(app)
        
        try:
            inspector = db.inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            print("Starting authentication migration...")
            print(f"Existing tables: {existing_tables}")
            
            # Step 1: Create users table
            print("\n📝 Step 1: Creating users table...")
            if 'users' not in existing_tables:
                db.session.execute(text("""
                    CREATE TABLE users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(80) UNIQUE NOT NULL,
                        email VARCHAR(120) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE NOT NULL
                    )
                """))
                db.session.commit()
                print("✓ users table created")
            else:
                print("✓ users table already exists")
            
            # Step 2: Add user_id to documents table
            print("\n📄 Step 2: Adding user_id to documents table...")
            if 'documents' in existing_tables:
                columns = [col['name'] for col in inspector.get_columns('documents')]
                if 'user_id' not in columns:
                    db.session.execute(text("""
                        ALTER TABLE documents 
                        ADD COLUMN user_id INTEGER REFERENCES users(id)
                    """))
                    db.session.commit()
                    print("✓ user_id column added to documents table")
                else:
                    print("✓ user_id column already exists in documents table")
            
            # Step 3: Add user_id to chat_sessions table
            print("\n💬 Step 3: Adding user_id to chat_sessions table...")
            if 'chat_sessions' in existing_tables:
                columns = [col['name'] for col in inspector.get_columns('chat_sessions')]
                if 'user_id' not in columns:
                    db.session.execute(text("""
                        ALTER TABLE chat_sessions 
                        ADD COLUMN user_id INTEGER REFERENCES users(id)
                    """))
                    db.session.commit()
                    print("✓ user_id column added to chat_sessions table")
                else:
                    print("✓ user_id column already exists in chat_sessions table")
            
            # Step 4: Create a default user for existing data
            print("\n👤 Step 4: Creating default user for existing data...")
            
            # Check if any users exist
            user_count = db.session.execute(text("SELECT COUNT(*) FROM users")).scalar()
            
            if user_count == 0:
                print("Creating default user...")
                # Generate password hash manually (simple example - you should use proper hashing)
                from werkzeug.security import generate_password_hash
                password_hash = generate_password_hash('admin123')
                
                db.session.execute(text("""
                    INSERT INTO users (username, email, password_hash, created_at, is_active)
                    VALUES (:username, :email, :password_hash, CURRENT_TIMESTAMP, TRUE)
                """), {
                    'username': 'admin',
                    'email': 'admin@example.com', 
                    'password_hash': password_hash
                })
                db.session.commit()
                
                # Get the created user ID
                user_id = db.session.execute(text("SELECT id FROM users WHERE username = 'admin'")).scalar()
                
                # Update existing documents and chat_sessions to belong to default user
                if 'documents' in existing_tables:
                    updated_docs = db.session.execute(text("""
                        UPDATE documents SET user_id = :user_id WHERE user_id IS NULL
                    """), {"user_id": user_id}).rowcount
                    db.session.commit()
                    print(f"✓ {updated_docs} existing documents assigned to default user")
                
                if 'chat_sessions' in existing_tables:
                    updated_sessions = db.session.execute(text("""
                        UPDATE chat_sessions SET user_id = :user_id WHERE user_id IS NULL
                    """), {"user_id": user_id}).rowcount
                    db.session.commit()
                    print(f"✓ {updated_sessions} existing chat sessions assigned to default user")
                
                print("✓ Default user created: username='admin', password='admin123'")
                print("  🚨 IMPORTANT: Change the default password after first login!")
            else:
                print(f"✓ {user_count} users already exist in database")
            
            # Step 5: Make user_id columns NOT NULL (after data migration)
            print("\n🔒 Step 5: Making user_id columns required...")
            
            if 'documents' in existing_tables:
                # Check if any documents still have NULL user_id
                null_docs = db.session.execute(text("SELECT COUNT(*) FROM documents WHERE user_id IS NULL")).scalar()
                if null_docs == 0:
                    try:
                        db.session.execute(text("ALTER TABLE documents ALTER COLUMN user_id SET NOT NULL"))
                        db.session.commit()
                        print("✓ documents.user_id set to NOT NULL")
                    except:
                        print("⚠️  Could not set documents.user_id to NOT NULL (might already be set)")
                else:
                    print(f"⚠️  Skipping NOT NULL constraint - {null_docs} documents still have NULL user_id")
            
            if 'chat_sessions' in existing_tables:
                # Check if any chat sessions still have NULL user_id
                null_sessions = db.session.execute(text("SELECT COUNT(*) FROM chat_sessions WHERE user_id IS NULL")).scalar()
                if null_sessions == 0:
                    try:
                        db.session.execute(text("ALTER TABLE chat_sessions ALTER COLUMN user_id SET NOT NULL"))
                        db.session.commit()
                        print("✓ chat_sessions.user_id set to NOT NULL")
                    except:
                        print("⚠️  Could not set chat_sessions.user_id to NOT NULL (might already be set)")
                else:
                    print(f"⚠️  Skipping NOT NULL constraint - {null_sessions} chat sessions still have NULL user_id")
            
            print("\n📊 Final verification...")
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            
            for table in sorted(tables):
                columns = [col['name'] for col in inspector.get_columns(table)]
                print(f"  ✓ {table}: {', '.join(columns)}")
            
            print("\n🎉 Authentication migration completed successfully!")
            
        except Exception as e:
            print(f"Error during migration: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    return True

if __name__ == '__main__':
    success = add_authentication()
    if success:
        print("\n🚀 Migration completed successfully!")
        print("\n📋 Next steps:")
        print("1. Start the application: python app.py")
        print("2. Access: http://localhost:5000 (will redirect to login)")
        print("3. Login with: username='admin', password='admin123'")
        print("4. 🚨 CHANGE THE DEFAULT PASSWORD immediately!")
        print("5. Create additional users through the register page")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)