from dotenv import load_dotenv
import os
from supabase import create_client, Client
from typing import List, Dict
import uuid
SYNC_USER_UUID = "00000000-0000-0000-0000-000000000001" 

load_dotenv()  # 🔥 This loads variables from .env into os.environ

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def test_connection():
    """Test the Supabase connection and list tables"""
    try:
        # Try to query the raw_files table structure
        response = supabase.table("raw_files").select("*").limit(1).execute()
        print(f"✅ Connection successful. raw_files table exists.")
        
        # Check processed_chunks table
        response2 = supabase.table("processed_chunks").select("*").limit(1).execute()
        print(f"✅ processed_chunks table exists.")
        
        return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def insert_raw_file(file_id: str, file_name: str, file_content: str):
    """Insert a raw file into the Supabase raw_files table."""
    # Map to your actual schema:
    # file_id -> use as unique identifier (you might want to store this in a metadata field)
    # file_name -> filename
    # file_content -> full_text
    
    data = {
        "filename": file_name,
        "full_text": file_content,
        "uploaded_by": SYNC_USER_UUID,   # You can customize this
        # Note: uploaded_at will be auto-set by default
    }

    try:
        print(f"🔄 Attempting to insert raw file: {file_name} ({len(file_content)} chars)")
        
        # First, check if a file with this filename already exists
        existing = supabase.table("raw_files").select("id, filename").eq("filename", file_name).execute()
        
        if existing.data:
            # Update existing record
            record_id = existing.data[0]['id']
            print(f"📋 Updating existing record with ID: {record_id}")
            
            response = supabase.table("raw_files").update(data).eq("id", record_id).execute()
            print(f"✅ Raw file updated: {file_name}")
        else:
            # Insert new record
            print(f"📋 Inserting new record for: {file_name}")
            response = supabase.table("raw_files").insert(data).execute()
            print(f"✅ Raw file inserted: {file_name}")
        
        if response.data:
            print(f"📊 Response data length: {len(response.data)}")
            # Store the Supabase ID for use with chunks
            supabase_file_id = response.data[0]['id']
            print(f"📝 Supabase file ID: {supabase_file_id}")
            return supabase_file_id  # Return the Supabase UUID
        else:
            print(f"⚠️ No data returned from insert/update")
            return None
            
    except Exception as e:
        print(f"❌ Failed to insert raw file {file_name}: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        
        # Try to get more details about the error
        if hasattr(e, 'details'):
            print(f"❌ Error details: {e.details}")
        if hasattr(e, 'message'):
            print(f"❌ Error message: {e.message}")
            
        return None

def insert_processed_chunks(file_id: str, file_name: str, chunks: List[str]):
    """Insert processed chunks into Supabase processed_chunks table."""
    
    if not chunks:
        print(f"⚠️ No chunks to insert for file: {file_name}")
        return False
    
    # First, get the Supabase file ID by filename
    try:
        file_response = supabase.table("raw_files").select("id").eq("filename", file_name).execute()
        if not file_response.data:
            print(f"❌ Could not find raw_files record for {file_name}")
            return False
        
        supabase_file_id = file_response.data[0]['id']
        print(f"📝 Using Supabase file ID: {supabase_file_id}")
        
    except Exception as e:
        print(f"❌ Error getting file ID for {file_name}: {e}")
        return False
    
    # Map to your actual schema:
    # file_id (original) -> we'll store in a metadata field or use supabase file_id
    # file_name -> not needed in chunks table per your schema
    # chunks -> chunk_text
    # chunk_index -> chunk_index
    
    entries = [
        {
            "file_id": supabase_file_id,  # UUID reference to raw_files
            "chunk_index": idx,
            "chunk_text": chunk,
            # embedding_id can be set later if needed
            # created_at will be auto-set
        }
        for idx, chunk in enumerate(chunks)
    ]

    try:
        print(f"🔄 Attempting to insert {len(entries)} chunks for file: {file_name}")
        
        # First delete existing chunks for this file
        delete_response = supabase.table("processed_chunks").delete().eq("file_id", supabase_file_id).execute()
        deleted_count = len(delete_response.data) if delete_response.data else 0
        print(f"🗑️ Deleted {deleted_count} existing chunks")
        
        # Insert new chunks in batches to avoid payload limits
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            print(f"📦 Inserting batch {i//batch_size + 1}: {len(batch)} chunks")
            
            batch_response = supabase.table("processed_chunks").insert(batch).execute()
            
            if batch_response.data:
                total_inserted += len(batch_response.data)
                print(f"✅ Batch inserted: {len(batch_response.data)} chunks")
            else:
                print(f"⚠️ Batch insert returned no data")
        
        print(f"✅ Total chunks inserted for file {file_name}: {total_inserted}")
        
        # Verify the insert
        verify = supabase.table("processed_chunks").select("file_id, chunk_index").eq("file_id", supabase_file_id).execute()
        print(f"✅ Verification: {len(verify.data) if verify.data else 0} chunks found in database")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to insert processed chunks for {file_name}: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        
        # Try to get more details about the error
        if hasattr(e, 'details'):
            print(f"❌ Error details: {e.details}")
        if hasattr(e, 'message'):
            print(f"❌ Error message: {e.message}")
            
        return False

def get_table_counts():
    """Get row counts for both tables"""
    try:
        raw_files_count = supabase.table("raw_files").select("id", count="exact").execute()
        chunks_count = supabase.table("processed_chunks").select("id", count="exact").execute()
        
        print(f"📊 Table counts:")
        print(f"   raw_files: {raw_files_count.count if hasattr(raw_files_count, 'count') else len(raw_files_count.data)}")
        print(f"   processed_chunks: {chunks_count.count if hasattr(chunks_count, 'count') else len(chunks_count.data)}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to get table counts: {e}")
        return False

def list_recent_files(limit=5):
    """List recent files in the database"""
    try:
        print(f"📋 Recent files in raw_files:")
        recent_raw = supabase.table("raw_files").select("id, filename, uploaded_at").order("uploaded_at", desc=True).limit(limit).execute()
        
        if recent_raw.data:
            for file in recent_raw.data:
                print(f"   - {file.get('filename', 'Unknown')} (ID: {str(file.get('id', 'Unknown'))[:8]}...)")
        else:
            print("   No files found")
            
        print(f"📋 Recent chunks in processed_chunks:")
        recent_chunks = supabase.table("processed_chunks").select("file_id, chunk_index, created_at").order("created_at", desc=True).limit(limit).execute()
        
        if recent_chunks.data:
            for chunk in recent_chunks.data:
                print(f"   - File {str(chunk.get('file_id', 'Unknown'))[:8]}... chunk {chunk.get('chunk_index', '?')}")
        else:
            print("   No chunks found")
            
        return True
    except Exception as e:
        print(f"❌ Failed to list recent files: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Supabase connection and operations...")
    
    if test_connection():
        get_table_counts()
        list_recent_files()
        
        # Test insert with dummy data
        print("\n🧪 Testing insert operations with dummy data...")
        supabase_file_id = insert_raw_file(
            "test_file_123",  # This is your original file_id 
            "test_file.txt", 
            "This is test content for debugging"
        )
        
        if supabase_file_id:
            test_chunks = ["chunk 1 test", "chunk 2 test", "chunk 3 test"]
            insert_processed_chunks("test_file_123", "test_file.txt", test_chunks)
            
        print("\n📊 Final counts after test:")
        get_table_counts()
    else:
        print("❌ Connection test failed - check your Supabase credentials and table setup")