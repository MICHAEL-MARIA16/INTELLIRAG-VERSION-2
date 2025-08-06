#!/usr/bin/env python3
"""
Quick script to diagnose the empty database issue
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

def check_database_status():
    """Check if database has content and configuration is correct"""
    
    # Check environment variables
    print("🔍 Checking Environment Variables:")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    
    print(f"   SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Missing'}")
    print(f"   SUPABASE_KEY: {'✅ Set' if supabase_key else '❌ Missing'}")
    print(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Missing'}")
    print(f"   GOOGLE_DRIVE_FOLDER_ID: {'✅ Set' if drive_folder_id else '❌ Missing'}")
    
    if not all([supabase_url, supabase_key]):
        print("❌ Missing required Supabase credentials!")
        return False
    
    # Check Supabase connection and content
    try:
        print("\n🔍 Checking Supabase Database:")
        supabase = create_client(supabase_url, supabase_key)
        
        # Check raw_files table
        raw_files = supabase.table("raw_files").select("id, filename").execute()
        print(f"   raw_files table: {len(raw_files.data)} files")
        
        # Check processed_chunks table  
        chunks = supabase.table("processed_chunks").select("id").execute()
        print(f"   processed_chunks table: {len(chunks.data)} chunks")
        
        if raw_files.data:
            print("   📄 Sample files:")
            for file in raw_files.data[:3]:
                print(f"      - {file.get('filename', 'Unknown')}")
        else:
            print("   ⚠️  No files found in database!")
            print("\n💡 SOLUTION: You need to run the sync process:")
            print("      python sync.py --mode full")
            print("   OR trigger sync via API:")
            print("      curl -X POST http://localhost:5000/api/sync")
        
        return len(raw_files.data) > 0
        
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return False

def check_google_drive_setup():
    """Check Google Drive configuration"""
    print("\n🔍 Checking Google Drive Setup:")
    
    credentials_file = "credentials.json"
    if os.path.exists(credentials_file):
        print(f"   ✅ credentials.json found")
    else:
        print(f"   ❌ credentials.json missing!")
        print("      Download from Google Cloud Console")
    
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if folder_id:
        print(f"   ✅ Drive folder ID configured: {folder_id[:10]}...")
    else:
        print(f"   ❌ GOOGLE_DRIVE_FOLDER_ID not set!")

if __name__ == "__main__":
    print("🔧 RAG Chatbot Diagnostic Check\n")
    
    db_has_content = check_database_status()
    check_google_drive_setup()
    
    print(f"\n📊 Summary:")
    if db_has_content:
        print("   ✅ Database has content - chatbot should work")
    else:
        print("   ❌ Database is empty - need to run sync first")
        print("\n🚀 Next Steps:")
        print("   1. Ensure Google Drive folder has documents")
        print("   2. Run: python sync.py --mode full")
        print("   3. Wait for sync to complete")
        print("   4. Test chatbot again")