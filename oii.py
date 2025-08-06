#!/usr/bin/env python3
"""
Database Diagnostic Script for RAG Chatbot
This script helps diagnose issues with database connectivity and data retrieval
"""

import os
import logging
from dotenv import load_dotenv
from supabase_utils import (
    test_connection, 
    get_table_counts, 
    search_files, 
    semantic_search,
    get_file_content,
    supabase
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment_variables():
    """Check if all required environment variables are set"""
    print("🔍 STEP 1: Checking Environment Variables")
    print("=" * 50)
    
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_KEY", 
        "GEMINI_API_KEY"
    ]
    
    optional_vars = [
        "GOOGLE_DRIVE_FOLDER_ID"
    ]
    
    all_good = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive data
            if "KEY" in var or "URL" in var:
                masked_value = value[:10] + "..." + value[-10:] if len(value) > 20 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: NOT SET")
            all_good = False
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"ℹ️  {var}: {value}")
        else:
            print(f"⚠️  {var}: NOT SET (optional)")
    
    print(f"\n{'✅ Environment check passed!' if all_good else '❌ Environment check failed!'}")
    return all_good

def test_supabase_connection():
    """Test basic Supabase connection"""
    print("\n🔍 STEP 2: Testing Supabase Connection")
    print("=" * 50)
    
    try:
        if test_connection():
            print("✅ Supabase connection successful!")
            return True
        else:
            print("❌ Supabase connection failed!")
            return False
    except Exception as e:
        print(f"❌ Supabase connection error: {e}")
        return False

def check_table_structure():
    """Check if required tables exist and their structure"""
    print("\n🔍 STEP 3: Checking Table Structure")
    print("=" * 50)
    
    tables_to_check = ["raw_files", "processed_chunks"]
    
    for table_name in tables_to_check:
        try:
            print(f"\n📋 Checking table: {table_name}")
            
            # Get table info by trying to select with limit 0
            response = supabase.table(table_name).select("*").limit(0).execute()
            print(f"✅ Table '{table_name}' exists and is accessible")
            
            # Get actual count
            count_response = supabase.table(table_name).select("id", count="exact").execute()
            count = count_response.count if hasattr(count_response, 'count') else len(count_response.data)
            print(f"📊 Table '{table_name}' has {count} records")
            
            # Show first few records if they exist
            if count > 0:
                sample_response = supabase.table(table_name).select("*").limit(3).execute()
                if sample_response.data:
                    print(f"📝 Sample records from '{table_name}':")
                    for i, record in enumerate(sample_response.data[:2], 1):
                        # Show key fields only
                        if table_name == "raw_files":
                            filename = record.get('filename', 'N/A')
                            file_id = record.get('drive_file_id', 'N/A')
                            has_text = 'Yes' if record.get('full_text') else 'No'
                            print(f"   {i}. File: {filename}, ID: {file_id}, Has Text: {has_text}")
                        elif table_name == "processed_chunks":
                            filename = record.get('filename', 'N/A')
                            chunk_index = record.get('chunk_index', 'N/A')
                            has_text = 'Yes' if record.get('chunk_text') else 'No'
                            has_embedding = 'Yes' if record.get('embedding') else 'No'
                            print(f"   {i}. File: {filename}, Chunk: {chunk_index}, Has Text: {has_text}, Has Embedding: {has_embedding}")
            
        except Exception as e:
            print(f"❌ Error checking table '{table_name}': {e}")

def test_search_functionality():
    """Test both text and semantic search"""
    print("\n🔍 STEP 4: Testing Search Functionality")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "hello",
        "test",
        "document",
        "information",
        "data"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Testing search with query: '{query}'")
        
        # Test text search
        try:
            text_results = search_files(query, limit=3)
            if text_results:
                print(f"✅ Text search found {len(text_results)} results")
                for i, result in enumerate(text_results[:2], 1):
                    filename = result.get('filename', 'Unknown')
                    source = result.get('source', 'Unknown')
                    print(f"   {i}. {filename} (source: {source})")
            else:
                print("⚠️  Text search returned no results")
        except Exception as e:
            print(f"❌ Text search error: {e}")
        
        # Test semantic search
        try:
            semantic_results = semantic_search(query, limit=3)
            if semantic_results:
                print(f"✅ Semantic search found {len(semantic_results)} results")
                for i, result in enumerate(semantic_results[:2], 1):
                    filename = result.get('filename', 'Unknown')
                    similarity = result.get('similarity', 0)
                    print(f"   {i}. {filename} (similarity: {similarity:.3f})")
            else:
                print("⚠️  Semantic search returned no results")
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
        
        # If we found results, break
        if text_results or semantic_results:
            break
    
    return any([text_results, semantic_results] for query in test_queries 
               if (text_results := search_files(query, limit=1)) or 
                  (semantic_results := semantic_search(query, limit=1)))

def test_file_content_retrieval():
    """Test retrieving content from a specific file"""
    print("\n🔍 STEP 5: Testing File Content Retrieval")
    print("=" * 50)
    
    try:
        # Get first file from raw_files
        response = supabase.table("raw_files").select("*").limit(1).execute()
        
        if not response.data:
            print("⚠️  No files found in raw_files table")
            return False
        
        first_file = response.data[0]
        file_id = first_file.get('drive_file_id')
        filename = first_file.get('filename', 'Unknown')
        
        print(f"📄 Testing content retrieval for: {filename}")
        print(f"🆔 File ID: {file_id}")
        
        # Test get_file_content function
        content = get_file_content(file_id)
        if content:
            full_text = content.get('full_text', '')
            print(f"✅ Successfully retrieved content: {len(full_text)} characters")
            if full_text:
                print(f"📝 Preview: {full_text[:200]}...")
            return True
        else:
            print("❌ Failed to retrieve file content")
            return False
            
    except Exception as e:
        print(f"❌ Error testing file content retrieval: {e}")
        return False

def test_embedding_availability():
    """Check if embeddings are working"""
    print("\n🔍 STEP 6: Testing Embedding Functionality")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("📦 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        embedding = model.encode([test_text])
        print(f"✅ Generated embedding with shape: {embedding.shape}")
        
        return True
        
    except ImportError as e:
        print(f"❌ SentenceTransformers not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

def show_recommendations():
    """Show recommendations based on diagnostic results"""
    print("\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    # Check if we have data
    try:
        raw_files_response = supabase.table("raw_files").select("id", count="exact").execute()
        raw_files_count = raw_files_response.count if hasattr(raw_files_response, 'count') else len(raw_files_response.data)
        
        chunks_response = supabase.table("processed_chunks").select("id", count="exact").execute()
        chunks_count = chunks_response.count if hasattr(chunks_response, 'count') else len(chunks_response.data)
        
        if raw_files_count == 0:
            print("🚨 CRITICAL: No files found in raw_files table!")
            print("   ➡️  You need to upload and process documents first")
            print("   ➡️  Run your document processing script to populate the database")
            print("   ➡️  Make sure your Google Drive integration is working")
        
        if chunks_count == 0:
            print("🚨 CRITICAL: No chunks found in processed_chunks table!")
            print("   ➡️  Documents may not be chunked properly")
            print("   ➡️  Check your text processing and chunking logic")
            print("   ➡️  Ensure embeddings are being generated and stored")
        
        if raw_files_count > 0 and chunks_count == 0:
            print("⚠️  FILES EXIST BUT NO CHUNKS:")
            print("   ➡️  Files are uploaded but not processed into searchable chunks")
            print("   ➡️  Run the chunking and embedding process")
        
        if raw_files_count > 0 and chunks_count > 0:
            print("✅ Database has data! Search issues might be:")
            print("   ➡️  Search query matching problems")
            print("   ➡️  Embedding similarity threshold too high") 
            print("   ➡️  Text preprocessing differences")
            print("   ➡️  Try simpler, more generic search terms")
        
    except Exception as e:
        print(f"❌ Could not check data counts: {e}")

def main():
    """Run full diagnostic"""
    print("🔧 RAG CHATBOT DATABASE DIAGNOSTIC")
    print("=" * 60)
    print("This script will help diagnose database connectivity and data retrieval issues.\n")
    
    # Run all checks
    env_ok = check_environment_variables()
    if not env_ok:
        print("\n🛑 Cannot continue without proper environment variables!")
        return
    
    connection_ok = test_supabase_connection()
    if not connection_ok:
        print("\n🛑 Cannot continue without database connection!")
        return
    
    check_table_structure()
    search_ok = test_search_functionality()
    content_ok = test_file_content_retrieval()
    embedding_ok = test_embedding_availability()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Environment Variables: {'✅' if env_ok else '❌'}")
    print(f"Database Connection: {'✅' if connection_ok else '❌'}")
    print(f"Search Functionality: {'✅' if search_ok else '❌'}")
    print(f"Content Retrieval: {'✅' if content_ok else '❌'}")
    print(f"Embedding System: {'✅' if embedding_ok else '❌'}")
    
    show_recommendations()
    
    print(f"\n{'🎉 Diagnostic Complete!' if all([env_ok, connection_ok]) else '⚠️  Issues Found - Check Recommendations'}")

if __name__ == "__main__":
    main()