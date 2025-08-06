#!/usr/bin/env python3
"""
RAG System Diagnostic Script
This script helps diagnose issues with your RAG chatbot system
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment_variables():
    """Check if all required environment variables are set"""
    print("🔍 Checking Environment Variables...")
    
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY", 
        "GEMINI_API_KEY"
    ]
    
    optional_vars = [
        "GOOGLE_DRIVE_FOLDER_ID"
    ]
    
    all_good = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value[:20]}...")
        else:
            print(f"  ❌ {var}: Not set")
            all_good = False
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  📝 {var}: {value[:20]}...")
        else:
            print(f"  ⚠️  {var}: Not set (optional)")
    
    return all_good

def test_supabase_connection():
    """Test Supabase connection and check table contents"""
    print("\n🔍 Testing Supabase Connection...")
    
    try:
        from supabase_utils import supabase, test_connection
        
        # Test basic connection
        if not test_connection():
            print("  ❌ Supabase connection failed")
            return False
        
        print("  ✅ Supabase connection successful")
        
        # Check raw_files table
        try:
            raw_files_response = supabase.table("raw_files").select("*").limit(5).execute()
            print(f"  📊 Raw files count: {len(raw_files_response.data)}")
            
            if raw_files_response.data:
                print("  📁 Sample files:")
                for i, file in enumerate(raw_files_response.data[:3]):
                    print(f"    {i+1}. {file.get('filename', 'Unknown')} (ID: {file.get('id', 'N/A')})")
            else:
                print("  ⚠️  No files found in raw_files table - this could be why search returns no results!")
                
        except Exception as e:
            print(f"  ❌ Error checking raw_files: {e}")
        
        # Check processed_chunks table
        try:
            chunks_response = supabase.table("processed_chunks").select("*").limit(5).execute()
            print(f"  📊 Processed chunks count: {len(chunks_response.data)}")
            
            if chunks_response.data:
                print("  📄 Sample chunks:")
                for i, chunk in enumerate(chunks_response.data[:3]):
                    text_preview = chunk.get('chunk_text', '')[:50] + "..." if chunk.get('chunk_text') else 'No text'
                    print(f"    {i+1}. {text_preview}")
            else:
                print("  ⚠️  No chunks found in processed_chunks table!")
                
        except Exception as e:
            print(f"  ❌ Error checking processed_chunks: {e}")
            
        return True
        
    except ImportError as e:
        print(f"  ❌ Cannot import supabase_utils: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_gemini_connection():
    """Test Gemini API connection"""
    print("\n🔍 Testing Gemini Connection...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("  ❌ GEMINI_API_KEY not found")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test with different models
        models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        
        for model_name in models_to_try:
            try:
                print(f"  🧪 Testing {model_name}...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello, this is a test.")
                print(f"  ✅ {model_name} works! Response: {response.text[:50]}...")
                return True
            except Exception as e:
                print(f"  ❌ {model_name} failed: {e}")
        
        print("  ❌ All Gemini models failed")
        return False
        
    except ImportError:
        print("  ❌ google-generativeai not installed")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def test_search_functionality():
    """Test search functionality with sample queries"""
    print("\n🔍 Testing Search Functionality...")
    
    try:
        from supabase_utils import search_files, semantic_search
        
        test_queries = ["test", "document", "file", "content"]
        
        for query in test_queries:
            print(f"  🔎 Testing search for: '{query}'")
            
            # Test text search
            try:
                text_results = search_files(query, limit=3)
                print(f"    📝 Text search found {len(text_results)} results")
                
                if text_results:
                    for i, result in enumerate(text_results[:2]):
                        filename = result.get('filename', 'Unknown')
                        source = result.get('source', 'Unknown')
                        print(f"      {i+1}. {filename} (source: {source})")
                
            except Exception as e:
                print(f"    ❌ Text search failed: {e}")
            
            # Test semantic search
            try:
                semantic_results = semantic_search(query, limit=3)
                print(f"    🧠 Semantic search found {len(semantic_results)} results")
                
                if semantic_results:
                    for i, result in enumerate(semantic_results[:2]):
                        filename = result.get('filename', 'Unknown')
                        similarity = result.get('similarity', 0)
                        print(f"      {i+1}. {filename} (similarity: {similarity:.3f})")
                
            except Exception as e:
                print(f"    ❌ Semantic search failed: {e}")
            
            print()  # Empty line for readability
            
    except ImportError as e:
        print(f"  ❌ Cannot import search functions: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")

def test_embedding_model():
    """Test sentence transformer embedding model"""
    print("\n🔍 Testing Embedding Model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("  📥 Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        
        print(f"  ✅ Embedding model works! Embedding shape: {embedding.shape}")
        return True
        
    except ImportError:
        print("  ❌ sentence-transformers not installed")
        return False
    except Exception as e:
        print(f"  ❌ Error loading embedding model: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 RAG System Diagnostic Tool")
    print("=" * 50)
    
    # Run all tests
    env_ok = check_environment_variables()
    supabase_ok = test_supabase_connection()
    gemini_ok = test_gemini_connection()
    embedding_ok = test_embedding_model()
    
    if supabase_ok:
        test_search_functionality()
    
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY:")
    print(f"  Environment Variables: {'✅' if env_ok else '❌'}")
    print(f"  Supabase Connection: {'✅' if supabase_ok else '❌'}")
    print(f"  Gemini API: {'✅' if gemini_ok else '❌'}")
    print(f"  Embedding Model: {'✅' if embedding_ok else '❌'}")
    
    if not supabase_ok:
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("  1. Check your Supabase credentials in .env file")
        print("  2. Verify your Supabase project is running")
        print("  3. Make sure tables 'raw_files' and 'processed_chunks' exist")
        print("  4. Run your sync script to populate the database")
    
    if not gemini_ok:
        print("\n🔧 GEMINI ISSUES:")
        print("  1. Check your GEMINI_API_KEY in .env file")
        print("  2. Verify the API key has proper permissions")
        print("  3. Check if your region supports the models")
    
    if supabase_ok and gemini_ok and embedding_ok:
        print("\n🎉 All systems appear to be working!")
        print("   If you're still getting 'no relevant information' responses,")
        print("   the issue might be with your search queries or document content.")

if __name__ == "__main__":
    main()