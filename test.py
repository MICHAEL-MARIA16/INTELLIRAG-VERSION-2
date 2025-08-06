#!/usr/bin/env python3
"""
Test script to verify semantic search is working properly
Run this to diagnose any issues with your semantic search setup
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from supabase_utils import (
        test_connection,
        semantic_search,
        embed_query,
        test_semantic_search_functionality,
        EMBEDDINGS_AVAILABLE,
        supabase
    )
    print("✅ Successfully imported supabase_utils")
except ImportError as e:
    print(f"❌ Failed to import supabase_utils: {e}")
    sys.exit(1)

def test_embedding_generation():
    """Test if we can generate embeddings"""
    print("\n🧪 TESTING EMBEDDING GENERATION")
    print("=" * 50)
    
    if not EMBEDDINGS_AVAILABLE:
        print("❌ SentenceTransformers not available!")
        return False
    
    test_query = "artificial intelligence and machine learning"
    embedding = embed_query(test_query)
    
    if embedding:
        print(f"✅ Successfully generated embedding for: '{test_query}'")
        print(f"📊 Embedding dimensions: {len(embedding)}")
        print(f"📊 Sample values: {embedding[:5]}...")
        return True
    else:
        print(f"❌ Failed to generate embedding for: '{test_query}'")
        return False

def test_database_embeddings():
    """Test if database has embeddings"""
    print("\n🧪 TESTING DATABASE EMBEDDINGS")
    print("=" * 50)
    
    try:
        # Check if we have chunks with embeddings
        response = supabase.table("processed_chunks").select("id, embedding").not_.is_("embedding", "null").limit(5).execute()
        
        if response.data:
            print(f"✅ Found {len(response.data)} chunks with embeddings in database")
            
            # Check embedding dimensions
            for i, chunk in enumerate(response.data):
                embedding = chunk.get('embedding')
                if embedding:
                    print(f"   Chunk {i+1}: {len(embedding)} dimensions")
                else:
                    print(f"   Chunk {i+1}: No embedding")
            return True
        else:
            print("❌ No chunks with embeddings found in database!")
            return False
            
    except Exception as e:
        print(f"❌ Error checking database embeddings: {e}")
        return False

def test_rpc_function():
    """Test if the match_chunks RPC function exists and works"""
    print("\n🧪 TESTING RPC FUNCTION")
    print("=" * 50)
    
    try:
        # Generate a test embedding
        test_embedding = embed_query("test query")
        if not test_embedding:
            print("❌ Cannot test RPC - embedding generation failed")
            return False
        
        # Try the RPC function
        response = supabase.rpc("match_chunks", {
            "query_embedding": test_embedding,
            "match_count": 3,
            "similarity_threshold": 0.1
        }).execute()
        
        if response.data is not None:
            print(f"✅ RPC function 'match_chunks' works! Returned {len(response.data)} results")
            
            # Show sample results
            for i, result in enumerate(response.data[:2]):
                similarity = result.get('similarity', 'N/A')
                filename = result.get('filename', 'Unknown')
                print(f"   Result {i+1}: {filename} (similarity: {similarity})")
            return True
        else:
            print("⚠️ RPC function exists but returned no data")
            return False
            
    except Exception as e:
        error_str = str(e)
        if "function match_chunks" in error_str and "does not exist" in error_str:
            print("❌ RPC function 'match_chunks' does not exist in database!")
            print("💡 You need to create this function in Supabase. Here's the SQL:")
            print("""
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding vector(384),
  match_count int DEFAULT 10,
  similarity_threshold float DEFAULT 0.3
)
RETURNS TABLE (
  file_id uuid,
  filename text,
  chunk_index int,
  chunk_text text,
  similarity float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    rf.drive_file_id::uuid as file_id,
    rf.filename,
    pc.chunk_index,
    pc.chunk_text,
    1 - (pc.embedding <=> query_embedding) AS similarity
  FROM processed_chunks pc
  JOIN raw_files rf ON pc.file_id = rf.id
  WHERE pc.embedding IS NOT NULL
    AND 1 - (pc.embedding <=> query_embedding) > similarity_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
$$;
            """)
            return False
        else:
            print(f"❌ RPC function test failed: {e}")
            return False

def test_semantic_search_full():
    """Test full semantic search functionality"""
    print("\n🧪 TESTING FULL SEMANTIC SEARCH")
    print("=" * 50)
    
    test_queries = [
        "artificial intelligence",
        "machine learning", 
        "data science",
        "technology trends",
        "business strategy"
    ]
    
    success_count = 0
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        try:
            results = semantic_search(query, limit=3, similarity_threshold=0.2)
            
            if results:
                print(f"✅ Found {len(results)} results")
                for i, result in enumerate(results[:2], 1):
                    filename = result.get('filename', 'Unknown')
                    similarity = result.get('similarity', 0)
                    print(f"   {i}. {filename} (similarity: {similarity:.3f})")
                success_count += 1
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 SUCCESS RATE: {success_count}/{len(test_queries)} queries worked")
    return success_count > 0

def main():
    """Run all tests"""
    print("🚀 SEMANTIC SEARCH DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Test 1: Basic connection
    print("\n🧪 TESTING SUPABASE CONNECTION")
    print("=" * 50)
    if not test_connection():
        print("❌ Supabase connection failed!")
        return 1
    print("✅ Supabase connection successful")
    
    # Test 2: Embedding generation
    if not test_embedding_generation():
        print("\n❌ EMBEDDING GENERATION FAILED - Cannot proceed with semantic search")
        return 1
    
    # Test 3: Database embeddings
    if not test_database_embeddings():
        print("\n❌ NO EMBEDDINGS IN DATABASE - You need to run sync to generate embeddings")
        return 1
    
    # Test 4: RPC function
    rpc_works = test_rpc_function()
    
    # Test 5: Full semantic search
    if not test_semantic_search_full():
        print("\n❌ SEMANTIC SEARCH NOT WORKING PROPERLY")
        if not rpc_works:
            print("💡 This is likely because the RPC function is missing - see the SQL above")
        return 1
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Semantic search is working correctly")
    return 0

if __name__ == "__main__":
    exit(main())