from dotenv import load_dotenv
import os
from supabase import create_client, Client
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import sys
from supabase import create_client
import json

# Fix Windows encoding issues
if sys.platform.startswith('win'):
    import locale
    try:
        # Try to set UTF-8 encoding for Windows
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        pass
    
    # Set stdout encoding to UTF-8 if possible
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    # Load model once globally
    model = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDINGS_AVAILABLE = True
    print("[OK] SentenceTransformers loaded successfully")
    test_embedding = model.encode("test")
    print(f"[INFO] Model embedding dimension: {test_embedding.shape[0]}")
    
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("[WARN] SentenceTransformers not available. Install with: pip install sentence-transformers")

SYNC_USER_UUID = "00000000-0000-0000-0000-000000000001" 

load_dotenv()  # This loads variables from .env into os.environ

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Check if embeddings exist in the table
response = supabase.table("processed_chunks").select("embedding").limit(5).execute()

# Print the result
print("[DEBUG] Sample Embeddings:", response.data)

# Optional: Print the length of first embedding to check dimension
if response.data and response.data[0]["embedding"]:
    print("[DEBUG] Embedding Vector Length:", len(response.data[0]["embedding"]))


def test_connection():
    """Test the Supabase connection and list tables"""
    try:
        # Try to query the raw_files table structure
        response = supabase.table("raw_files").select("*").limit(1).execute()
        print("[OK] Connection successful. raw_files table exists.")
        
        # Check processed_chunks table
        response2 = supabase.table("processed_chunks").select("*").limit(1).execute()
        print("[OK] processed_chunks table exists.")
        
        return True
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        return False

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    if not EMBEDDINGS_AVAILABLE:
        print("[WARN] Embeddings not available - returning empty embeddings")
        return [[] for _ in texts]
    
    try:
        print(f"[INFO] Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts).tolist()
        print(f"[OK] Generated embeddings with shape: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}")
        return embeddings
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        return [[] for _ in texts]

def insert_raw_file(file_id: str, file_name: str, file_content: str, file_hash: str = None, mime_type: str = None, file_size: str = None):
    """Insert a raw file into the Supabase raw_files table."""
    
    data = {
        "filename": file_name,
        "full_text": file_content,
        "uploaded_by": SYNC_USER_UUID,
        "drive_file_id": file_id,  # Store the original Google Drive file ID
        "file_hash": file_hash,
        "mime_type": mime_type,
        "file_size": file_size,
        # Note: uploaded_at will be auto-set by default
    }

    try:
        print(f"[INFO] Attempting to insert raw file: {file_name} ({len(file_content)} chars)")
        
        # First, check if a file with this drive_file_id already exists
        existing = supabase.table("raw_files").select("id, filename, drive_file_id").eq("drive_file_id", file_id).execute()
        
        if existing.data:
            # Update existing record
            record_id = existing.data[0]['id']
            print(f"[INFO] Updating existing record with ID: {record_id}")
            
            response = supabase.table("raw_files").update(data).eq("id", record_id).execute()
            print(f"[OK] Raw file updated: {file_name}")
        else:
            # Insert new record
            print(f"[INFO] Inserting new record for: {file_name}")
            response = supabase.table("raw_files").insert(data).execute()
            print(f"[OK] Raw file inserted: {file_name}")
        
        if response.data:
            print(f"[INFO] Response data length: {len(response.data)}")
            # Store the Supabase ID for use with chunks
            supabase_file_id = response.data[0]['id']
            print(f"[INFO] Supabase file ID: {supabase_file_id}")
            return supabase_file_id  # Return the Supabase UUID
        else:
            print(f"[WARN] No data returned from insert/update")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to insert raw file {file_name}: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        
        # Try to get more details about the error
        if hasattr(e, 'details'):
            print(f"[ERROR] Error details: {e.details}")
        if hasattr(e, 'message'):
            print(f"[ERROR] Error message: {e.message}")
            
        return None

def insert_processed_chunks(file_id: str, file_name: str, chunks: List[str]):
    """Insert processed chunks into Supabase processed_chunks table, including embeddings."""
    
    if not chunks:
        print(f"[WARN] No chunks to insert for file: {file_name}")
        return False
    
    # First, get the Supabase file ID by drive_file_id
    try:
        file_response = supabase.table("raw_files").select("id").eq("drive_file_id", file_id).execute()
        if not file_response.data:
            print(f"[ERROR] Could not find raw_files record for drive_file_id: {file_id}")
            return False
        
        supabase_file_id = file_response.data[0]['id']
        print(f"[INFO] Using Supabase file ID: {supabase_file_id}")
        
    except Exception as e:
        print(f"[ERROR] Error getting file ID for {file_name}: {e}")
        return False
    
    # Generate embeddings for all chunks
    print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
    embeddings = generate_embeddings(chunks)
    
    entries = [
        {
            "file_id": supabase_file_id,  # UUID reference to raw_files
            "chunk_index": idx,
            "chunk_text": chunk,
            "embedding": embedding if embedding else None,  # This should be a list[float] or None
            # created_at will be auto-set
        }
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    try:
        print(f"[INFO] Attempting to insert {len(entries)} chunks with embeddings for file: {file_name}")
        
        # First delete existing chunks for this file
        delete_response = supabase.table("processed_chunks").delete().eq("file_id", supabase_file_id).execute()
        deleted_count = len(delete_response.data) if delete_response.data else 0
        print(f"[INFO] Deleted {deleted_count} existing chunks")
        
        # Insert new chunks in batches to avoid payload limits
        batch_size = 50  # Reduced batch size due to embeddings
        total_inserted = 0
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            print(f"[INFO] Inserting batch {i//batch_size + 1}: {len(batch)} chunks with embeddings")
            
            batch_response = supabase.table("processed_chunks").insert(batch).execute()
            
            if batch_response.data:
                total_inserted += len(batch_response.data)
                print(f"[OK] Batch inserted: {len(batch_response.data)} chunks with embeddings")
            else:
                print(f"[WARN] Batch insert returned no data")
        
        print(f"[OK] Total chunks with embeddings inserted for file {file_name}: {total_inserted}")
        
        # Verify the insert
        verify = supabase.table("processed_chunks").select("file_id, chunk_index, embedding").eq("file_id", supabase_file_id).execute()
        print(f"[OK] Verification: {len(verify.data) if verify.data else 0} chunks found in database")
        
        # Check if embeddings were properly stored
        if verify.data:
            embeddings_count = sum(1 for chunk in verify.data if chunk.get('embedding'))
            print(f"[OK] Embeddings stored: {embeddings_count}/{len(verify.data)} chunks have embeddings")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to insert processed chunks for {file_name}: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        
        # Try to get more details about the error
        if hasattr(e, 'details'):
            print(f"[ERROR] Error details: {e.details}")
        if hasattr(e, 'message'):
            print(f"[ERROR] Error message: {e.message}")
            
        return False

def get_existing_file_hashes() -> Dict[str, str]:
    """Get existing file hashes from Supabase, indexed by drive_file_id"""
    try:
        print("[INFO] Fetching existing file hashes from Supabase...")
        response = supabase.table("raw_files").select("drive_file_id, file_hash").execute()
        
        if not response.data:
            print("[INFO] No existing files found in database")
            return {}
        
        # Create a mapping of drive_file_id -> file_hash
        file_hashes = {}
        for record in response.data:
            drive_file_id = record.get('drive_file_id')
            file_hash = record.get('file_hash')
            if drive_file_id and file_hash:
                file_hashes[drive_file_id] = file_hash
        
        print(f"[INFO] Found {len(file_hashes)} files with hashes in database")
        return file_hashes
        
    except Exception as e:
        print(f"[ERROR] Failed to get existing file hashes: {e}")
        return {}

def delete_file_by_drive_id(drive_file_id: str) -> bool:
    """Delete a file and its chunks by Google Drive file ID"""
    try:
        print(f"[INFO] Deleting file with drive_file_id: {drive_file_id}")
        
        # First get the Supabase file ID
        file_response = supabase.table("raw_files").select("id").eq("drive_file_id", drive_file_id).execute()
        
        if not file_response.data:
            print(f"[WARN] No file found with drive_file_id: {drive_file_id}")
            return True  # Not an error if file doesn't exist
        
        supabase_file_id = file_response.data[0]['id']
        
        # Delete chunks first (foreign key constraint)
        chunks_response = supabase.table("processed_chunks").delete().eq("file_id", supabase_file_id).execute()
        chunks_deleted = len(chunks_response.data) if chunks_response.data else 0
        print(f"[INFO] Deleted {chunks_deleted} chunks")
        
        # Delete the raw file
        file_delete_response = supabase.table("raw_files").delete().eq("id", supabase_file_id).execute()
        files_deleted = len(file_delete_response.data) if file_delete_response.data else 0
        print(f"[INFO] Deleted {files_deleted} raw file")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to delete file {drive_file_id}: {e}")
        return False
    
from sentence_transformers import SentenceTransformer

# Load your model (load once and reuse)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_query(query: str) -> list:
    """Embed the input query string into a vector"""
    try:
        embedding = embedding_model.encode(query)
        return embedding.tolist()  # Supabase RPC needs a list, not NumPy array
    except Exception as e:
        print(f"[ERROR] Failed to embed query: {e}")
        return []

# Add this function to your supabase_utils.py to fix the embedding conversion error

import numpy as np
import json
from typing import List, Dict, Any, Optional

def safe_cosine_similarity(vec1, vec2) -> float:
    """
    Safe cosine similarity calculation with proper type handling
    """
    try:
        # Handle different input types
        if isinstance(vec1, str):
            try:
                vec1 = json.loads(vec1)
            except:
                return 0.0
        
        if isinstance(vec2, str):
            try:
                vec2 = json.loads(vec2)
            except:
                return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)
        
        # Check for valid dimensions
        if vec1.shape != vec2.shape:
            return 0.0
        
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure the result is a valid float
        if np.isnan(similarity) or np.isinf(similarity):
            return 0.0
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error in safe_cosine_similarity: {e}")
        return 0.0

def enhanced_semantic_search(query: str, limit: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Enhanced semantic search with better error handling
    """
    try:
        logger.info(f"[INFO] Performing SEMANTIC search for: '{query}' (limit: {limit}, threshold: {similarity_threshold})")
        
        # Check if embeddings are available
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("[WARN] Embeddings not available")
            return []
        
        # Generate query embedding
        logger.info(f"[INFO] Embedding query: '{query[:50]}...' (length: {len(query)} chars)")
        
        try:
            query_embedding = embedding_model.encode([query])
            logger.info(f"[SUCCESS] Query embedded - vector dimension: {len(query_embedding[0])}")
            query_vector = query_embedding[0].tolist()
        except Exception as e:
            logger.error(f"[ERROR] Failed to embed query: {e}")
            return []
        
        # Try RPC function first (if available)
        logger.info("[INFO] Attempting RPC function match_chunks...")
        try:
            rpc_result = supabase.rpc('match_chunks', {
                'query_embedding': query_vector,
                'match_threshold': similarity_threshold,
                'match_count': limit
            }).execute()
            
            if rpc_result.data and len(rpc_result.data) > 0:
                logger.info(f"[SUCCESS] RPC match_chunks returned {len(rpc_result.data)} results")
                
                # Format RPC results
                formatted_results = []
                for item in rpc_result.data:
                    formatted_results.append({
                        'file_id': item.get('file_id'),
                        'filename': item.get('filename', 'Unknown'),
                        'chunk_text': item.get('chunk_text', ''),
                        'chunk_index': item.get('chunk_index'),
                        'similarity': item.get('similarity', 0.0)
                    })
                
                return formatted_results
            else:
                logger.info("[INFO] RPC match_chunks returned no data")
        except Exception as e:
            logger.warning(f"[WARN] RPC function failed: {e}")
        
        # Fallback to manual similarity calculation
        logger.info("[INFO] Using enhanced fallback similarity search...")
        
        # Get chunks with embeddings in batches
        logger.info("[INFO] Fetching chunks with embeddings from database...")
        
        # First, count how many chunks have embeddings
        count_response = supabase.table("processed_chunks").select("id").eq("embedding", "not.is.null").execute()
        total_chunks = len(count_response.data) if count_response.data else 0
        logger.info(f"[INFO] Found {total_chunks} chunks with embeddings in database")
        
        if total_chunks == 0:
            logger.warning("[WARN] No chunks with embeddings found")
            return []
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        all_similarities = []
        
        for offset in range(0, total_chunks, batch_size):
            batch_num = (offset // batch_size) + 1
            logger.info(f"[INFO] Processing batch {batch_num}: fetching {batch_size} chunks...")
            
            try:
                response = supabase.table("processed_chunks").select(
                    "file_id, chunk_index, chunk_text, embedding"
                ).not_.is_("embedding", "null").range(offset, offset + batch_size - 1).execute()
                
                if not response.data:
                    logger.warning(f"[WARN] Batch {batch_num} returned no data")
                    continue
                
                logger.info(f"[INFO] Processing {len(response.data)} chunks for similarity...")
                
                for chunk in response.data:
                    try:
                        # Get stored embedding
                        stored_embedding = chunk.get('embedding')
                        if not stored_embedding:
                            continue
                        
                        # Calculate similarity using safe function
                        similarity = safe_cosine_similarity(query_vector, stored_embedding)
                        
                        if similarity >= similarity_threshold:
                            all_similarities.append({
                                'file_id': chunk['file_id'],
                                'chunk_index': chunk['chunk_index'],
                                'chunk_text': chunk['chunk_text'],
                                'similarity': similarity
                            })
                    
                    except Exception as chunk_e:
                        logger.error(f"[ERROR] Error processing chunk: {chunk_e}")
                        continue
                        
            except Exception as batch_e:
                logger.error(f"[ERROR] Error processing batch {batch_num}: {batch_e}")
                continue
        
        # Sort by similarity and limit results
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = all_similarities[:limit]
        
        logger.info(f"[INFO] Found {len(top_results)} chunks above similarity threshold")
        
        if not top_results:
            logger.warning(f"[WARN] No chunks found above similarity threshold {similarity_threshold}")
            return []
        
        # Get file information for results
        final_results = []
        file_ids = list(set([r['file_id'] for r in top_results]))
        
        # Get file information in batch
        if file_ids:
            files_response = supabase.table("raw_files").select("id, filename").in_("id", file_ids).execute()
            file_info = {f['id']: f['filename'] for f in files_response.data} if files_response.data else {}
            
            for result in top_results:
                filename = file_info.get(result['file_id'], 'Unknown File')
                final_results.append({
                    'file_id': result['file_id'],
                    'filename': filename,
                    'chunk_text': result['chunk_text'],
                    'chunk_index': result['chunk_index'],
                    'similarity': result['similarity'],
                    'source': 'semantic'
                })
        
        logger.info(f"[SUCCESS] Semantic search completed: {len(final_results)} results")
        return final_results
        
    except Exception as e:
        logger.error(f"[ERROR] Semantic search failed: {e}")
        return []

# Replace your existing semantic_search function with this enhanced version
def semantic_search(query: str, limit: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Main semantic search function with enhanced error handling
    """
    return enhanced_semantic_search(query, limit, similarity_threshold)


print(f"[DEBUG] EMBEDDINGS_AVAILABLE = {EMBEDDINGS_AVAILABLE}")
def semantic_search(query: str, limit: int = 10, similarity_threshold: float = 0.3) -> List[Dict]:
    """Perform semantic search using Supabase RPC with enhanced error handling"""
    if not EMBEDDINGS_AVAILABLE:
        print("[WARN] Semantic search not available - embeddings not loaded")
        return []

    try:
        print(f"[INFO] Performing SEMANTIC search for: '{query}' (limit: {limit}, threshold: {similarity_threshold})")

        # Step 1: Embed the query
        query_embedding = embed_query(query)
        if not query_embedding:
            print("[ERROR] Failed to generate embedding for query")
            return []

        print(f"[INFO] Query embedded successfully - vector length: {len(query_embedding)}")

        # Step 2: Try RPC function first
        try:
            print("[INFO] Attempting RPC function match_chunks...")
            response = supabase.rpc("match_chunks", {
                "query_embedding": query_embedding, 
                "match_count": limit,
                "similarity_threshold": similarity_threshold
            }).execute()

            if response.data and len(response.data) > 0:
                print(f"[SUCCESS] RPC match_chunks returned {len(response.data)} results")
                
                # Filter results by similarity threshold
                filtered_results = []
                for result in response.data:
                    similarity = result.get('similarity', 0)
                    if similarity >= similarity_threshold:
                        filtered_results.append(result)
                
                print(f"[INFO] {len(filtered_results)} results above similarity threshold {similarity_threshold}")
                return filtered_results
            else:
                print("[WARN] RPC match_chunks returned no data")
        except Exception as rpc_error:
            print(f"[WARN] RPC function failed: {rpc_error}")
            print("[INFO] Falling back to manual similarity search...")

        # Step 3: Fallback to manual similarity calculation
        return enhanced_fallback_similarity_search(query_embedding, limit=limit, similarity_threshold=similarity_threshold)

    except Exception as e:
        print(f"[ERROR] Semantic search completely failed: {e}")
        return []


def enhanced_fallback_similarity_search(query_embedding: List[float], limit: int = 10, similarity_threshold: float = 0.3) -> List[Dict]:
    """Enhanced fallback similarity search with better performance"""
    try:
        print("[INFO] Using enhanced fallback similarity search...")
        
        # Get chunks with embeddings in batches for better performance
        print("[INFO] Fetching chunks with embeddings from database...")
        
        # First, get total count of chunks with embeddings
        count_response = supabase.table("processed_chunks").select("id", count="exact").not_.is_("embedding", "null").execute()
        total_chunks = count_response.count if hasattr(count_response, 'count') else len(count_response.data) if count_response.data else 0
        
        print(f"[INFO] Found {total_chunks} chunks with embeddings in database")
        
        if total_chunks == 0:
            print("[WARN] No chunks with embeddings found in database")
            return []
        
        # Fetch chunks in batches
        batch_size = 1000
        all_similarities = []
        
        for offset in range(0, min(total_chunks, 5000), batch_size):  # Limit to 5000 for performance
            print(f"[INFO] Processing batch {offset//batch_size + 1}: fetching {batch_size} chunks...")
            
            batch_response = supabase.table("processed_chunks").select(
                "file_id, chunk_index, chunk_text, embedding"
            ).not_.is_("embedding", "null").range(offset, offset + batch_size - 1).execute()
            
            if not batch_response.data:
                break
            
            print(f"[INFO] Processing {len(batch_response.data)} chunks for similarity...")
            
            # Calculate similarities for this batch
            for chunk in batch_response.data:
                if chunk.get('embedding'):
                    try:
                        chunk_embedding = chunk['embedding']
                        similarity = cosine_similarity(query_embedding, chunk_embedding)
                        
                        # Only keep results above threshold
                        if similarity >= similarity_threshold:
                            all_similarities.append({
                                'chunk': chunk,
                                'similarity': similarity
                            })
                    except Exception as e:
                        print(f"[WARN] Error calculating similarity for chunk {chunk.get('chunk_index', '?')}: {e}")
                        continue
        
        print(f"[INFO] Found {len(all_similarities)} chunks above similarity threshold")
        
        if not all_similarities:
            print(f"[WARN] No chunks found above similarity threshold {similarity_threshold}")
            return []
        
        # Sort by similarity and take top results
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_similarities = all_similarities[:limit]
        
        print(f"[INFO] Taking top {len(top_similarities)} results")
        
        # Get file information for results
        if top_similarities:
            file_ids = list(set([item['chunk']['file_id'] for item in top_similarities]))
            print(f"[INFO] Fetching file info for {len(file_ids)} unique files...")
            
            files_info = supabase.table("raw_files").select("id, filename, drive_file_id").in_("id", file_ids).execute()
            file_lookup = {file['id']: file for file in files_info.data} if files_info.data else {}
            
            results = []
            for item in top_similarities:
                chunk = item['chunk']
                similarity = item['similarity']
                file_info = file_lookup.get(chunk['file_id'])
                
                if file_info:
                    results.append({
                        'file_id': file_info['drive_file_id'],
                        'filename': file_info['filename'],
                        'chunk_index': chunk['chunk_index'],
                        'chunk_text': chunk['chunk_text'],
                        'similarity': similarity,
                        'source': 'semantic_fallback'
                    })
            
            print(f"[SUCCESS] Enhanced fallback search completed - {len(results)} results")
            return results
        
        return []
        
    except Exception as e:
        print(f"[ERROR] Enhanced fallback similarity search failed: {e}")
        return []


def embed_query(query: str) -> List[float]:
    """Embed the input query string into a vector with enhanced error handling"""
    if not EMBEDDINGS_AVAILABLE:
        print("[ERROR] Cannot embed query - SentenceTransformers not available")
        return []
    
    try:
        print(f"[INFO] Embedding query: '{query[:50]}...' (length: {len(query)} chars)")
        
        # Use the global model
        embedding = embedding_model.encode(query)
        embedding_list = embedding.tolist()
        
        print(f"[SUCCESS] Query embedded - vector dimension: {len(embedding_list)}")
        return embedding_list
        
    except Exception as e:
        print(f"[ERROR] Failed to embed query: {e}")
        return []


def test_semantic_search_functionality():
    """Test semantic search with actual data"""
    try:
        print("\n[TEST] Testing semantic search functionality...")
        
        # Check if embeddings exist
        embedding_check = supabase.table("processed_chunks").select("embedding").not_.is_("embedding", "null").limit(1).execute()
        
        if not embedding_check.data:
            print("[ERROR] No embeddings found in database!")
            return False
        
        print(f"[OK] Embeddings found in database")
        
        # Test queries
        test_queries = [
            "artificial intelligence",
            "machine learning",
            "data science",
            "technology",
            "business"
        ]
        
        for query in test_queries:
            print(f"\n[TEST] Testing query: '{query}'")
            results = semantic_search(query, limit=3, similarity_threshold=0.1)  # Lower threshold for testing
            
            if results:
                print(f"[SUCCESS] Found {len(results)} semantic results")
                for i, result in enumerate(results[:2], 1):
                    print(f"  {i}. {result.get('filename', 'Unknown')} (similarity: {result.get('similarity', 0):.3f})")
                return True  # Found working results
            else:
                print(f"[WARN] No results for '{query}'")
        
        print("[ERROR] No semantic search results found for any test query")
        return False
        
    except Exception as e:
        print(f"[ERROR] Semantic search test failed: {e}")
        return False
    
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    try:
        import math
        
        # Calculate dot product
        dot_product = sum(x * y for x, y in zip(a, b))
        
        # Calculate magnitudes
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        # Avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        return dot_product / (magnitude_a * magnitude_b)
    except Exception as e:
        print(f"[ERROR] Error calculating cosine similarity: {e}")
        return 0

def get_table_counts():
    """Get row counts for both tables"""
    try:
        raw_files_count = supabase.table("raw_files").select("id", count="exact").execute()
        chunks_count = supabase.table("processed_chunks").select("id", count="exact").execute()
        
        print(f"[INFO] Table counts:")
        print(f"   raw_files: {raw_files_count.count if hasattr(raw_files_count, 'count') else len(raw_files_count.data)}")
        print(f"   processed_chunks: {chunks_count.count if hasattr(chunks_count, 'count') else len(chunks_count.data)}")
        
        # Also check embedding stats
        if chunks_count.data or hasattr(chunks_count, 'count'):
            try:
                embeddings_count = supabase.table("processed_chunks").select("id", count="exact").not_.is_("embedding", "null").execute()
                embedding_count = embeddings_count.count if hasattr(embeddings_count, 'count') else len(embeddings_count.data)
                print(f"   chunks with embeddings: {embedding_count}")
            except:
                print(f"   chunks with embeddings: Unable to count")
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to get table counts: {e}")
        return False

def list_recent_files(limit=5):
    """List recent files in the database"""
    try:
        print(f"[INFO] Recent files in raw_files:")
        recent_raw = supabase.table("raw_files").select("id, filename, drive_file_id, uploaded_at").order("uploaded_at", desc=True).limit(limit).execute()
        
        if recent_raw.data:
            for file in recent_raw.data:
                drive_id = file.get('drive_file_id', 'Unknown')
                print(f"   - {file.get('filename', 'Unknown')} (Drive ID: {drive_id[:10]}...)")
        else:
            print("   No files found")
            
        print(f"[INFO] Recent chunks in processed_chunks:")
        recent_chunks = supabase.table("processed_chunks").select("file_id, chunk_index, created_at, embedding").order("created_at", desc=True).limit(limit).execute()
        
        if recent_chunks.data:
            for chunk in recent_chunks.data:
                has_embedding = "[OK]" if chunk.get('embedding') else "[NONE]"
                print(f"   - File {str(chunk.get('file_id', 'Unknown'))[:8]}... chunk {chunk.get('chunk_index', '?')} {has_embedding}")
        else:
            print("   No chunks found")
            
        return True
    except Exception as e:
        print(f"[ERROR] Failed to list recent files: {e}")
        return False

def search_files(query: str, limit: int = 10) -> List[Dict]:
    """Search files by text content using ILIKE for pattern matching"""
    try:
        print(f"[INFO] Searching for: '{query}' (limit: {limit})")
        
        # Convert query to lowercase and add wildcards for ILIKE search
        search_pattern = f"%{query.lower()}%"
        
        results = []
        
        # Search in raw files using ILIKE (case-insensitive pattern matching)
        try:
            raw_search = supabase.table("raw_files").select("id, filename, drive_file_id, full_text").ilike("full_text", search_pattern).limit(limit).execute()
            
            if raw_search.data:
                for file in raw_search.data:
                    # Extract a snippet around the match
                    full_text = file.get('full_text', '')
                    snippet = full_text[:200] + '...' if len(full_text) > 200 else full_text
                    
                    results.append({
                        'file_id': file['drive_file_id'],
                        'filename': file['filename'],
                        'chunk_text': snippet,
                        'source': 'raw_file'
                    })
                    
            print(f"[INFO] Found {len(raw_search.data) if raw_search.data else 0} raw file matches")
        except Exception as e:
            print(f"[WARN] Raw file search failed: {e}")
        
        # Search in processed chunks using ILIKE
        try:
            chunk_search = supabase.table("processed_chunks").select("file_id, chunk_index, chunk_text").ilike("chunk_text", search_pattern).limit(limit).execute()
            
            if chunk_search.data:
                # Get file info for chunks
                file_ids = list(set([chunk['file_id'] for chunk in chunk_search.data]))
                files_info = supabase.table("raw_files").select("id, filename, drive_file_id").in_("id", file_ids).execute()
                
                file_lookup = {file['id']: file for file in files_info.data} if files_info.data else {}
                
                for chunk in chunk_search.data:
                    file_info = file_lookup.get(chunk['file_id'])
                    if file_info:
                        results.append({
                            'file_id': file_info['drive_file_id'],
                            'filename': file_info['filename'],
                            'chunk_index': chunk['chunk_index'],
                            'chunk_text': chunk['chunk_text'][:300] + '...' if len(chunk['chunk_text']) > 300 else chunk['chunk_text'],
                            'source': 'chunk'
                        })
                        
            print(f"[INFO] Found {len(chunk_search.data) if chunk_search.data else 0} chunk matches")
        except Exception as e:
            print(f"[WARN] Chunk search failed: {e}")
        
        print(f"[INFO] Total search results: {len(results)}")
        return results
        
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return []

def get_file_content(drive_file_id: str) -> Optional[Dict]:
    """Get full content of a file by drive_file_id"""
    try:
        response = supabase.table("raw_files").select("*").eq("drive_file_id", drive_file_id).execute()
        
        if response.data:
            return response.data[0]
        return None
        
    except Exception as e:
        print(f"[ERROR] Failed to get file content: {e}")
        return None

def get_file_chunks(drive_file_id: str) -> List[Dict]:
    """Get all chunks for a file by drive_file_id"""
    try:
        # First get the Supabase file ID
        file_response = supabase.table("raw_files").select("id").eq("drive_file_id", drive_file_id).execute()
        
        if not file_response.data:
            return []
        
        supabase_file_id = file_response.data[0]['id']
        
        # Get chunks
        chunks_response = supabase.table("processed_chunks").select("*").eq("file_id", supabase_file_id).order("chunk_index").execute()
        
        return chunks_response.data if chunks_response.data else []
        
    except Exception as e:
        print(f"[ERROR] Failed to get file chunks: {e}")
        return []

def cleanup_orphaned_chunks():
    """Remove chunks that don't have corresponding raw files"""
    try:
        print("[INFO] Cleaning up orphaned chunks...")
        
        # Get all chunk file_ids
        chunks_response = supabase.table("processed_chunks").select("file_id").execute()
        if not chunks_response.data:
            print("No chunks found")
            return True
        
        chunk_file_ids = list(set([chunk['file_id'] for chunk in chunks_response.data]))
        
        # Get all raw file ids
        files_response = supabase.table("raw_files").select("id").execute()
        if not files_response.data:
            print("No raw files found")
            return True
            
        raw_file_ids = set([file['id'] for file in files_response.data])
        
        # Find orphaned chunks
        orphaned_file_ids = [fid for fid in chunk_file_ids if fid not in raw_file_ids]
        
        if not orphaned_file_ids:
            print("[OK] No orphaned chunks found")
            return True
        
        print(f"[INFO] Found {len(orphaned_file_ids)} orphaned file references, deleting chunks...")
        
        deleted_count = 0
        for file_id in orphaned_file_ids:
            delete_response = supabase.table("processed_chunks").delete().eq("file_id", file_id).execute()
            if delete_response.data:
                deleted_count += len(delete_response.data)
        
        print(f"[OK] Deleted {deleted_count} orphaned chunks")
        return True
        
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        return False

def test_search_functionality():
    """Test search functionality with actual data"""
    try:
        print("[INFO] Testing search functionality...")
        
        # First, let's see what data we have
        print("\n[INFO] Checking available data...")
        files_response = supabase.table("raw_files").select("filename, drive_file_id").limit(5).execute()
        
        if files_response.data:
            print(f"[INFO] Sample files in database:")
            for file in files_response.data:
                print(f"   - {file.get('filename', 'Unknown')}")
        else:
            print("[WARN] No files found in database")
            return False
        
        # Check chunks
        chunks_response = supabase.table("processed_chunks").select("chunk_text").limit(3).execute()
        
        if chunks_response.data:
            print(f"\n[INFO] Sample chunk content:")
            for i, chunk in enumerate(chunks_response.data):
                text = chunk.get('chunk_text', '')
                preview = text[:100] + '...' if len(text) > 100 else text
                print(f"   Chunk {i+1}: {preview}")
        else:
            print("[WARN] No chunks found in database")
            return False
        
        # Test basic text search
        print("\n[INFO] Testing text search with common words...")
        test_queries = ["the", "and", "is", "of", "to"]
        
        for query in test_queries:
            results = search_files(query, limit=2)
            if results:
                print(f"[OK] Found {len(results)} results for '{query}'")
                break
        else:
            print("[WARN] No results found for any common words")
        
        # Test semantic search if available
        if EMBEDDINGS_AVAILABLE:
            print("\n[INFO] Testing semantic search...")
            semantic_results = semantic_search("test", limit=2)
            if semantic_results:
                print(f"[OK] Semantic search working - found {len(semantic_results)} results")
            else:
                print("[WARN] Semantic search returned no results")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test search functionality failed: {e}")
        return False
    print("[INFO] Testing Supabase connection and operations...")
    
    if test_connection():
        get_table_counts()
        list_recent_files()
        
        # Test insert with dummy data
        print("\n[INFO] Testing insert operations with dummy data...")
        supabase_file_id = insert_raw_file(
            "test_file_123",  # This is your original file_id 
            "test_file.txt", 
            "This is test content for debugging embeddings functionality",
            "dummy_hash_123",
            "text/plain",
            "42"
        )
        
        if supabase_file_id:
            test_chunks = [
                "This is the first test chunk about machine learning", 
                "This is the second test chunk about artificial intelligence", 
                "This is the third test chunk about natural language processing"
            ]
            insert_processed_chunks("test_file_123", "test_file.txt", test_chunks)
            
        print("\n[INFO] Final counts after test:")
        get_table_counts()
        
        # Test search
        print("\n[INFO] Testing text search...")
        results = search_files("test")
        for result in results:
            print(f"  Found: {result}")
            
        # Test semantic search if available
        if EMBEDDINGS_AVAILABLE:
            print("\n[INFO] Testing semantic search...")
            semantic_results = semantic_search("artificial intelligence")
            for result in semantic_results:
                print(f"  Semantic: {result}")
        
        # Test cleanup
        print("\n[INFO] Testing cleanup...")
        cleanup_orphaned_chunks()
        
    else:
        print("[ERROR] Connection test failed - check your Supabase credentials and table setup")