from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
import subprocess
import sys
import threading
import time
from threading import Lock
import json
import re
from typing import List, Dict
from supabase import create_client

# Load environment variables first
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["*"])  # Configure for your frontend's actual domain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load from .env - Supabase configuration
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
google_drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

# Import Supabase utilities
try:
    from supabase_utils import (
        test_connection, 
        get_table_counts, 
        search_files, 
        semantic_search,
        get_file_content,
        get_file_chunks,
        supabase
    )
    SUPABASE_AVAILABLE = True
    logger.info("✅ Supabase utilities loaded successfully")
except ImportError as e:
    SUPABASE_AVAILABLE = False
    logger.error(f"❌ Failed to import Supabase utilities: {e}")

# Import Google Gemini for chat functionality
try:
    import google.generativeai as genai
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        GEMINI_AVAILABLE = True
        logger.info("✅ Google Gemini configured successfully")
    else:
        GEMINI_AVAILABLE = False
        logger.warning("⚠️ GEMINI_API_KEY not found in environment")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️ Google Gemini not available. Install with: pip install google-generativeai")

# Global variable to track sync status with thread safety
sync_status = {
    "running": False, 
    "message": "",
    "success": None,
    "last_run": None,
    "details": None,
    "error": None
}
sync_lock = Lock()  # Thread-safe access to sync_status

# ENHANCED QUERY CLASSIFICATION
def classify_query_intent(query: str) -> dict:
    """Enhanced query classification to handle different intents"""
    query_lower = query.lower().strip()
    
    # Define intent patterns
    greeting_patterns = [
        'hi', 'hii', 'hello', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'greetings', 'howdy', 'what\'s up'
    ]
    
    help_patterns = [
        'help', 'what can you do', 'capabilities', 'what do you know',
        'what information', 'how can you help', 'what\'s available',
        'what can you tell me', 'what do you have', 'what\'s in your database'
    ]
    
    status_patterns = [
        'status', 'how many documents', 'what files', 'database status',
        'system status', 'health check'
    ]
    
    # Check for simple greetings (short queries)
    if len(query.split()) <= 3:
        if any(pattern in query_lower for pattern in greeting_patterns):
            return {'type': 'greeting', 'confidence': 'high'}
    
    # Check for help requests
    if any(pattern in query_lower for pattern in help_patterns):
        return {'type': 'help', 'confidence': 'high'}
    
    # Check for status requests
    if any(pattern in query_lower for pattern in status_patterns):
        return {'type': 'status', 'confidence': 'high'}
    
    # Check for conversational elements in knowledge queries
    conversational_starters = ['can you', 'please tell me', 'i want to know', 'could you explain']
    has_conversational = any(starter in query_lower for starter in conversational_starters)
    
    # Default to knowledge query
    return {
        'type': 'knowledge', 
        'confidence': 'medium',
        'conversational': has_conversational
    }

def generate_greeting_response() -> str:
    """Generate a friendly greeting response"""
    greetings = [
        "Hello! I'm IntelliRAG, your AI assistant specialized in analyzing and answering questions from Google Drive documents.",
        "Hi there! I'm IntelliRAG, and I have access to your Google Drive knowledge base.",
        "Hey! I'm IntelliRAG, your friendly AI assistant for document analysis and Q&A."
    ]
    
    capabilities = [
        "I can help you find information from your uploaded documents, answer questions about their content, and provide summaries.",
        "I'm designed to search through your document collection and provide detailed answers based on the content I find.",
        "I can analyze your documents and provide comprehensive answers to your questions about their content."
    ]
    
    import random
    greeting = random.choice(greetings)
    capability = random.choice(capabilities)
    
    return f"{greeting}\n\n{capability}\n\nWhat would you like to know about your documents today?"

def generate_help_response() -> str:
    """Generate a helpful capabilities response"""
    return """🤖 **What I can help you with:**

**Document Search & Analysis:**
- Find specific information across all your Google Drive documents
- Answer questions about document content with citations
- Provide summaries of documents or specific topics
- Search by keywords, concepts, or natural language queries

**My Specialties:**
- Semantic search (understanding meaning, not just keywords)
- Cross-document analysis and comparison
- Extracting specific data points or facts
- Explaining complex information in simple terms

**How to use me effectively:**
- Ask specific questions about your documents
- Request summaries of topics or files
- Ask for comparisons between different documents
- Use natural language - I understand context!

**Examples of what you can ask:**
- "What does the marketing report say about Q3 performance?"
- "Summarize the main points from the project proposal"
- "Find information about budget allocations"
- "What are the key risks mentioned in the analysis?"

What specific information would you like me to help you find?"""

def generate_status_response() -> str:
    """Generate system status response"""
    try:
        # Get database stats
        raw_count = supabase.table("raw_files").select("id", count="exact").execute()
        chunks_count = supabase.table("processed_chunks").select("id", count="exact").execute()
        
        total_files = len(raw_count.data) if raw_count.data else 0
        total_chunks = len(chunks_count.data) if chunks_count.data else 0
        
        return f"""📊 **System Status:**

**Knowledge Base:**
- 📁 Total Documents: {total_files}
- 📄 Processed Chunks: {total_chunks}
- 🔍 Search Status: {'Active' if total_files > 0 else 'No documents found'}

**AI Capabilities:**
- 🧠 Semantic Search: {'Available' if total_chunks > 0 else 'Waiting for documents'}
- 💬 Conversational AI: Active
- 🔗 Google Drive Integration: Connected

{'Your knowledge base is ready for questions!' if total_files > 0 else 'Please sync your documents first to start asking questions.'}"""

    except Exception as e:
        return f"📊 **System Status:** Unable to retrieve current status. Error: {str(e)}"

def extract_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from a natural language query"""
    stopwords = {
        'what', 'is', 'the', 'are', 'how', 'does', 'do', 'can', 'will', 'would', 'could',
        'tell', 'me', 'about', 'explain', 'describe', 'of', 'in', 'on', 'at', 'to', 'for',
        'with', 'by', 'from', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'that', 'this',
        'they', 'them', 'their', 'there', 'where', 'when', 'why', 'which', 'who', 'whom'
    }
    
    # Clean and split the query
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    words = query_clean.split()
    
    # Filter out stopwords and short words
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    
    # If no keywords found, use the original query
    if not keywords:
        keywords = [query.strip()]
    
    return keywords

def safe_semantic_search(query: str, limit: int = 5) -> List[Dict]:
    """Safe wrapper for semantic search with error handling"""
    try:
        logger.info(f"🔍 Attempting semantic search for: '{query}'")
        results = semantic_search(query, limit=limit, similarity_threshold=0.25)
        
        if results and len(results) > 0:
            logger.info(f"✅ Semantic search found {len(results)} results")
            return results
        else:
            logger.info("ℹ️ Semantic search returned no results")
            return []
            
    except Exception as e:
        logger.error(f"❌ Semantic search error: {e}")
        return []

def enhanced_search_documents(query: str, limit: int = 10) -> List[Dict]:
    """Enhanced search that handles errors gracefully"""
    logger.info(f"Enhanced search for: '{query}' (limit: {limit})")
    
    all_results = []
    
    # PRIORITY 1: Try semantic search with error handling
    if SUPABASE_AVAILABLE:
        try:
            semantic_results = safe_semantic_search(query, limit=limit)
            
            if semantic_results:
                for result in semantic_results:
                    result['search_type'] = 'semantic'
                    result['priority'] = 1
                    all_results.append(result)
                logger.info(f"✅ Semantic search contributed {len(semantic_results)} results")
                
                # If we have good semantic results, prioritize them
                if len(semantic_results) >= 3:
                    return semantic_results[:limit]
        except Exception as e:
            logger.error(f"❌ Semantic search module error: {e}")
    
    # FALLBACK 1: Text search
    if len(all_results) < 3:
        try:
            logger.info("📝 Using text search as fallback")
            text_results = search_files(query, limit=max(3, limit//2))
            if text_results:
                for result in text_results:
                    result['search_type'] = 'text'
                    result['priority'] = 2
                    all_results.append(result)
                logger.info(f"📝 Text search found {len(text_results)} results")
        except Exception as e:
            logger.warning(f"Text search failed: {e}")
    
    # FALLBACK 2: Keyword search
    if len(all_results) < 2:
        logger.info("🔍 Using keyword search as last resort")
        keywords = extract_keywords(query)
        for keyword in keywords[:2]:
            try:
                keyword_results = search_files(keyword, limit=2)
                for result in keyword_results:
                    result['search_type'] = 'keyword'
                    result['priority'] = 3
                    all_results.append(result)
                logger.info(f"🔍 Keyword '{keyword}' found {len(keyword_results)} results")
            except Exception as e:
                logger.warning(f"Keyword search for '{keyword}' failed: {e}")
    
    # Remove duplicates and sort
    seen = set()
    unique_results = []
    for result in all_results:
        unique_id = f"{result.get('file_id', '')}-{result.get('chunk_index', '')}"
        if unique_id not in seen:
            seen.add(unique_id)
            unique_results.append(result)
    
    # Sort by priority (semantic first)
    def sort_key(result):
        priority = result.get('priority', 999)
        search_type = result.get('search_type', 'unknown')
        
        if search_type == 'semantic':
            return (1, -result.get('similarity', 0))
        elif search_type == 'text':
            return (2, 0)
        else:
            return (3, 0)
    
    unique_results.sort(key=sort_key)
    final_results = unique_results[:limit]
    
    # Log results
    semantic_count = len([r for r in final_results if r.get('search_type') == 'semantic'])
    text_count = len([r for r in final_results if r.get('search_type') == 'text'])
    keyword_count = len([r for r in final_results if r.get('search_type') == 'keyword'])
    
    logger.info(f"🎯 Final results: {len(final_results)} total ({semantic_count} semantic, {text_count} text, {keyword_count} keyword)")
    
    return final_results

def enhanced_generate_response_with_context(query: str, search_results: list, intent: dict) -> str:
    """Enhanced response generation with intent-aware handling"""
    if not GEMINI_AVAILABLE:
        return "⚠️ Gemini AI not available. Please check your API key configuration."
    
    try:
        logger.info(f"Generating response for {len(search_results)} search results with intent: {intent['type']}")
        
        # Handle different intents
        if intent['type'] == 'greeting':
            return generate_greeting_response()
        elif intent['type'] == 'help':
            return generate_help_response()
        elif intent['type'] == 'status':
            return generate_status_response()
        
        # For knowledge queries, check if we have results
        if not search_results:
            return """I couldn't find relevant information in my knowledge base to answer your specific question. 

However, I'm here to help! I can:
- Search through all your Google Drive documents
- Answer questions about document content
- Provide summaries and analysis
- Find specific information across multiple files

Could you try rephrasing your question, or ask me about what types of documents I have access to?"""
        
        # Prepare context for knowledge queries
        context_parts = []
        for i, result in enumerate(search_results[:5], 1):
            filename = result.get('filename', 'Unknown File')
            
            # Get the actual content
            content = ""
            if result.get('chunk_text'):
                content = result.get('chunk_text', '')
            elif result.get('source') == 'raw_file':
                try:
                    file_content = get_file_content(result.get('file_id'))
                    if file_content:
                        content = file_content.get('full_text', '')[:1000]
                except Exception as e:
                    logger.warning(f"Could not get file content: {e}")
                    content = result.get('content', '')[:1000]
            
            if content:
                context_parts.append(f"Source {i} - {filename}:\n{content}\n")
        
        if not context_parts:
            return "I found some potentially relevant documents, but couldn't extract readable content from them. This might be a database indexing issue."
        
        context = "\n".join(context_parts)
        logger.info(f"Context length: {len(context)} characters")
        
        # Enhanced prompt with personality
        prompt = f"""You are IntelliRAG, a friendly and knowledgeable AI assistant specializing in Google Drive document analysis. You have a warm, professional personality and are enthusiastic about helping users find information.

**Your personality:**
- Friendly, approachable, and conversational
- Confident about information in your knowledge base
- Honest when information isn't available
- Professional but not overly formal

**Context from knowledge base:**
{context}

**User Question:** {query}

**Instructions:**
- Provide a comprehensive answer based ONLY on the context above
- Be conversational and friendly in your tone
- If the context contains relevant information, use it confidently
- Cite sources naturally (e.g., "According to the marketing report..." or "As mentioned in the project document...")
- If the context doesn't fully address the question, mention what information IS available
- Stay focused on the knowledge base content
- Be enthusiastic about helping the user

**Response:**"""

        try:
            response = model.generate_content(prompt)
            if response and response.text:
                logger.info("Response generated successfully")
                return response.text
            else:
                return "I apologize, but I received an empty response from the AI model. Please try rephrasing your question."
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            if "quota" in str(e).lower():
                return "I apologize, but the AI service is currently at capacity. Please try again in a moment."
            elif "safety" in str(e).lower():
                return "I apologize, but I cannot provide a response to this query due to content policies."
            else:
                return f"I encountered a technical issue while generating the response. Please try again."
        
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
        return "I apologize, but I encountered an unexpected error. Please try again."

def test_database_populated():
    """Test if database actually contains data"""
    try:
        raw_count = supabase.table("raw_files").select("id", count="exact").execute()
        chunks_count = supabase.table("processed_chunks").select("id", count="exact").execute()
        
        total_files = len(raw_count.data) if raw_count.data else 0
        total_chunks = len(chunks_count.data) if chunks_count.data else 0
        
        logger.info(f"Database status: {total_files} files, {total_chunks} chunks")
        
        if total_files == 0 and total_chunks == 0:
            logger.error("DATABASE IS EMPTY! No files or chunks found.")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False

def update_sync_status(updates):
    """Thread-safe update of sync status"""
    with sync_lock:
        sync_status.update(updates)
        logger.info(f"Sync status updated: {updates}")

def run_sync_script():
    """Run the sync script in a separate thread"""
    update_sync_status({
        "running": True, 
        "message": "Synchronization started...",
        "success": None,
        "error": None,
        "details": None
    })
    
    logger.info("Starting synchronization process...")
    
    try:
        sync_script_path = "sync.py"
        
        sync_cmd = [
            sys.executable, sync_script_path,
            "--mode", "full"
        ]
        
        if google_drive_folder_id:
            sync_cmd.extend(["--folder-id", google_drive_folder_id])
        
        result = subprocess.run(
            sync_cmd, 
            capture_output=True, 
            text=True, 
            timeout=1800
        )
        
        if result.returncode == 0:
            update_sync_status({
                "running": False,
                "success": True,
                "message": "Synchronization completed successfully",
                "details": result.stdout,
                "last_run": time.time(),
                "error": None
            })
            logger.info("Synchronization completed successfully")
        else:
            update_sync_status({
                "running": False,
                "success": False,
                "message": "Synchronization failed",
                "error": result.stderr,
                "last_run": time.time(),
                "details": result.stdout
            })
            logger.error(f"Synchronization failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        update_sync_status({
            "running": False,
            "success": False,
            "message": "Synchronization timed out",
            "error": "Process exceeded 30 minute timeout",
            "last_run": time.time()
        })
        logger.error("Synchronization timed out")
    except Exception as e:
        update_sync_status({
            "running": False,
            "success": False,
            "message": f"Error running sync: {str(e)}",
            "error": str(e),
            "last_run": time.time()
        })
        logger.error(f"Error during sync: {e}")

# FRONTEND ROUTES
@app.route("/")
def index():
    """Serve the main frontend"""
    return render_template("index.html")

@app.route("/ui")
def frontend():
    """Alternative route to serve the frontend"""
    return render_template("index.html")

# API ROUTES
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "supabase_available": SUPABASE_AVAILABLE,
        "gemini_available": GEMINI_AVAILABLE,
        "supabase_url": supabase_url[:20] + "..." if supabase_url else None
    })

@app.route("/api/ask", methods=["POST"])
def ask():
    """Enhanced ask endpoint with intent recognition and proper greeting handling"""
    if not SUPABASE_AVAILABLE:
        return jsonify({
            "answer": {
                "text": "⚠️ Database connection not available.",
                "sources": []
            },
            "query": None,
            "status": "error"
        }), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "answer": {
                    "text": "⚠️ No data provided in request.",
                    "sources": []
                },
                "query": None,
                "status": "error"
            }), 400

        query = data.get("query", "").strip()
        if not query:
            return jsonify({
                "answer": {
                    "text": "⚠️ Please provide a question to search for.",
                    "sources": []
                },
                "query": None,
                "status": "error"
            }), 400

        logger.info(f"🔍 Processing query: '{query}'")
        
        # STEP 1: Classify the query intent
        intent = classify_query_intent(query)
        logger.info(f"🎯 Query intent: {intent['type']} (confidence: {intent['confidence']})")
        
        # STEP 2: Handle non-knowledge queries immediately
        if intent['type'] in ['greeting', 'help', 'status']:
            response_text = enhanced_generate_response_with_context(query, [], intent)
            return jsonify({
                "answer": {
                    "text": response_text,
                    "sources": []
                },
                "query": query,
                "status": "success",
                "intent": intent['type']
            })
        
        # STEP 3: For knowledge queries, check database and search
        if not test_database_populated():
            return jsonify({
                "answer": {
                    "text": "The knowledge base appears to be empty. Please run the sync process to load documents first, then you can ask me questions about your documents!",
                    "sources": []
                },
                "query": query,
                "status": "error"
            })

        # STEP 4: Search for knowledge queries
        search_results = []
        
        try:
            logger.info("🎯 Attempting document search...")
            search_results = enhanced_search_documents(query, limit=5)
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
        
        logger.info(f"📊 Search found {len(search_results)} results")
        
        # Log search method distribution
        search_methods = {}
        for result in search_results:
            method = result.get('search_type', 'unknown')
            search_methods[method] = search_methods.get(method, 0) + 1
        
        logger.info(f"🔍 Search methods used: {search_methods}")
        
        # STEP 5: Generate response with context
        response_text = enhanced_generate_response_with_context(query, search_results, intent)
        
        # Extract source filenames
        source_names = list(set([result.get("filename", "Unknown") for result in search_results if result.get("filename")]))
        
        return jsonify({
            "answer": {
                "text": response_text,
                "sources": source_names
            },
            "query": query,
            "status": "success",
            "intent": intent['type'],
            "debug": {
                "results_found": len(search_results),
                "search_methods": search_methods,
                "primary_method": "semantic" if search_methods.get('semantic', 0) > 0 else "text_fallback"
            }
        })
        
    except Exception as e:
        logger.exception("Error processing query")
        return jsonify({
            "answer": {
                "text": "I encountered an unexpected error while processing your question. Please try again, or feel free to ask me about my capabilities!",
                "sources": []
            },
            "query": query if 'query' in locals() else None,
            "status": "error",
            "debug": {"error": str(e)}
        }), 500

# Keep all the existing routes (stats, sync, etc.) - they remain unchanged
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    if not SUPABASE_AVAILABLE:
        return jsonify({
            "total_documents": 0,
            "total_files": 0,
            "status": "error",
            "error": "Supabase not available"
        }), 500
    
    try:
        raw_files_response = supabase.table("raw_files").select("id", count="exact").execute()
        total_files = raw_files_response.count if hasattr(raw_files_response, 'count') else len(raw_files_response.data)
        
        chunks_response = supabase.table("processed_chunks").select("id", count="exact").execute()
        total_documents = chunks_response.count if hasattr(chunks_response, 'count') else len(chunks_response.data)
        
        return jsonify({
            "total_documents": total_documents,
            "total_files": total_files,
            "status": "active"
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            "total_documents": 0,
            "total_files": 0,
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/sync', methods=['POST'])
def sync_knowledge_base():
    """Trigger knowledge base synchronization"""
    with sync_lock:
        if sync_status.get("running", False):
            return jsonify({
                "error": "Synchronization is already running",
                "current_status": sync_status.get("message", "Unknown status")
            }), 409
    
    sync_thread = threading.Thread(target=run_sync_script, daemon=True)
    sync_thread.start()
    time.sleep(0.5)
    
    with sync_lock:
        current_status = sync_status.copy()
    
    return jsonify({
        "message": "Synchronization started",
        "status": "running",
        "details": current_status
    })

@app.route('/api/sync/status', methods=['GET'])
def get_sync_status():
    """Get current sync status"""
    with sync_lock:
        status_copy = sync_status.copy()
    
    if status_copy.get("last_run"):
        status_copy["last_run_readable"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", 
            time.localtime(status_copy["last_run"])
        )
    
    return jsonify(status_copy)

@app.route('/api/sync/reset', methods=['POST'])
def reset_sync_status():
    """Reset sync status"""
    update_sync_status({
        "running": False,
        "message": "Status manually reset",
        "success": None,
        "error": None,
        "details": None
    })
    
    logger.info("Sync status manually reset")
    return jsonify({"message": "Sync status reset successfully"})

@app.route("/api/status", methods=["GET"])
def status():
    """Get system status"""
    with sync_lock:
        current_sync_status = sync_status.copy()
    
    supabase_connected = False
    if SUPABASE_AVAILABLE:
        try:
            supabase_connected = test_connection()
        except:
            pass
    
    return jsonify({
        "supabase_url": supabase_url[:20] + "..." if supabase_url else None,
        "supabase_connected": supabase_connected,
        "gemini_configured": GEMINI_AVAILABLE,
        "google_drive_folder_id": google_drive_folder_id[:10] + "..." if google_drive_folder_id else None,
        "sync_status": current_sync_status,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/search', methods=['POST'])
def search_endpoint():
    """Direct search endpoint"""
    if not SUPABASE_AVAILABLE:
        return jsonify({"error": "Supabase not available"}), 500
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        search_type = data.get("type", "text")
        limit = data.get("limit", 10)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        results = []
        if search_type == "semantic":
            results = safe_semantic_search(query, limit)
        else:
            results = enhanced_search_documents(query, limit)
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results),
            "search_type": search_type
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/files/<file_id>', methods=['GET'])
def get_file_details(file_id):
    """Get details of a specific file"""
    if not SUPABASE_AVAILABLE:
        return jsonify({"error": "Supabase not available"}), 500
    
    try:
        # Get file content
        file_content = get_file_content(file_id)
        if not file_content:
            return jsonify({"error": "File not found"}), 404
        
        # Get file chunks
        chunks = get_file_chunks(file_id)
        
        return jsonify({
            "file": file_content,
            "chunks": chunks,
            "chunk_count": len(chunks)
        })
        
    except Exception as e:
        logger.error(f"Error getting file details: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    logger.info(f"Starting Flask server on port {port}")
    logger.info(f"Frontend will be available at: http://127.0.0.1:{port}")
    logger.info(f"Alternative frontend URL: http://127.0.0.1:{port}/ui")
    
    # Display configuration
    logger.info("🔧 Configuration:")
    logger.info(f"   Supabase URL: {supabase_url[:30] + '...' if supabase_url else 'Not configured'}")
    logger.info(f"   Gemini API: {'Configured' if GEMINI_AVAILABLE else 'Not configured'}")
    logger.info(f"   Google Drive Folder: {google_drive_folder_id[:20] + '...' if google_drive_folder_id else 'Not configured'}")
    
    app.run(host="0.0.0.0", port=port, debug=debug)