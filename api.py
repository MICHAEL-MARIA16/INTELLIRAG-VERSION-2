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

# Set environment variable before importing chatbot
os.environ["TRANSFORMERS_NO_ADDITIONAL_MODULES"] = "1"

from chatbot import RAGChatbot

# Load environment variables first
load_dotenv()

# Initialize Flask app (only once!)
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:5000", "http://localhost:5000"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load from .env
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("COLLECTION_NAME", "documents")
gemini_api_key = os.getenv("GEMINI_API_KEY")

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

# Initialize chatbot
try:
    chatbot = RAGChatbot(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        gemini_api_key=gemini_api_key,
    )
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {e}")
    chatbot = None

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
        "chatbot_ready": chatbot is not None
    })

@app.route("/api/ask", methods=["POST"])
def ask():
    """Main query endpoint"""
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        logger.info(f"Processing query: {query[:50]}...")
        result = chatbot.chat(query)

        # Extract just file names from sources
        source_names = [s.get("file_name", "Unknown") for s in result.get("sources", [])]

        return jsonify({
            "answer": {
                "text": result.get("response", "No response generated."),
                "sources": source_names
            },
            "query": query,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        # Import QdrantManager from your utils
        from qdrant_utils import QdrantManager
        
        # Initialize your Qdrant manager
        qdrant_manager = QdrantManager(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )
        
        # Get stats from Qdrant
        total_documents = qdrant_manager.count_documents()
        
        # Get unique file count
        unique_files = qdrant_manager.count_unique_files()
        
        return jsonify({
            "total_documents": total_documents,
            "total_files": unique_files,
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
        # Path to your sync.py script
        sync_script_path = "sync.py"  # Adjust path as needed
        
        # Run the sync script
        result = subprocess.run([
            sys.executable, sync_script_path,
            "--mode", "full"
        ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
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

@app.route('/api/sync', methods=['POST'])
def sync_knowledge_base():
    """Trigger knowledge base synchronization"""
    with sync_lock:
        if sync_status.get("running", False):
            return jsonify({
                "error": "Synchronization is already running",
                "current_status": sync_status.get("message", "Unknown status")
            }), 409
    
    # Start sync in background thread
    sync_thread = threading.Thread(target=run_sync_script, daemon=True)
    sync_thread.start()
    
    # Give the thread a moment to start and update status
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
    
    # Add human-readable timestamp if available
    if status_copy.get("last_run"):
        status_copy["last_run_readable"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", 
            time.localtime(status_copy["last_run"])
        )
    
    return jsonify(status_copy)

@app.route('/api/sync/reset', methods=['POST'])
def reset_sync_status():
    """Reset sync status (useful for debugging stuck states)"""
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
    
    return jsonify({
        "qdrant_url": qdrant_url,
        "collection_name": collection_name,
        "chatbot_ready": chatbot is not None,
        "gemini_configured": bool(gemini_api_key),
        "sync_status": current_sync_status,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
    })

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
    app.run(host="0.0.0.0", port=port, debug=debug)