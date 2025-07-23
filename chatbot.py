import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
import asyncio
from pathlib import Path

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from qdrant_utils import QdrantManager
from drive_loader import GoogleDriveLoader

app = Flask(__name__)

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
        gemini_api_key: str,
        drive_folder_id: Optional[str] = None,
        credentials_path: str = "credentials.json",
        model_name: str = "gemini-1.5-flash"
    ):
        # Initialize Qdrant
        self.qdrant_manager = QdrantManager(qdrant_url, qdrant_api_key, collection_name)
        
        # Initialize embedding model (same as used in sync)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(model_name)
        
        # Initialize Drive loader (optional, for direct file access)
        self.drive_loader = None
        if drive_folder_id:
            self.drive_loader = GoogleDriveLoader(drive_folder_id, credentials_path)
        
        # Chat configuration
        self.max_context_length = 8000  # Gemini context window management
        self.search_limit = 5
        self.score_threshold = 0.6
        self.conversation_history = []
        
        # System prompt template
        self.system_prompt = """You are a helpful AI assistant with access to documents from a Google Drive folder. 
Use the provided context from the documents to answer questions accurately and comprehensively.

Guidelines:
- Base your answers primarily on the provided context from the documents
- If the context doesn't contain enough information, say so clearly
- Provide specific references to document names when possible
- Be conversational and helpful
- If asked about documents not in the context, explain that you can only access the synchronized documents

Context from documents:
{context}

Previous conversation (for reference):
{history}

User Question: {question}

Answer:"""

    def initialize(self) -> bool:
        """Initialize the chatbot system"""
        try:
            # Check Qdrant connection
            logger.info("Checking Qdrant connection...")
            if not self.qdrant_manager.health_check():
                logger.error("Failed to connect to Qdrant")
                return False
            logger.info("Qdrant connection successful")
            
            # Initialize Drive loader if provided
            if self.drive_loader:
                logger.info("Authenticating with Google Drive...")
                self.drive_loader.authenticate()
                logger.info("Drive loader authenticated")
            
            # Skip Gemini test during initialization to avoid hanging
            logger.info("Gemini API will be tested during first query")
            
            logger.info("RAG Chatbot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            return False

    def search_documents(self, query: str, limit: int = None, score_threshold: float = None) -> List[Dict[str, Any]]:
        """Search for relevant documents based on query"""
        try:
            # Use instance defaults if not provided
            limit = limit or self.search_limit
            score_threshold = score_threshold or self.score_threshold
            
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in Qdrant
            results = self.qdrant_manager.search_similar_documents(
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            logger.info(f"Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context for the LLM"""
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            file_name = result.get('file_name', 'Unknown')
            text = result.get('text', '')
            score = result.get('score', 0)
            
            context_parts.append(
                f"Document {i} (from {file_name}, relevance: {score:.2f}):\n{text}\n"
            )
        
        return "\n".join(context_parts)

    def format_conversation_history(self, limit: int = 3) -> str:
        """Format recent conversation history"""
        if not self.conversation_history:
            return "No previous conversation."
        
        # Get last `limit` exchanges
        recent_history = self.conversation_history[-limit:]
        history_parts = []
        
        for exchange in recent_history:
            history_parts.append(f"User: {exchange['user']}")
            history_parts.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(history_parts)

    def generate_response(self, query: str, context: str, history: str) -> str:
        """Generate response using Gemini"""
        try:
            # Format the prompt
            prompt = self.system_prompt.format(
                context=context,
                history=history,
                question=query
            )
            
            # Generate response
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"

    def chat(self, user_input: str) -> Dict[str, Any]:
        """Main chat function - process user input and return response"""
        try:
            start_time = datetime.now()
            
            # Search for relevant documents
            search_results = self.search_documents(user_input)
            
            # Format context and history
            context = self.format_context(search_results)
            history = self.format_conversation_history()
            
            # Generate response
            response = self.generate_response(user_input, context, history)
            
            # Update conversation history
            self.conversation_history.append({
                'user': user_input,
                'assistant': response,
                'timestamp': datetime.now().isoformat(),
                'sources': [result['file_name'] for result in search_results]
            })
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'response': response,
                'sources': search_results,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in chat function: {e}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'sources': [],
                'processing_time': 0,
                'timestamp': datetime.now().isoformat()
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            doc_count = self.qdrant_manager.count_documents()
            collection_info = self.qdrant_manager.get_collection_info()
            
            return {
                'qdrant_healthy': self.qdrant_manager.health_check(),
                'total_documents': doc_count,
                'collection_name': self.qdrant_manager.collection_name,
                'embedding_model': 'all-MiniLM-L6-v2',
                'llm_model': 'gemini-1.5-flash',
                'conversation_length': len(self.conversation_history),
                'last_updated': collection_info.config.params if collection_info else None
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def save_conversation_history(self, filepath: str):
        """Save conversation history to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Conversation history saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")

    def load_conversation_history(self, filepath: str):
        """Load conversation history from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info(f"Conversation history loaded from {filepath}")
            else:
                logger.warning(f"Conversation history file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")

    def search_files_by_name(self, filename_query: str) -> List[Dict[str, Any]]:
        """Search for files by filename (if Drive loader is available)"""
        if not self.drive_loader:
            return []
        
        try:
            files = self.drive_loader.get_files_in_folder()
            matching_files = [
                file for file in files 
                if filename_query.lower() in file['name'].lower()
            ]
            return matching_files
        except Exception as e:
            logger.error(f"Error searching files by name: {e}")
            return []

    def get_document_summary(self, file_name: str) -> Optional[str]:
        """Get a summary of a specific document"""
        try:
            # Search for all chunks from this file
            results = self.qdrant_manager.client.scroll(
                collection_name=self.qdrant_manager.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "file_name",
                            "match": {"value": file_name}
                        }
                    ]
                },
                limit=100,
                with_payload=True
            )
            
            if not results[0]:
                return None
            
            # Combine all text chunks
            all_text = []
            for point in results[0]:
                all_text.append(point.payload['text'])
            
            combined_text = "\n".join(all_text)
            
            # Generate summary using Gemini
            summary_prompt = f"""Please provide a concise summary of the following document content:

Document: {file_name}

Content:
{combined_text[:4000]}  # Limit to avoid token limits

Provide a summary in 2-3 paragraphs highlighting the main topics and key information."""

            response = self.gemini_model.generate_content(summary_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return None


def interactive_chat_mode(chatbot: RAGChatbot):
    """Run interactive chat mode"""
    print("RAG Chatbot Interactive Mode")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type '/status' to see system status")
    print("Type '/clear' to clear conversation history")
    print("Type '/summary <filename>' to get document summary")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if user_input == '/status':
                status = chatbot.get_system_status()
                print(f"\nSystem Status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input == '/clear':
                chatbot.clear_conversation_history()
                print("Conversation history cleared.")
                continue
            
            if user_input.startswith('/summary '):
                filename = user_input[9:].strip()
                summary = chatbot.get_document_summary(filename)
                if summary:
                    print(f"\nSummary of '{filename}':\n{summary}")
                else:
                    print(f"Could not generate summary for '{filename}' - file not found or error occurred.")
                continue
            
            # Process regular chat message
            print("Thinking...")
            result = chatbot.chat(user_input)
            
            print(f"\nAssistant: {result['response']}")
            
            if result['sources']:
                print(f"\nSources ({len(result['sources'])} documents):")
                for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                    print(f"  {i}. {source['file_name']} (relevance: {source['score']:.2f})")
            
            print(f"\nProcessing time: {result['processing_time']:.2f}s")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


# Global chatbot instance
chatbot = None

def initialize_chatbot():
    """Initialize the global chatbot instance"""
    global chatbot
    
    # Get environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("COLLECTION_NAME", os.getenv("QDRANT_COLLECTION"))
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    
    # Check required parameters
    if not all([qdrant_url, qdrant_api_key, collection_name, gemini_api_key]):
        logger.error("Missing required environment variables")
        return False
    
    # Initialize chatbot
    chatbot = RAGChatbot(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        gemini_api_key=gemini_api_key,
        drive_folder_id=drive_folder_id,
        credentials_path="credentials.json"
    )
    
    # Initialize system
    return chatbot.initialize()


# Flask routes
@app.route('/health')
def health():
    """Health check endpoint"""
    if chatbot is None:
        return jsonify({'status': 'error', 'message': 'Chatbot not initialized'}), 500
    
    try:
        status = chatbot.get_system_status()
        return jsonify({'status': 'healthy', 'system': status})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    if chatbot is None:
        return jsonify({'error': 'Chatbot not initialized'}), 500
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        result = chatbot.chat(query)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """System status endpoint"""
    if chatbot is None:
        return jsonify({'error': 'Chatbot not initialized'}), 500
    
    try:
        status = chatbot.get_system_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history endpoint"""
    if chatbot is None:
        return jsonify({'error': 'Chatbot not initialized'}), 500
    
    try:
        chatbot.clear_conversation_history()
        return jsonify({'message': 'Conversation history cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({
        "message": "👋 Welcome to the RAG Chatbot API!",
        "status": "online",
        "available_routes": [
            {"GET": "/health"},
            {"GET": "/status"},
            {"POST": "/chat"},
            {"POST": "/clear"}
        ]
    })


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot with Google Drive and Qdrant")
    parser.add_argument("--qdrant-url", help="Qdrant URL", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant-api-key", help="Qdrant API key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection-name", help="Qdrant collection name", default=os.getenv("COLLECTION_NAME", os.getenv("QDRANT_COLLECTION")))
    parser.add_argument("--gemini-api-key", help="Google Gemini API key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--drive-folder-id", help="Google Drive folder ID (optional)", default=os.getenv("GOOGLE_DRIVE_FOLDER_ID"))
    parser.add_argument("--credentials-path", default="credentials.json", help="Google credentials path")
    parser.add_argument("--mode", choices=["interactive", "api"], default="api", 
                       help="Run mode: interactive or api server")
    parser.add_argument("--query", help="Single query for non-interactive mode")
    parser.add_argument("--history-file", help="Load/save conversation history file")
    parser.add_argument("--port", type=int, help="Port for web server", default=int(os.getenv("PORT", 5000)))
    
    args = parser.parse_args()
    
    # Check required parameters for CLI modes
    if args.mode == "interactive" or args.query:
        if not args.qdrant_url:
            logger.error("Qdrant URL is required. Set QDRANT_URL in .env file or use --qdrant-url")
            return 1
        
        if not args.qdrant_api_key:
            logger.error("Qdrant API key is required. Set QDRANT_API_KEY in .env file or use --qdrant-api-key")
            return 1
        
        if not args.collection_name:
            logger.error("Collection name is required. Set COLLECTION_NAME or QDRANT_COLLECTION in .env file or use --collection-name")
            return 1
        
        if not args.gemini_api_key:
            logger.error("Gemini API key is required. Set GEMINI_API_KEY in .env file or use --gemini-api-key")
            return 1
        
        # Initialize chatbot for CLI modes
        global chatbot
        chatbot = RAGChatbot(
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            collection_name=args.collection_name,
            gemini_api_key=args.gemini_api_key,
            drive_folder_id=args.drive_folder_id,
            credentials_path=args.credentials_path
        )
        
        # Initialize system
        if not chatbot.initialize():
            logger.error("Failed to initialize chatbot system")
            return 1
        
        # Load conversation history if specified
        if args.history_file:
            chatbot.load_conversation_history(args.history_file)
    
    # Handle different modes
    if args.query:
        # Single query mode
        result = chatbot.chat(args.query)
        print(f"Question: {args.query}")
        print(f"Answer: {result['response']}")
        if result['sources']:
            print(f"Sources: {[s['file_name'] for s in result['sources']]}")
    
    elif args.mode == "interactive":
        # Interactive chat mode
        interactive_chat_mode(chatbot)
    
    elif args.mode == "api":
        # API server mode
        logger.info("Starting Flask API server...")
        if not initialize_chatbot():
            logger.error("Failed to initialize chatbot for API mode")
            return 1
        
        app.run(host='0.0.0.0', port=args.port, debug=False)
    
    # Save conversation history if specified
    if args.history_file and chatbot and chatbot.conversation_history:
        chatbot.save_conversation_history(args.history_file)
    
    return 0


if __name__ == '__main__':
    exit(main())