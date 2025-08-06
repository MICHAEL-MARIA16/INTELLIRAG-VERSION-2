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

from supabase_utils import (
    test_connection, 
    get_table_counts, 
    search_files, 
    semantic_search,
    get_file_content,
    get_file_chunks,
    supabase
)
from drive_loader import GoogleDriveLoader

app = Flask(__name__)

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(
        self,
        gemini_api_key: str,
        drive_folder_id: Optional[str] = None,
        credentials_path: str = "credentials.json",
        model_name: str = "gemini-1.5-flash"
    ):
        # Initialize embedding model (same as used in supabase_utils)
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings_available = True
            logger.info("✅ Embedding model loaded successfully")
        except ImportError:
            self.embedding_model = None
            self.embeddings_available = False
            logger.warning("⚠️ SentenceTransformers not available")
        
        # Initialize Gemini with proper error handling and model validation
        try:
            genai.configure(api_key=gemini_api_key)
            
            # Ensure we're using the correct model name
            if model_name == "gemini-pro":
                logger.warning("⚠️ gemini-pro may not be available, switching to gemini-1.5-flash")
                model_name = "gemini-1.5-flash"
            
            # Store the model name for validation
            self.model_name = model_name
            logger.info(f"🔧 Configuring Gemini with model: {model_name}")
            
            # Try to initialize the model and validate it works
            self.gemini_model = genai.GenerativeModel(model_name)
            logger.info(f"✅ Gemini model '{model_name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini model '{model_name}': {e}")
            # Try fallback models
            fallback_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
            for fallback in fallback_models:
                if fallback != model_name:
                    try:
                        logger.info(f"🔄 Trying fallback model: {fallback}")
                        self.gemini_model = genai.GenerativeModel(fallback)
                        self.model_name = fallback
                        logger.info(f"✅ Successfully switched to fallback model: {fallback}")
                        break
                    except Exception as fallback_e:
                        logger.warning(f"⚠️ Fallback model {fallback} also failed: {fallback_e}")
                        continue
            else:
                logger.error("❌ All model attempts failed")
                raise Exception("No available Gemini models found")
        
        # Initialize Drive loader (optional, for direct file access)
        self.drive_loader = None
        if drive_folder_id:
            self.drive_loader = GoogleDriveLoader(drive_folder_id, credentials_path)
        
        # Chat configuration
        self.max_context_length = 8000
        self.search_limit = 5
        self.score_threshold = 0.7
        self.conversation_history = []
        
        # FIXED: Single system prompt definition with proper greeting handling
        self.system_prompt = """You are IntelliRAG, a friendly and intelligent AI assistant specializing in information from Google Drive documents. You are designed to be helpful, conversational, and knowledgeable about your domain.

**Your personality:**
- Warm, friendly, and approachable
- Confident about information in your knowledge base
- Honest when information isn't available
- Conversational and natural in your responses

**How you should respond:**

🌟 **For greetings and social interactions** (hi, hello, hey, good morning, etc.):
- Respond warmly and naturally
- Introduce yourself as IntelliRAG when appropriate
- Explain that you specialize in information from Google Drive documents
- Ask how you can help with questions related to your knowledge base
- Example: "Hello! I'm IntelliRAG, your AI assistant. I have access to a specialized knowledge base built from Google Drive documents. How can I help you today?"

🎯 **For questions with relevant information in your knowledge base:**
- Provide comprehensive and accurate answers based on the context below
- Reference source documents when helpful
- Be confident and detailed in your responses
- Show enthusiasm about being able to help

❓ **For questions outside your knowledge base:**
- Politely explain that the information isn't in your current knowledge base
- Stay positive and friendly - don't apologize excessively
- Offer to help with topics that might be in your documents
- Suggest they can ask about what information you do have available
- Example: "I don't have information about that in my knowledge base, but I'd be happy to help with questions about the documents I have access to. Would you like to know what types of information I can help with?"

🤝 **For help requests and capability questions:**
- Explain what you can do: answer questions from your Google Drive knowledge base
- Mention you can search through documents, provide summaries, and find specific information
- Be enthusiastic about your capabilities within your domain

**Available context from your knowledge base:**
{context}

**Recent conversation:**
{history}

**User's question:**
{question}

Respond in a natural, helpful, and friendly way that matches the user's needs:"""

    def initialize(self) -> bool:
        """Initialize the chatbot system"""
        try:
            # Check Supabase connection
            logger.info("🔍 Checking Supabase connection...")
            if not test_connection():
                logger.error("❌ Failed to connect to Supabase")
                return False
            logger.info("✅ Supabase connection successful")
            
            # Get current table counts
            get_table_counts()
            
            # Initialize Drive loader if provided
            if self.drive_loader:
                logger.info("🔍 Authenticating with Google Drive...")
                self.drive_loader.authenticate()
                logger.info("✅ Drive loader authenticated")
            
            # Test Gemini API with a simple call
            logger.info("🔍 Testing Gemini API connection...")
            try:
                test_response = self.gemini_model.generate_content("Hello, this is a test.")
                logger.info(f"✅ Gemini API test successful with model: {self.model_name}")
                logger.info(f"📝 Test response length: {len(test_response.text)} characters")
            except Exception as e:
                logger.error(f"❌ Gemini API test failed: {e}")
                logger.error(f"🔧 Make sure model '{self.model_name}' is available in your region")
                return False
            
            logger.info("🎉 RAG Chatbot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing chatbot: {e}")
            return False

    def search_documents(self, query: str, limit: int = None, score_threshold: float = None) -> List[Dict[str, Any]]:
        """Enhanced search for relevant documents - WITH BETTER GREETING DETECTION"""
        try:
            # Use instance defaults if not provided
            limit = limit or self.search_limit
            score_threshold = score_threshold or self.score_threshold
            
            # ENHANCED greeting detection with more patterns
            greeting_patterns = [
                'hi', 'hii', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 
                'greetings', 'howdy', 'what\'s up', 'sup', 'yo', 'hiya', 'hey there'
            ]
            
            help_patterns = [
                'help', 'what can you do', 'capabilities', 'what do you know', 
                'what information', 'how can you help', 'what\'s available',
                'what can you tell me', 'what do you have', 'what\'s in your database',
                'how do you work', 'what are you', 'who are you'
            ]
            
            status_patterns = [
                'status', 'how many documents', 'what files', 'database status',
                'system status', 'health check', 'how many files'
            ]
            
            query_lower = query.lower().strip()
            query_words = query_lower.split()
            
            # Check for simple greetings (short queries with greeting words)
            if len(query_words) <= 3:
                if any(pattern in query_lower for pattern in greeting_patterns):
                    logger.info(f"🤝 Detected simple greeting: '{query}' - skipping document search")
                    return []
            
            # Check for help/capability requests
            if any(pattern in query_lower for pattern in help_patterns):
                logger.info(f"❓ Detected help request: '{query}' - skipping document search")
                return []
            
            # Check for status requests
            if any(pattern in query_lower for pattern in status_patterns):
                logger.info(f"📊 Detected status request: '{query}' - skipping document search")
                return []
            
            # If we get here, it's a knowledge query - proceed with search
            logger.info(f"🔍 Knowledge query detected: '{query}' - proceeding with document search")
            
            search_results = []
            
            # PRIORITY 1: Semantic search if embeddings are available
            if self.embeddings_available:
                try:
                    logger.info(f"🎯 Using SEMANTIC search for query: '{query}'")
                    semantic_results = semantic_search(query, limit=limit)
                    
                    if semantic_results:
                        # Convert semantic results to our expected format
                        for result in semantic_results:
                            # Apply score threshold for semantic results
                            similarity_score = result.get('similarity', 0.0)
                            if similarity_score >= score_threshold:
                                search_results.append({
                                    'file_name': result.get('filename', 'Unknown'),
                                    'text': result.get('chunk_text', ''),
                                    'score': similarity_score,
                                    'chunk_index': result.get('chunk_index'),
                                    'source': 'semantic'
                                })
                        
                        logger.info(f"🎯 Semantic search found {len(search_results)} results above threshold {score_threshold}")
                        
                        # If we have good semantic results, return them
                        if search_results:
                            return search_results
                    else:
                        logger.warning("⚠️ Semantic search returned no results")
                        
                except Exception as e:
                    logger.error(f"❌ Semantic search failed: {e}")
            else:
                logger.warning("⚠️ Embeddings not available - cannot use semantic search")
            
            # FALLBACK: Text search only if semantic search failed
            logger.info(f"🔍 Falling back to TEXT search for query: '{query}'")
            try:
                text_results = search_files(query, limit=limit)
                if text_results:
                    for result in text_results:
                        if result.get('source') == 'chunk':
                            text_content = result.get('chunk_text', '')
                        else:
                            file_content = get_file_content(result.get('file_id'))
                            if file_content:
                                text_content = file_content.get('full_text', '')[:1000]
                            else:
                                text_content = ''
                        
                        search_results.append({
                            'file_name': result.get('filename', 'Unknown'),
                            'text': text_content,
                            'score': 0.8,
                            'chunk_index': result.get('chunk_index'),
                            'source': 'text_fallback'
                        })
                    logger.info(f"📝 Text search found {len(text_results)} fallback results")
            except Exception as e:
                logger.error(f"❌ Text search also failed: {e}")
            
            logger.info(f"📊 Total documents found: {len(search_results)}")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Error searching documents: {e}")
            return []

    def classify_query_intent(self, query: str) -> str:
        """Classify the intent of the user's query"""
        query_lower = query.lower().strip()
        query_words = query_lower.split()
        
        # Simple greeting patterns
        greeting_patterns = ['hi', 'hii', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
        help_patterns = ['help', 'what can you do', 'capabilities', 'what do you know', 'how can you help']
        status_patterns = ['status', 'how many documents', 'what files', 'database status']
        
        # Short greetings
        if len(query_words) <= 3 and any(pattern in query_lower for pattern in greeting_patterns):
            return "greeting"
        
        # Help requests
        if any(pattern in query_lower for pattern in help_patterns):
            return "help"
        
        # Status requests
        if any(pattern in query_lower for pattern in status_patterns):
            return "status"
        
        # Default to knowledge query
        return "knowledge"

    def generate_response(self, query: str, context: str, history: str) -> str:
        """Enhanced response generation with better intent handling"""
        try:
            # Classify the query intent
            intent = self.classify_query_intent(query)
            
            logger.info(f"🤖 Query intent classified as: {intent}")
            logger.info(f"🤖 Generating response using model: {self.model_name}")
            
            # Handle different intents with specialized responses
            if intent == "greeting":
                return self.generate_greeting_response()
            elif intent == "help":
                return self.generate_help_response() 
            elif intent == "status":
                return self.generate_status_response()
            
            # For knowledge queries, use the enhanced system prompt
            enhanced_system_prompt = """You are IntelliRAG, a friendly and intelligent AI assistant specializing in information from Google Drive documents. 

**Your personality:**
- Warm, friendly, and conversational
- Confident about information in your knowledge base
- Honest when information isn't available
- Professional but approachable

**For this knowledge query, use the context below to provide a comprehensive answer:**

**Available context from your knowledge base:**
{context}

**Recent conversation (for context):**
{history}

**User's question:**
{question}

**Instructions:**
- Answer based primarily on the provided context from the knowledge base
- Be conversational and friendly in your tone
- If the context contains relevant information, use it confidently and cite sources naturally
- If context doesn't fully address the question, mention what information IS available
- Stay focused on helping the user with their knowledge base content
- Be enthusiastic about your ability to help with document analysis

Provide a helpful, accurate response:"""

            # Check if we have meaningful context
            has_context = context != "No relevant documents found in knowledge base for this query." and context.strip()
            
            if not has_context:
                return """I couldn't find specific information about that topic in your knowledge base. 

However, I'm here to help you explore your document collection! I can:
- Search through all your Google Drive documents
- Find specific information and provide detailed answers
- Summarize document content
- Compare information across multiple files

Would you like to ask about something specific, or would you like me to tell you what types of documents I have access to?"""
            
            # Format the prompt
            prompt = enhanced_system_prompt.format(
                context=context,
                history=history if history != "No previous conversation." else "",
                question=query
            )
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                top_k=40
            )
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            logger.info(f"✅ Response generated successfully")
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Error generating response with {self.model_name}: {e}")
            return self.handle_generation_error(e)
    
    def generate_greeting_response(self) -> str:
        """Generate a friendly greeting response"""
        greetings = [
            "Hello! I'm IntelliRAG, your AI assistant for exploring and analyzing your Google Drive documents.",
            "Hi there! I'm IntelliRAG, here to help you find information from your document collection.",
            "Hey! I'm IntelliRAG, your friendly AI assistant specializing in document analysis and Q&A."
        ]
        
        capabilities = [
            "I can search through your documents, answer questions about their content, and provide detailed summaries.",
            "I'm designed to help you explore your knowledge base and find exactly what you're looking for.",
            "I can analyze your documents and provide comprehensive answers based on their content."
        ]
        
        import random
        greeting = random.choice(greetings)
        capability = random.choice(capabilities)
        
        return f"{greeting}\n\n{capability}\n\nWhat would you like to know about your documents today? 📚"
    
    def generate_help_response(self) -> str:
        """Generate a helpful capabilities response"""
        return """🤖 **I'm IntelliRAG - Here's how I can help you:**

**🔍 Document Search & Analysis:**
- Search across all your Google Drive documents using natural language
- Find specific information, facts, or data points
- Understand context and meaning, not just keywords

**📊 What I Can Do:**
- Answer detailed questions about your document content
- Provide summaries of specific files or topics
- Compare information across multiple documents
- Extract key insights and data points
- Explain complex information in simple terms

**💡 Pro Tips:**
- Ask specific questions for the best results
- Use natural language - I understand context!
- Request comparisons between different documents
- Ask for summaries of specific topics

**🎯 Example Questions:**
- "What does the quarterly report say about sales performance?"
- "Summarize the main points from the project proposal"
- "Find information about budget allocations in the finance docs"
- "What are the key risks mentioned across all documents?"

Ready to explore your knowledge base? Ask me anything! 🚀"""
    
    def generate_status_response(self) -> str:
        """Generate a system status response"""
        try:
            # Get database statistics
            raw_count = supabase.table("raw_files").select("id", count="exact").execute()
            chunks_count = supabase.table("processed_chunks").select("id", count="exact").execute()
            
            total_files = len(raw_count.data) if raw_count.data else 0
            total_chunks = len(chunks_count.data) if chunks_count.data else 0
            
            return f"""📊 **IntelliRAG System Status:**

**📁 Knowledge Base:**
- **Documents Indexed:** {total_files} files
- **Content Chunks:** {total_chunks} searchable segments
- **Search Capability:** {'🟢 Active' if total_files > 0 else '🔴 No documents found'}

**🧠 AI Features:**
- **Semantic Search:** {'🟢 Available' if self.embeddings_available else '🔴 Unavailable'}
- **Text Analysis:** 🟢 Active
- **Conversational AI:** 🟢 Active (Model: {self.model_name})

**📈 Performance:**
- **Response Quality:** High (context-aware answers)
- **Search Accuracy:** Enhanced with multiple methods
- **Conversation Memory:** {len(self.conversation_history)} exchanges

{'🎉 Your knowledge base is ready for questions!' if total_files > 0 else '⚠️ Please sync your Google Drive documents to start asking questions.'}"""

        except Exception as e:
            return f"📊 **System Status:** Unable to retrieve detailed status. Basic systems are operational.\n\nError details: {str(e)}"
    
    def handle_generation_error(self, error: Exception) -> str:
        """Handle various types of generation errors gracefully"""
        error_str = str(error).lower()
        
        if "404" in error_str and "not found" in error_str:
            return "I'm experiencing a technical issue with my AI model. Please try again in a moment, or ask me about my capabilities!"
        elif "quota" in error_str or "limit" in error_str:
            return "I'm experiencing high demand right now. Please try again in a few minutes, or feel free to ask me about what I can help you with!"
        elif "permission_denied" in error_str:
            return "I'm having authentication issues. Please contact support if this persists. In the meantime, feel free to ask about my capabilities!"
        else:
            return "I encountered a technical issue, but I'm still here to help! Try rephrasing your question or ask me what I can do for you. 🤖"
        
    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Enhanced context formatting"""
        if not search_results:
            return "No relevant documents found in knowledge base for this query."
        
        context_parts = []
        context_parts.append(f"Found {len(search_results)} relevant document(s):")
        context_parts.append("")
        
        for i, result in enumerate(search_results, 1):
            file_name = result.get('file_name', 'Unknown Document')
            text = result.get('text', '').strip()
            score = result.get('score', 0)
            
            context_parts.append(f"--- Document {i}: {file_name} (Relevance: {score:.2f}) ---")
            context_parts.append(text)
            context_parts.append("")
        
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

    def classify_query_type(self, query: str) -> str:
        """Classify the type of query for logging purposes"""
        query_lower = query.lower().strip()
        
        greeting_keywords = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        help_keywords = ['help', 'what can you do', 'capabilities', 'what do you know']
        
        if any(keyword in query_lower for keyword in greeting_keywords):
            return "Greeting"
        elif any(keyword in query_lower for keyword in help_keywords):
            return "Help/Capabilities"
        else:
            return "Knowledge Query"

    def test_gemini_connection(self) -> bool:
        """Test Gemini API connection with current model"""
        try:
            logger.info(f"🧪 Testing Gemini connection with model: {self.model_name}")
            
            test_response = self.gemini_model.generate_content("Hello, this is a connection test.")
            
            if test_response and test_response.text:
                logger.info(f"✅ Gemini connection test successful")
                logger.info(f"📝 Test response: {test_response.text[:100]}...")
                return True
            else:
                logger.error("❌ Gemini connection test failed - no response received")
                return False
                
        except Exception as e:
            logger.error(f"❌ Gemini connection test failed: {e}")
            
            # Check if it's a model availability issue
            if "404" in str(e) and "not found" in str(e):
                logger.error(f"🔧 Model '{self.model_name}' appears to not be available")
                logger.info("💡 Try using one of these models instead:")
                logger.info("   - gemini-1.5-flash")
                logger.info("   - gemini-1.5-pro") 
            
            return False

    def list_available_models(self):
        """List available Gemini models"""
        try:
            logger.info("📋 Listing available Gemini models...")
            models = genai.list_models()
            
            generation_models = []
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    generation_models.append(model.name)
                    logger.info(f"   ✅ {model.name}")
            
            if not generation_models:
                logger.warning("⚠️ No models found that support content generation")
            
            return generation_models
            
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return []

    def chat(self, user_input: str) -> Dict[str, Any]:
        """Main chat function - process user input and return response"""
        try:
            start_time = datetime.now()
            
            # Search for relevant documents (will return empty for greetings)
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
            logger.error(f"❌ Error in chat function: {e}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'sources': [],
                'processing_time': 0,
                'timestamp': datetime.now().isoformat()
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            # Test Supabase connection
            supabase_healthy = test_connection()
            
            # Test Gemini connection
            gemini_healthy = self.test_gemini_connection()
            
            # Get table counts
            raw_files_count = 0
            chunks_count = 0
            try:
                raw_files_response = supabase.table("raw_files").select("id", count="exact").execute()
                raw_files_count = raw_files_response.count if hasattr(raw_files_response, 'count') else len(raw_files_response.data)
                
                chunks_response = supabase.table("processed_chunks").select("id", count="exact").execute()
                chunks_count = chunks_response.count if hasattr(chunks_response, 'count') else len(chunks_response.data)
            except Exception as e:
                logger.error(f"❌ Error getting counts: {e}")
            
            return {
                'supabase_healthy': supabase_healthy,
                'gemini_healthy': gemini_healthy,
                'gemini_model': self.model_name,
                'total_raw_files': raw_files_count,
                'total_chunks': chunks_count,
                'embedding_model': 'all-MiniLM-L6-v2',
                'embeddings_available': self.embeddings_available,
                'conversation_length': len(self.conversation_history),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {'error': str(e)}

    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("🗑️ Conversation history cleared")

    def save_conversation_history(self, filepath: str):
        """Save conversation history to file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Conversation history saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Error saving conversation history: {e}")

    def load_conversation_history(self, filepath: str):
        """Load conversation history from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info(f"📂 Conversation history loaded from {filepath}")
            else:
                logger.warning(f"⚠️ Conversation history file not found: {filepath}")
        except Exception as e:
            logger.error(f"❌ Error loading conversation history: {e}")

    def search_files_by_name(self, filename_query: str) -> List[Dict[str, Any]]:
        """Search for files by filename"""
        try:
            # Search in Supabase raw_files table
            response = supabase.table("raw_files").select("*").ilike("filename", f"%{filename_query}%").execute()
            
            matching_files = []
            if response.data:
                for file in response.data:
                    matching_files.append({
                        'id': file['drive_file_id'],
                        'name': file['filename'],
                        'supabase_id': file['id']
                    })
            
            return matching_files
        except Exception as e:
            logger.error(f"❌ Error searching files by name: {e}")
            return []

    def get_document_summary(self, file_name: str) -> Optional[str]:
        """Get a summary of a specific document"""
        try:
            # Search for the file by name
            response = supabase.table("raw_files").select("*").ilike("filename", f"%{file_name}%").execute()
            
            if not response.data:
                return None
            
            file = response.data[0]
            full_text = file.get('full_text', '')
            
            if not full_text:
                return None
            
            # Generate summary using Gemini
            summary_prompt = f"""Please provide a concise summary of the following document content:

Document: {file['filename']}

Content:
{full_text[:4000]}  # Limit to avoid token limits

Provide a summary in 2-3 paragraphs highlighting the main topics and key information."""

            response = self.gemini_model.generate_content(summary_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Error generating document summary: {e}")
            return None


def interactive_chat_mode(chatbot: RAGChatbot):
    """Run interactive chat mode"""
    print("🤖 RAG Chatbot Interactive Mode")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type '/status' to see system status")
    print("Type '/clear' to clear conversation history")
    print("Type '/models' to list available Gemini models")
    print("Type '/test' to test Gemini connection")
    print("Type '/summary <filename>' to get document summary")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if user_input == '/status':
                status = chatbot.get_system_status()
                print(f"\n📊 System Status:")
                for key, value in status.items():
                    if key == 'gemini_healthy':
                        emoji = "✅" if value else "❌"
                        print(f"  {emoji} {key}: {value}")
                    else:
                        print(f"  📈 {key}: {value}")
                continue
            
            if user_input == '/models':
                print("\n📋 Listing available Gemini models...")
                models = chatbot.list_available_models()
                if models:
                    print(f"Current model: {chatbot.model_name}")
                else:
                    print("❌ Could not retrieve model list")
                continue
            
            if user_input == '/test':
                print("\n🧪 Testing Gemini connection...")
                if chatbot.test_gemini_connection():
                    print("✅ Gemini connection is working!")
                else:
                    print("❌ Gemini connection failed")
                continue
            
            if user_input == '/clear':
                chatbot.clear_conversation_history()
                print("🗑️ Conversation history cleared.")
                continue
            
            if user_input.startswith('/summary '):
                filename = user_input[9:].strip()
                summary = chatbot.get_document_summary(filename)
                if summary:
                    print(f"\n📄 Summary of '{filename}':\n{summary}")
                else:
                    print(f"❌ Could not generate summary for '{filename}' - file not found or error occurred.")
                continue
            
            # Process regular chat message
            print("🤔 Thinking...")
            result = chatbot.chat(user_input)
            
            print(f"\n🤖 Assistant: {result['response']}")
            
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])} documents):")
                for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                    print(f"  {i}. {source['file_name']} (relevance: {source['score']:.2f})")
            
            print(f"\n⏱️ Processing time: {result['processing_time']:.2f}s")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# Global chatbot instance
chatbot = None

def initialize_chatbot():
    """Initialize the global chatbot instance"""
    global chatbot
    
    # Get environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    drive_folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    
    # Check required parameters
    if not gemini_api_key:
        logger.error("❌ Missing required environment variable: GEMINI_API_KEY")
        return False
    
    # Initialize chatbot with explicit model name
    chatbot = RAGChatbot(
        gemini_api_key=gemini_api_key,
        drive_folder_id=drive_folder_id,
        credentials_path="credentials.json",
        model_name="gemini-1.5-flash"
    )
    
    # Initialize system
    success = chatbot.initialize()
    
    if success:
        logger.info("🎉 Chatbot initialization completed successfully")
    else:
        logger.error("❌ Chatbot initialization failed")
    
    return success


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
        return jsonify({"answer": result})
    
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
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


@app.route('/models')
def list_models():
    """List available models endpoint"""
    if chatbot is None:
        return jsonify({'error': 'Chatbot not initialized'}), 500
    
    try:
        models = chatbot.list_available_models()
        return jsonify({
            'current_model': chatbot.model_name,
            'available_models': models
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/')
def home():
    return jsonify({
        "message": "👋 Welcome to the RAG Chatbot API!",
        "status": "online",
        "database": "Supabase",
        "ai_model": "Gemini 1.5 Flash",
        "available_routes": [
            {"GET": "/health"},
            {"GET": "/status"},
            {"GET": "/models"},
            {"POST": "/chat"},
            {"POST": "/clear"}
        ]
    })


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot with Google Drive and Supabase")
    parser.add_argument("--gemini-api-key", help="Google Gemini API key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--drive-folder-id", help="Google Drive folder ID (optional)", default=os.getenv("GOOGLE_DRIVE_FOLDER_ID"))
    parser.add_argument("--credentials-path", default="credentials.json", help="Google credentials path")
    parser.add_argument("--model-name", default="gemini-1.5-flash", help="Gemini model name")
    parser.add_argument("--mode", choices=["interactive", "api"], default="api", 
                       help="Run mode: interactive or api server")
    parser.add_argument("--query", help="Single query for non-interactive mode")
    parser.add_argument("--history-file", help="Load/save conversation history file")
    parser.add_argument("--port", type=int, help="Port for web server", default=int(os.getenv("PORT", 5000)))
    
    args = parser.parse_args()
    
    # Check required parameters for CLI modes
    if args.mode == "interactive" or args.query:
        if not args.gemini_api_key:
            logger.error("❌ Gemini API key is required. Set GEMINI_API_KEY in .env file or use --gemini-api-key")
            return 1
        
        # Initialize chatbot for CLI modes
        global chatbot
        chatbot = RAGChatbot(
            gemini_api_key=args.gemini_api_key,
            drive_folder_id=args.drive_folder_id,
            credentials_path=args.credentials_path,
            model_name=args.model_name
        )
        
        # Initialize system
        if not chatbot.initialize():
            logger.error("❌ Failed to initialize chatbot system")
            return 1
        
        # Load conversation history if specified
        if args.history_file:
            chatbot.load_conversation_history(args.history_file)
    
    # Handle different modes
    if args.query:
        # Single query mode
        result = chatbot.chat(args.query)
        print(f"❓ Question: {args.query}")
        print(f"🤖 Answer: {result['response']}")
        if result['sources']:
            print(f"📚 Sources: {[s['file_name'] for s in result['sources']]}")
    
    elif args.mode == "interactive":
        # Interactive chat mode
        interactive_chat_mode(chatbot)
    
    elif args.mode == "api":
        # API server mode
        logger.info("🚀 Starting Flask API server...")
        if not initialize_chatbot():
            logger.error("❌ Failed to initialize chatbot for API mode")
            return 1
        
        app.run(host='0.0.0.0', port=args.port, debug=False)
    
    # Save conversation history if specified
    if args.history_file and chatbot and chatbot.conversation_history:
        chatbot.save_conversation_history(args.history_file)
    
    return 0


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # No arguments passed — default to API mode for Railway
        os.environ["FLASK_RUN_FROM_CLI"] = "false"
        if not initialize_chatbot():
            logger.error("❌ Failed to initialize chatbot for API mode")
            exit(1)
        app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
    else:
        exit(main())