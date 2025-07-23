import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
from qdrant_client.models import PayloadSchemaType

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue, CollectionInfo, UpdateCollection, ScrollRequest
)
from qdrant_client.http.exceptions import ResponseHandlingException
from dotenv import load_dotenv

# Optional LangChain imports
try:
    from langchain_qdrant import QdrantVectorStore
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Install with: pip install langchain langchain-qdrant langchain-community")

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, url: str, api_key: str, collection_name: str, use_langchain: bool = False):
        print(f"Connecting to Qdrant at: {url}")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_size = 384  # all-MiniLM-L6-v2 embedding size
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        if self.use_langchain:
            self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_store = None
            logger.info("LangChain integration enabled")
        else:
            if use_langchain:
                logger.warning("LangChain requested but not available")
        
    def create_collection(self) -> bool:
        """Create collection if it doesn't exist with proper indexes"""
        try:
            collections = self.client.get_collections()
            collection_exists = any(
                collection.name == self.collection_name 
                for collection in collections.collections
            )

            if not collection_exists:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

            # Ensure all necessary indexes exist
            self._ensure_all_indexes()
            
            # Initialize LangChain vector store if enabled
            if self.use_langchain:
                self.vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embeddings=self.embeddings
                )
                logger.info("LangChain QdrantVectorStore initialized")
            
            return True

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False

    def _ensure_all_indexes(self) -> None:
        """Ensure all necessary payload indexes exist"""
        indexes_to_create = [
            ("file_id", PayloadSchemaType.KEYWORD),
            ("file_hash", PayloadSchemaType.KEYWORD),
            ("metadata.file_name", PayloadSchemaType.KEYWORD),
            ("metadata.mime_type", PayloadSchemaType.KEYWORD),
        ]
        
        for field_name, field_type in indexes_to_create:
            self.ensure_payload_index(field_name, field_type)

    def ensure_payload_index(self, field_name: str, field_type: PayloadSchemaType = PayloadSchemaType.KEYWORD) -> None:
        """Ensure a payload index exists on the given field"""
        try:
            # Get existing indexes
            try:
                existing_indexes = self.client.get_payload_indexes(collection_name=self.collection_name)
                indexed_fields = {idx.field_name for idx in existing_indexes}
            except AttributeError:
                # Fallback if get_payload_indexes doesn't exist in older versions
                logger.warning("get_payload_indexes not available, attempting to create index anyway")
                indexed_fields = set()
            
            if field_name not in indexed_fields:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"Created payload index on '{field_name}'")
            else:
                logger.info(f"Payload index on '{field_name}' already exists")
                
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Payload index on '{field_name}' already exists")
            else:
                logger.error(f"Failed to create payload index on '{field_name}': {e}")

    def sync_with_drive_files(self, drive_files: List[Dict[str, Any]], processed_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synchronize Qdrant with Google Drive files
        Returns sync statistics
        """
        sync_stats = {
            'files_added': 0,
            'files_updated': 0,
            'files_deleted': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_deleted': 0,
            'errors': []
        }
        
        try:
            # Get current state from Qdrant
            existing_files = self.get_all_file_info()
            drive_file_ids = {file_info['id'] for file_info in drive_files}
            existing_file_ids = set(existing_files.keys())
            
            # Determine what needs to be done
            files_to_delete = existing_file_ids - drive_file_ids
            files_to_process = []
            
            # Group processed documents by file_id for easier handling
            docs_by_file = {}
            for doc in processed_documents:
                file_id = doc['file_id']
                if file_id not in docs_by_file:
                    docs_by_file[file_id] = []
                docs_by_file[file_id].append(doc)
            
            # Check which files need updates
            for file_info in drive_files:
                file_id = file_info['id']
                current_hash = self._generate_file_hash(file_info)
                
                if file_id in existing_files:
                    stored_hash = existing_files[file_id].get('file_hash')
                    if stored_hash != current_hash:
                        files_to_process.append(('update', file_id, current_hash))
                        logger.info(f"File {file_info['name']} needs update (hash changed)")
                    else:
                        logger.debug(f"File {file_info['name']} unchanged")
                else:
                    files_to_process.append(('add', file_id, current_hash))
                    logger.info(f"New file {file_info['name']} needs to be added")
            
            # Delete files that no longer exist in Drive
            for file_id in files_to_delete:
                file_info = existing_files.get(file_id, {})
                file_name = file_info.get('file_name', file_id)
                logger.info(f"Deleting file {file_name} (no longer in Drive)")
                
                if self.delete_documents_by_file_id(file_id):
                    sync_stats['files_deleted'] += 1
                    # Count deleted documents
                    deleted_count = file_info.get('document_count', 0)
                    sync_stats['documents_deleted'] += deleted_count
                else:
                    sync_stats['errors'].append(f"Failed to delete file {file_name}")
            
            # Process files that need adding/updating
            for operation, file_id, file_hash in files_to_process:
                if file_id not in docs_by_file:
                    logger.warning(f"No processed documents found for file_id {file_id}")
                    continue
                
                documents = docs_by_file[file_id]
                
                if operation == 'update':
                    # Delete existing documents first
                    if self.delete_documents_by_file_id(file_id):
                        logger.info(f"Deleted existing documents for {file_id}")
                    else:
                        sync_stats['errors'].append(f"Failed to delete existing documents for {file_id}")
                        continue
                
                # Upsert new/updated documents
                if self.upsert_documents(documents):
                    if operation == 'add':
                        sync_stats['files_added'] += 1
                        sync_stats['documents_added'] += len(documents)
                        logger.info(f"Added {len(documents)} documents for new file {file_id}")
                    else:
                        sync_stats['files_updated'] += 1
                        sync_stats['documents_updated'] += len(documents)
                        logger.info(f"Updated {len(documents)} documents for file {file_id}")
                else:
                    sync_stats['errors'].append(f"Failed to upsert documents for {file_id}")
            
            return sync_stats
            
        except Exception as e:
            logger.error(f"Error in sync_with_drive_files: {e}")
            sync_stats['errors'].append(f"Sync error: {str(e)}")
            return sync_stats

    def get_all_file_info(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive file information from Qdrant"""
        try:
            file_info = {}
            offset = None
            
            while True:
                # Scroll through all points to get file information
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=['file_id', 'file_hash', 'file_name', 'metadata']
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    payload = point.payload
                    file_id = payload.get('file_id')
                    
                    if file_id:
                        if file_id not in file_info:
                            file_info[file_id] = {
                                'file_id': file_id,
                                'file_hash': payload.get('file_hash'),
                                'file_name': payload.get('file_name'),
                                'metadata': payload.get('metadata', {}),
                                'document_count': 0,
                                'last_updated': payload.get('updated_at')
                            }
                        
                        file_info[file_id]['document_count'] += 1
                
                offset = next_offset
                if offset is None:
                    break
            
            logger.info(f"Retrieved information for {len(file_info)} files from Qdrant")
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {}

    def _generate_file_hash(self, file_info: Dict[str, Any]) -> str:
        """Generate consistent hash for file identification"""
        import hashlib
        # Use file ID and modification time for hash
        hash_data = f"{file_info['id']}_{file_info.get('modifiedTime', '')}"
        return hashlib.md5(hash_data.encode()).hexdigest()

    def delete_documents_by_file_id(self, file_id: str) -> bool:
        """Delete all documents associated with a file (with fallback for missing index)"""
        try:
            # Try filter-based deletion first (requires index)
            result = self.client.delete(
                collection_name=self.collection_name,
                points_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_id",
                            match=MatchValue(value=file_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted documents for file_id: {file_id}")
            return True
            
        except Exception as e:
            if "Index required" in str(e) or "not found" in str(e):
                logger.warning(f"Index not available, using fallback deletion for {file_id}")
                return self._delete_by_scroll(file_id)
            else:
                logger.error(f"Error deleting documents for file_id {file_id}: {e}")
                return False

    def _delete_by_scroll(self, file_id: str) -> bool:
        """Fallback deletion method using scroll (when index is not available)"""
        try:
            deleted_count = 0
            offset = None
            
            while True:
                # Scroll through points to find matching file_id
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=['file_id']
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                # Find points with matching file_id
                ids_to_delete = []
                for point in points:
                    if point.payload and point.payload.get("file_id") == file_id:
                        ids_to_delete.append(point.id)
                
                # Delete matching points
                if ids_to_delete:
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=ids_to_delete
                    )
                    deleted_count += len(ids_to_delete)
                
                offset = next_offset
                if offset is None:
                    break
            
            logger.info(f"Fallback deletion: removed {deleted_count} documents for file_id {file_id}")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error in fallback deletion for file_id {file_id}: {e}")
            return False

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert or update documents in the collection with better error handling"""
        try:
            if not documents:
                logger.warning("No documents to upsert")
                return True
                
            if self.use_langchain:
                return self._upsert_documents_langchain(documents)
            else:
                return self._upsert_documents_native(documents)
            
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            return False
    
    def _upsert_documents_native(self, documents: List[Dict[str, Any]]) -> bool:
        """Native qdrant-client implementation with better error handling"""
        try:
            points = []
            for doc in documents:
                # Generate deterministic UUID based on file_id and chunk_index
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc['file_id']}_{doc['chunk_index']}"))
                
                # Ensure embedding is in correct format
                embedding = doc['embedding']
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif not isinstance(embedding, list):
                    embedding = list(embedding)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'file_id': doc['file_id'],
                        'file_name': doc['file_name'],
                        'file_hash': doc['file_hash'],
                        'chunk_index': doc['chunk_index'],
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'updated_at': datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            # Batch upsert with error handling
            batch_size = 100
            total_batches = (len(points) - 1) // batch_size + 1
            successful_batches = 0
            
            for i in range(0, len(points), batch_size):
                try:
                    batch = points[i:i + batch_size]
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True  # Wait for operation to complete
                    )
                    successful_batches += 1
                    logger.info(f"Upserted batch {successful_batches}/{total_batches}")
                    
                except Exception as batch_error:
                    logger.error(f"Error upserting batch {successful_batches + 1}: {batch_error}")
                    # Continue with next batch rather than failing completely
                    continue
            
            if successful_batches == total_batches:
                logger.info(f"Successfully upserted all {len(documents)} documents")
                return True
            else:
                logger.warning(f"Upserted {successful_batches}/{total_batches} batches successfully")
                return successful_batches > 0  # Return True if at least some batches succeeded
                
        except Exception as e:
            logger.error(f"Error in native upsert: {e}")
            return False
    
    def _upsert_documents_langchain(self, documents: List[Dict[str, Any]]) -> bool:
        """LangChain implementation"""
        if not self.vector_store:
            logger.error("LangChain vector store not initialized")
            return False
        
        try:
            # Convert documents to LangChain Document format
            langchain_docs = []
            doc_ids = []
            
            for doc in documents:
                langchain_doc = Document(
                    page_content=doc['text'],
                    metadata={
                        'file_id': doc['file_id'],
                        'file_name': doc['file_name'],
                        'file_hash': doc['file_hash'],
                        'chunk_index': doc['chunk_index'],
                        'updated_at': datetime.now().isoformat(),
                        **doc['metadata']
                    }
                )
                langchain_docs.append(langchain_doc)
                doc_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc['file_id']}_{doc['chunk_index']}")))
            
            # Add documents to vector store
            self.vector_store.add_documents(documents=langchain_docs, ids=doc_ids)
            logger.info(f"Successfully upserted {len(documents)} documents using LangChain")
            return True
            
        except Exception as e:
            logger.error(f"Error in LangChain upsert: {e}")
            return False

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sync statistics"""
        try:
            collection_info = self.get_collection_info()
            file_info = self.get_all_file_info()
            
            stats = {
                'total_documents': collection_info.points_count if collection_info else 0,
                'total_files': len(file_info),
                'files_by_type': {},
                'largest_file': None,
                'oldest_file': None,
                'newest_file': None,
                'collection_name': self.collection_name,
                'vector_size': self.vector_size,
                'last_updated': datetime.now().isoformat()
            }
            
            # Analyze files
            if file_info:
                largest_file_docs = 0
                oldest_time = None
                newest_time = None
                
                for file_id, info in file_info.items():
                    # File type stats
                    mime_type = info.get('metadata', {}).get('mime_type', 'unknown')
                    if mime_type not in stats['files_by_type']:
                        stats['files_by_type'][mime_type] = 0
                    stats['files_by_type'][mime_type] += 1
                    
                    # Largest file
                    if info['document_count'] > largest_file_docs:
                        largest_file_docs = info['document_count']
                        stats['largest_file'] = {
                            'name': info.get('file_name', 'unknown'),
                            'document_count': info['document_count']
                        }
                    
                    # Time tracking
                    updated_time = info.get('last_updated')
                    if updated_time:
                        if oldest_time is None or updated_time < oldest_time:
                            oldest_time = updated_time
                            stats['oldest_file'] = {
                                'name': info.get('file_name', 'unknown'),
                                'last_updated': updated_time
                            }
                        
                        if newest_time is None or updated_time > newest_time:
                            newest_time = updated_time
                            stats['newest_file'] = {
                                'name': info.get('file_name', 'unknown'),
                                'last_updated': updated_time
                            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting sync statistics: {e}")
            return {'error': str(e)}

    def search_similar_documents(
        self, 
        query_embedding: List[float] = None, 
        query_text: str = None,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if self.use_langchain and query_text:
                return self._search_langchain(query_text, limit, score_threshold)
            elif query_embedding:
                return self._search_native(query_embedding, limit, score_threshold)
            else:
                logger.error("Either query_embedding or query_text (with LangChain) must be provided")
                return []
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _search_native(self, query_embedding: List[float], limit: int, score_threshold: float) -> List[Dict[str, Any]]:
        """Native search implementation"""
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        results = []
        for result in search_results:
            results.append({
                'id': result.id,
                'score': result.score,
                'text': result.payload['text'],
                'metadata': result.payload.get('metadata', {}),
                'file_name': result.payload['file_name'],
                'file_id': result.payload['file_id']
            })
        
        return results
    
    def _search_langchain(self, query_text: str, limit: int, score_threshold: float) -> List[Dict[str, Any]]:
        """LangChain search implementation"""
        if not self.vector_store:
            logger.error("LangChain vector store not initialized")
            return []
        
        # Use similarity search with score
        results = self.vector_store.similarity_search_with_score(
            query=query_text,
            k=limit,
            score_threshold=score_threshold
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'id': doc.metadata.get('id', 'unknown'),
                'score': score,
                'text': doc.page_content,
                'metadata': doc.metadata,
                'file_name': doc.metadata.get('file_name', 'unknown'),
                'file_id': doc.metadata.get('file_id', 'unknown')
            })
        
        return formatted_results

    def get_all_file_hashes(self) -> Dict[str, str]:
        """Get all file hashes stored in the collection (backward compatibility)"""
        file_info = self.get_all_file_info()
        return {file_id: info.get('file_hash') for file_id, info in file_info.items() if info.get('file_hash')}
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get collection information"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def delete_collection(self) -> bool:
        """Delete the entire collection (use with caution!)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def count_documents(self) -> int:
        """Count total number of documents in the collection"""
        try:
            # Use the existing method but with improved error handling
            collection_info = self.get_collection_info()
            if collection_info:
                return collection_info.points_count
            else:
                # Fallback method using count API
                try:
                    count_result = self.client.count(
                        collection_name=self.collection_name,
                        count_filter=None,  # Count all documents
                    )
                    return count_result.count
                except Exception as fallback_error:
                    logger.error(f"Fallback count method also failed: {fallback_error}")
                    return 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0

    def count_unique_files(self) -> int:
        """Count unique files in the collection"""
        try:
            # Use the existing get_all_file_info method which already handles this efficiently
            file_info = self.get_all_file_info()
            return len(file_info)
            
        except Exception as e:
            logger.error(f"Error counting unique files: {e}")
            # Fallback to scroll method if get_all_file_info fails
            try:
                file_ids = set()
                offset = None
                
                while True:
                    scroll_result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=100,
                        offset=offset,
                        with_payload=['file_id'],
                        with_vector=False
                    )
                    
                    points, next_offset = scroll_result
                    
                    if not points:  # No more results
                        break
                        
                    for point in points:
                        if point.payload and 'file_id' in point.payload:
                            file_ids.add(point.payload['file_id'])
                    
                    offset = next_offset
                    if offset is None:
                        break
                
                logger.info(f"Fallback method found {len(file_ids)} unique files")
                return len(file_ids)
                
            except Exception as fallback_error:
                logger.error(f"Fallback count unique files method also failed: {fallback_error}")
                return 0

    def get_enhanced_collection_info(self) -> Dict[str, Any]:
        """Get detailed collection information with additional statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get additional stats
            unique_files = self.count_unique_files()
            
            enhanced_info = {
                "collection_name": self.collection_name,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "unique_files_count": unique_files,
                "config": {
                    "distance": collection_info.config.params.vectors.distance,
                    "size": collection_info.config.params.vectors.size
                },
                "retrieved_at": datetime.now().isoformat()
            }
            
            # Add average documents per file if we have both counts
            if unique_files > 0:
                enhanced_info["avg_documents_per_file"] = round(collection_info.points_count / unique_files, 2)
            else:
                enhanced_info["avg_documents_per_file"] = 0
            
            return enhanced_info
            
        except Exception as e:
            logger.error(f"Error getting enhanced collection info: {e}")
            return {
                "error": str(e),
                "collection_name": self.collection_name,
                "retrieved_at": datetime.now().isoformat()
            }

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics"""
        try:
            stats = {
                "basic_info": self.get_enhanced_collection_info(),
                "sync_stats": self.get_sync_statistics(),
                "health_status": self.health_check()
            }
            
            # Add performance metrics if available
            try:
                collection_info = self.client.get_collection(self.collection_name)
                if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'optimizer_config'):
                    stats["optimizer_config"] = {
                        "deleted_threshold": collection_info.config.optimizer_config.deleted_threshold,
                        "vacuum_min_vector_number": collection_info.config.optimizer_config.vacuum_min_vector_number,
                        "default_segment_number": collection_info.config.optimizer_config.default_segment_number,
                    }
            except Exception as optimizer_error:
                logger.debug(f"Could not retrieve optimizer config: {optimizer_error}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection statistics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Get all point IDs and delete them
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=False,
                with_vectors=False
            )
            
            point_ids = [point.id for point in points]
            
            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                logger.info(f"Cleared {len(point_ids)} documents from collection")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Qdrant client is healthy"""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    # Additional LangChain-specific methods
    def create_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None):
        """Create a LangChain retriever (only available when using LangChain)"""
        if not self.use_langchain or not self.vector_store:
            logger.error("LangChain not enabled or vector store not initialized")
            return None
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def get_langchain_vector_store(self):
        """Get the LangChain vector store instance"""
        return self.vector_store if self.use_langchain else None

    def get_documents_by_file_id(self, file_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve all documents for a specific file_id"""
        try:
            documents = []
            offset = None
            retrieved = 0
            
            while True:
                # Use scroll to get documents with the specific file_id
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=min(1000, limit - retrieved) if limit else 1000,
                    offset=offset,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_id",
                                match=MatchValue(value=file_id)
                            )
                        ]
                    ),
                    with_payload=True,
                    with_vector=False
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    documents.append({
                        'id': point.id,
                        'text': point.payload.get('text', ''),
                        'chunk_index': point.payload.get('chunk_index', 0),
                        'file_name': point.payload.get('file_name', ''),
                        'metadata': point.payload.get('metadata', {}),
                        'updated_at': point.payload.get('updated_at', '')
                    })
                    retrieved += 1
                    
                    if limit and retrieved >= limit:
                        break
                
                offset = next_offset
                if offset is None or (limit and retrieved >= limit):
                    break
            
            logger.info(f"Retrieved {len(documents)} documents for file_id: {file_id}")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents for file_id {file_id}: {e}")
            # Fallback method without filter
            return self._get_documents_by_file_id_fallback(file_id, limit)
    
    def _get_documents_by_file_id_fallback(self, file_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Fallback method to get documents by file_id when filtering is not available"""
        try:
            documents = []
            offset = None
            retrieved = 0
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=['file_id', 'text', 'chunk_index', 'file_name', 'metadata', 'updated_at'],
                    with_vector=False
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    if point.payload and point.payload.get('file_id') == file_id:
                        documents.append({
                            'id': point.id,
                            'text': point.payload.get('text', ''),
                            'chunk_index': point.payload.get('chunk_index', 0),
                            'file_name': point.payload.get('file_name', ''),
                            'metadata': point.payload.get('metadata', {}),
                            'updated_at': point.payload.get('updated_at', '')
                        })
                        retrieved += 1
                        
                        if limit and retrieved >= limit:
                            break
                
                offset = next_offset
                if offset is None or (limit and retrieved >= limit):
                    break
            
            logger.info(f"Fallback method retrieved {len(documents)} documents for file_id: {file_id}")
            return documents
            
        except Exception as e:
            logger.error(f"Error in fallback method for file_id {file_id}: {e}")
            return []

    def get_files_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all files in the collection"""
        try:
            file_info = self.get_all_file_info()
            
            summary = []
            for file_id, info in file_info.items():
                summary.append({
                    'file_id': file_id,
                    'file_name': info.get('file_name', 'Unknown'),
                    'document_count': info.get('document_count', 0),
                    'file_hash': info.get('file_hash', ''),
                    'last_updated': info.get('last_updated', ''),
                    'mime_type': info.get('metadata', {}).get('mime_type', 'unknown')
                })
            
            # Sort by document count (descending) for better overview
            summary.sort(key=lambda x: x['document_count'], reverse=True)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting files summary: {e}")
            return []

    def optimize_collection(self) -> bool:
        """Trigger collection optimization"""
        try:
            # This will trigger optimization of the collection
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=UpdateCollection.optimizer_config
            )
            logger.info(f"Triggered optimization for collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing collection: {e}")
            return False

    def backup_collection_info(self) -> Dict[str, Any]:
        """Create a backup of collection metadata and structure info"""
        try:
            backup_info = {
                'timestamp': datetime.now().isoformat(),
                'collection_name': self.collection_name,
                'collection_info': self.get_enhanced_collection_info(),
                'files_summary': self.get_files_summary(),
                'statistics': self.get_sync_statistics(),
                'vector_size': self.vector_size,
                'langchain_enabled': self.use_langchain
            }
            
            logger.info(f"Created backup info for collection: {self.collection_name}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Error creating backup info: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'collection_name': self.collection_name
            }

    def validate_collection_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of the collection"""
        try:
            integrity_report = {
                'timestamp': datetime.now().isoformat(),
                'collection_name': self.collection_name,
                'issues': [],
                'warnings': [],
                'stats': {}
            }
            
            # Check basic collection health
            if not self.health_check():
                integrity_report['issues'].append("Collection health check failed")
                return integrity_report
            
            # Get collection info
            collection_info = self.get_collection_info()
            if not collection_info:
                integrity_report['issues'].append("Could not retrieve collection information")
                return integrity_report
            
            integrity_report['stats']['total_points'] = collection_info.points_count
            
            # Check for orphaned documents (documents without proper file_id)
            orphaned_count = 0
            invalid_embeddings = 0
            offset = None
            
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=['file_id', 'text'],
                    with_vector=True
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    # Check for missing file_id
                    if not point.payload or not point.payload.get('file_id'):
                        orphaned_count += 1
                    
                    # Check for invalid embeddings
                    if not point.vector or len(point.vector) != self.vector_size:
                        invalid_embeddings += 1
                
                offset = next_offset
                if offset is None:
                    break
            
            integrity_report['stats']['orphaned_documents'] = orphaned_count
            integrity_report['stats']['invalid_embeddings'] = invalid_embeddings
            
            if orphaned_count > 0:
                integrity_report['warnings'].append(f"Found {orphaned_count} documents without file_id")
            
            if invalid_embeddings > 0:
                integrity_report['issues'].append(f"Found {invalid_embeddings} documents with invalid embeddings")
            
            # Check file consistency
            file_info = self.get_all_file_info()
            total_file_documents = sum(info.get('document_count', 0) for info in file_info.values())
            
            if total_file_documents != collection_info.points_count:
                integrity_report['warnings'].append(
                    f"Mismatch between total documents ({collection_info.points_count}) "
                    f"and sum of file documents ({total_file_documents})"
                )
            
            integrity_report['stats']['unique_files'] = len(file_info)
            integrity_report['stats']['total_file_documents'] = total_file_documents
            
            logger.info(f"Collection integrity validation completed for: {self.collection_name}")
            return integrity_report
            
        except Exception as e:
            logger.error(f"Error validating collection integrity: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'collection_name': self.collection_name
            }

    def cleanup_orphaned_documents(self) -> Dict[str, int]:
        """Remove documents that don't have valid file_id"""
        try:
            cleanup_stats = {
                'documents_checked': 0,
                'orphaned_found': 0,
                'documents_deleted': 0,
                'errors': 0
            }
            
            orphaned_ids = []
            offset = None
            
            # Find orphaned documents
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=['file_id'],
                    with_vector=False
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                for point in points:
                    cleanup_stats['documents_checked'] += 1
                    
                    # Check if document has valid file_id
                    if not point.payload or not point.payload.get('file_id') or point.payload.get('file_id').strip() == '':
                        orphaned_ids.append(point.id)
                        cleanup_stats['orphaned_found'] += 1
                
                offset = next_offset
                if offset is None:
                    break
            
            # Delete orphaned documents in batches
            batch_size = 100
            for i in range(0, len(orphaned_ids), batch_size):
                try:
                    batch_ids = orphaned_ids[i:i + batch_size]
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=batch_ids
                    )
                    cleanup_stats['documents_deleted'] += len(batch_ids)
                    logger.info(f"Deleted batch of {len(batch_ids)} orphaned documents")
                    
                except Exception as batch_error:
                    logger.error(f"Error deleting orphaned documents batch: {batch_error}")
                    cleanup_stats['errors'] += len(batch_ids)
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {
                'error': str(e),
                'documents_checked': 0,
                'orphaned_found': 0,
                'documents_deleted': 0,
                'errors': 0
            }