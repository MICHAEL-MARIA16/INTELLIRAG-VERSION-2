import os
import logging
import schedule
import time
from typing import Set, Dict, Any, List
from datetime import datetime, timedelta
import threading
import argparse
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
import signal
import sys
import platform
from supabase_utils import insert_raw_file, insert_processed_chunks, get_existing_file_hashes, delete_file_by_drive_id, test_connection, get_table_counts

# Optional LangChain imports
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

load_dotenv()  # this loads .env from the root by default

from drive_loader import GoogleDriveLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"

# Add timeout handler
def timeout_handler(signum, frame):
    logger.error("TIMEOUT: Operation took too long, forcing exit")
    sys.exit(1)

class DriveSupabaseSync:
    def __init__(
        self,
        folder_id: str,
        credentials_path: str = "credentials.json",
        use_langchain: bool = False
    ):
        self.drive_loader = GoogleDriveLoader(folder_id, credentials_path)
        self.last_sync_time = None
        self.sync_lock = threading.Lock()
        self.config_file = "sync_config.json"
        self.use_langchain = use_langchain
        
        # Initialize LangChain text splitter if available
        if self.use_langchain and LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            logger.info("LangChain text splitter initialized")
        
    def initialize(self) -> bool:
        """Initialize the sync system"""
        try:
            logger.info("STEP 1: Authenticating with Google Drive...")
            # Authenticate with Google Drive
            self.drive_loader.authenticate()
            logger.info("STEP 1: ✓ Google Drive authentication successful")
            
            logger.info("STEP 2: Testing Supabase connection...")
            if not test_connection():
                logger.error("STEP 2: ✗ Supabase connection FAILED")
                return False
            logger.info("STEP 2: ✓ Supabase connection ready")
            get_table_counts()
            
            logger.info("STEP 3: Loading sync configuration...")
            # Load last sync time from config file
            self.load_sync_config()
            logger.info("STEP 3: ✓ Sync configuration loaded")
            
            logger.info("✓ Sync system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error initializing sync system: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def load_sync_config(self):
        """Load sync configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    if 'last_sync_time' in config:
                        self.last_sync_time = datetime.fromisoformat(config['last_sync_time'])
                        logger.info(f"Loaded last sync time: {self.last_sync_time}")
        except Exception as e:
            logger.warning(f"Could not load sync config: {e}")
    
    def save_sync_config(self):
        """Save sync configuration to file"""
        try:
            config = {}
            if self.last_sync_time:
                config['last_sync_time'] = self.last_sync_time.isoformat()
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save sync config: {e}")
    
    def perform_full_sync(self) -> bool:
        """Perform a complete synchronization"""
        with self.sync_lock:
            try:
                logger.info("🚀 STARTING FULL SYNCHRONIZATION...")

                # Setup timeout using signal only on non-Windows
                if not IS_WINDOWS:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(1800)  # 30 minutes
                
                logger.info("PHASE 1: Fetching files from Google Drive...")
                start_time = time.time()
                
                try:
                    drive_files = self.drive_loader.get_files_in_folder()
                    logger.info(f"PHASE 1: ✓ Found {len(drive_files)} files in Drive folder ({time.time() - start_time:.2f}s)")
                except Exception as e:
                    logger.error(f"PHASE 1: ✗ Failed to get files from Drive: {e}")
                    return False
                
                # Show file details for debugging
                logger.info("Files found:")
                for i, file in enumerate(drive_files[:5]):  # Show first 5 files
                    size = file.get('size', 'unknown')
                    logger.info(f"  {i+1}. {file['name']} (ID: {file['id'][:10]}...) (Size: {size} bytes)")
                if len(drive_files) > 5:
                    logger.info(f"  ... and {len(drive_files) - 5} more files")
                
                drive_file_ids = {file['id'] for file in drive_files}
                
                logger.info("PHASE 2: Computing file hashes...")
                start_time = time.time()
                drive_file_hashes = {}
                
                for i, file in enumerate(drive_files, 1):
                    logger.info(f"PHASE 2: Computing hash for file {i}/{len(drive_files)}: {file['name']}")
                    try:
                        file_start = time.time()
                        drive_file_hashes[file['id']] = self.drive_loader.get_file_hash(file)
                        logger.info(f"PHASE 2: ✓ Hash computed in {time.time() - file_start:.2f}s")
                    except Exception as e:
                        logger.error(f"PHASE 2: ✗ Error computing hash for {file['name']}: {e}")
                        continue
                
                logger.info(f"PHASE 2: ✓ All hashes computed ({time.time() - start_time:.2f}s)")
                
                logger.info("PHASE 3: Fetching existing file hashes from Supabase...")
                start_time = time.time()
                try:
                    existing_hashes = get_existing_file_hashes()
                    logger.info(f"PHASE 3: ✓ Retrieved {len(existing_hashes)} existing hashes ({time.time() - start_time:.2f}s)")
                except Exception as e:
                    logger.error(f"PHASE 3: ✗ Error getting existing hashes: {e}")
                    return False
                
                existing_file_ids = set(existing_hashes.keys())
                
                # Find files to process
                files_to_delete = existing_file_ids - drive_file_ids
                files_to_process = []
                
                for file in drive_files:
                    file_id = file['id']
                    current_hash = drive_file_hashes.get(file_id)
                    if current_hash is None:
                        continue
                    
                    existing_hash = existing_hashes.get(file_id)
                    
                    # Process if new file or file has changed
                    if existing_hash != current_hash:
                        files_to_process.append(file)
                        status = "NEW" if existing_hash is None else "CHANGED"
                        logger.info(f"  File to process: {file['name']} ({status})")
                
                logger.info(f"PHASE 4: Files to process: {len(files_to_process)}, Files to delete: {len(files_to_delete)}")
                
                # Delete removed files
                deleted_count = 0
                for file_id in files_to_delete:
                    if delete_file_by_drive_id(file_id):
                        deleted_count += 1
                
                if deleted_count > 0:
                    logger.info(f"PHASE 4: ✓ Deleted {deleted_count} files from Supabase")
                
                # Process new/updated files
                if not files_to_process:
                    logger.info("PHASE 5: No files to process, sync complete!")
                    self.last_sync_time = datetime.now()
                    self.save_sync_config()
                    if not IS_WINDOWS:
                        signal.alarm(0)  # Cancel timeout
                    return True
                
                logger.info(f"PHASE 5: Processing {len(files_to_process)} files...")
                
                total_processed_docs = 0
                for idx, file_info in enumerate(files_to_process, 1):
                    logger.info(f"📄 PROCESSING FILE {idx}/{len(files_to_process)}: {file_info['name']}")
                    file_start_time = time.time()
                    
                    try:
                        # Delete existing records for this file first
                        logger.info(f"  Step 1: Deleting existing records for file...")
                        delete_file_by_drive_id(file_info['id'])
                        
                        # Extract text
                        logger.info(f"  Step 2: Extracting text...")
                        extract_start = time.time()
                        text = self.drive_loader.extract_text_from_file(file_info)
                        extract_time = time.time() - extract_start
                        
                        if not text:
                            logger.warning(f"  ⚠️  No text extracted from {file_info['name']}")
                            continue
                        
                        logger.info(f"  Step 2: ✓ Extracted {len(text):,} characters in {extract_time:.2f}s")
                        
                        # Insert raw file with hash
                        logger.info(f"  Step 3: Inserting raw file into Supabase...")
                        file_hash = drive_file_hashes[file_info['id']]
                        supabase_file_id = insert_raw_file(
                            file_info['id'], 
                            file_info['name'], 
                            text,
                            file_hash,
                            file_info.get('mimeType'),
                            file_info.get('size')
                        )
                        
                        if not supabase_file_id:
                            logger.error(f"  ❌ Failed to insert raw file")
                            continue
                            
                        logger.info(f"  Step 3: ✓ Raw file inserted (ID: {supabase_file_id})")
                        
                        # Create chunks
                        logger.info(f"  Step 4: Creating text chunks...")
                        chunk_start = time.time()
                        chunks = self.chunk_text_langchain(text)
                        chunk_time = time.time() - chunk_start
                        logger.info(f"  Step 4: ✓ Created {len(chunks)} chunks in {chunk_time:.2f}s")
                        
                        # Insert chunks
                        logger.info(f"  Step 5: Inserting chunks into Supabase...")
                        chunks_success = insert_processed_chunks(
                            file_info['id'], 
                            file_info['name'], 
                            chunks
                        )
                        
                        if not chunks_success:
                            logger.error(f"  ❌ Failed to insert chunks")
                            continue
                            
                        logger.info(f"  Step 5: ✓ {len(chunks)} chunks inserted successfully")
                        total_processed_docs += len(chunks)
                        
                        file_total_time = time.time() - file_start_time
                        logger.info(f"📄 ✅ COMPLETED {file_info['name']} in {file_total_time:.2f}s total")
                        
                    except Exception as e:
                        logger.error(f"📄 ❌ ERROR processing {file_info['name']}: {e}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        continue
                
                # Cancel the timeout if non-Windows
                if not IS_WINDOWS:
                    signal.alarm(0)
                
                # Save sync time
                self.last_sync_time = datetime.now()
                self.save_sync_config()
                
                # Final summary
                logger.info(f"🎉 FULL SYNC COMPLETED SUCCESSFULLY!")
                logger.info(f"   📊 Statistics:")
                logger.info(f"   • Files processed: {len(files_to_process)}")
                logger.info(f"   • Documents added: {total_processed_docs}")
                logger.info(f"   • Files deleted: {deleted_count}")
                
                # Final Supabase check
                logger.info("🔍 FINAL: Final Supabase table counts")
                get_table_counts()
                
                return True
                
            except Exception as e:
                if not IS_WINDOWS:
                    signal.alarm(0)  # Cancel timeout
                logger.error(f"❌ FATAL ERROR during full sync: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return False

    def chunk_text_langchain(self, text: str) -> List[str]:
        """Use LangChain's text splitter for chunking"""
        if self.use_langchain and hasattr(self, 'text_splitter'):
            return self.text_splitter.split_text(text)
        else:
            # Fallback to drive_loader's chunking method
            return self.drive_loader.chunk_text(text)

    def perform_incremental_sync(self) -> bool:
        """Perform incremental sync based on modification time"""
        if not self.last_sync_time:
            logger.info("No previous sync time found, performing full sync")
            return self.perform_full_sync()
        
        # For now, just redirect to full sync for testing
        logger.info("Redirecting to full sync for debugging...")
        return self.perform_full_sync()

def main():
    parser = argparse.ArgumentParser(description="Google Drive to Supabase Sync Tool")
    
    # Make arguments optional and use environment variables as defaults
    parser.add_argument("--folder-id", 
                       default=os.getenv("GOOGLE_DRIVE_FOLDER_ID"), 
                       help="Google Drive folder ID (default: from GOOGLE_DRIVE_FOLDER_ID env var)")
    parser.add_argument("--credentials-path", 
                       default="credentials.json", 
                       help="Path to Google credentials JSON")
    parser.add_argument("--mode", 
                       choices=["full", "incremental", "status", "cleanup", "schedule", "search"], 
                       default="incremental", 
                       help="Sync mode")
    parser.add_argument("--use-langchain", 
                       action="store_true", 
                       help="Enable LangChain text splitting features")
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.folder_id:
        logger.error("Missing required argument: folder-id (or GOOGLE_DRIVE_FOLDER_ID env var)")
        return 1
    
    # Check if LangChain is requested but not available
    if args.use_langchain and not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain requested but not available. Install with: pip install langchain")
        args.use_langchain = False
    
    # Display configuration
    logger.info(f"🔧 Configuration:")
    logger.info(f"   Folder ID: {args.folder_id}")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   LangChain: {args.use_langchain and LANGCHAIN_AVAILABLE}")
    
    # Initialize sync manager
    sync_manager = DriveSupabaseSync(
        folder_id=args.folder_id,
        credentials_path=args.credentials_path,
        use_langchain=args.use_langchain
    )
    
    # Initialize the system
    if not sync_manager.initialize():
        logger.error("❌ Failed to initialize sync system")
        return 1
    
    # Execute based on mode
    if args.mode in ["full", "incremental"]:
        logger.info(f"🚀 Starting {args.mode} synchronization...")
        success = sync_manager.perform_incremental_sync() if args.mode == "incremental" else sync_manager.perform_full_sync()
        if success:
            logger.info(f"✅ {args.mode.title()} sync completed successfully")
            return 0
        else:
            logger.error(f"❌ {args.mode.title()} sync failed")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())