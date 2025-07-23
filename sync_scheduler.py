#!/usr/bin/env python3
"""
Sync scheduler for RAG application.
Runs sync.py periodically based on SYNC_INTERVAL environment variable.
"""

import schedule
import time
import subprocess
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logging():
    """Setup logging with proper path handling for Windows/Linux"""
    # Get log level
    log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO'))
    
    # Create logs directory if it doesn't exist (Windows-compatible)
    log_dir = Path("logs")
    
    # Handle directory creation more robustly
    if log_dir.exists():
        if not log_dir.is_dir():
            # If 'logs' exists but is a file, remove it and create directory
            try:
                log_dir.unlink()  # Remove the file
                log_dir.mkdir()   # Create directory
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not replace logs file with directory: {e}")
                log_dir = Path(".")  # Fallback to current directory
    else:
        # Directory doesn't exist, create it
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create logs directory: {e}")
            log_dir = Path(".")  # Fallback to current directory
    
    # Set log file path (Windows compatible)
    log_file = log_dir / "rag_sync.log"
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

def run_sync():
    """Run the sync process."""
    logger.info('Starting scheduled sync...')
    try:
        # Use current directory instead of /app
        current_dir = Path.cwd()
        
        # Check if sync.py exists
        sync_script = current_dir / "sync.py"
        if not sync_script.exists():
            logger.error(f"sync.py not found in {current_dir}")
            return
        
        result = subprocess.run(
            [sys.executable, 'sync.py'],  # Use current Python interpreter
            capture_output=True, 
            text=True, 
            cwd=str(current_dir),
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            logger.info('Sync completed successfully')
            if result.stdout.strip():
                logger.info(f'Sync output: {result.stdout.strip()}')
        else:
            logger.error(f'Sync failed with return code {result.returncode}')
            if result.stderr.strip():
                logger.error(f'Sync stderr: {result.stderr.strip()}')
            if result.stdout.strip():
                logger.error(f'Sync stdout: {result.stdout.strip()}')
                
    except subprocess.TimeoutExpired:
        logger.error('Sync process timed out after 30 minutes')
    except Exception as e:
        logger.error(f'Sync error: {e}')

def main():
    """Main scheduler loop."""
    logger.info("=== RAG Sync Scheduler Starting ===")
    
    # Get sync interval from environment
    interval_seconds = int(os.getenv('SYNC_INTERVAL', 3600))
    logger.info(f"Sync interval set to {interval_seconds} seconds ({interval_seconds/3600:.1f} hours)")
    
    # Check if sync.py exists before starting
    if not Path("sync.py").exists():
        logger.error("sync.py not found in current directory. Make sure you're in the correct directory.")
        return 1
    
    # Schedule sync
    schedule.every(interval_seconds).seconds.do(run_sync)
    
    # Run initial sync
    logger.info('Running initial sync...')
    run_sync()
    
    # Start scheduler
    logger.info(f'Scheduler started. Running sync every {interval_seconds} seconds')
    logger.info('Press Ctrl+C to stop the scheduler')
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info('Scheduler stopped by user')
    except Exception as e:
        logger.error(f'Scheduler error: {e}')
    
    return 0

if __name__ == "__main__":
    exit(main())