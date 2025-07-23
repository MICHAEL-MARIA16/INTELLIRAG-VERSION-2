import os
import io
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
import csv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

import pypdf as PyPDF2
from docx import Document
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveLoader:
    def __init__(self, folder_id: str, credentials_path: str = "credentials.json"):
        self.folder_id = folder_id
        self.credentials_path = credentials_path
        self.service = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
        self.supported_types = {
            'application/pdf': self._extract_pdf_text,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_docx_text,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self._extract_excel_text,
            'text/plain': self._extract_text_file,
            'text/csv': self._extract_csv_text,  # Added CSV support
            'application/csv': self._extract_csv_text,  # Alternative CSV mime type
            'application/vnd.google-apps.document': self._extract_google_doc,
            'application/vnd.google-apps.spreadsheet': self._extract_google_sheet
        }
        
    def authenticate(self):
        """Authenticate using a service account"""
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
    
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=self.scopes
        )
    
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Successfully authenticated with Google Drive using service account")

        
    def get_files_in_folder(self) -> List[Dict[str, Any]]:
        """Get all files in the specified Google Drive folder"""
        if not self.service:
            self.authenticate()
            
        try:
            query = f"'{self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                fields="files(id,name,mimeType,modifiedTime,md5Checksum,size)"
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder")
            return files
            
        except HttpError as error:
            logger.error(f"Error fetching files: {error}")
            return []
    
    def download_file_content(self, file_id: str, mime_type: str) -> Optional[bytes]:
        """Download file content from Google Drive"""
        try:
            if mime_type.startswith('application/vnd.google-apps'):
                # Handle Google Docs, Sheets, etc.
                export_mime = {
                    'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                }.get(mime_type)
                
                if export_mime:
                    request = self.service.files().export_media(fileId=file_id, mimeType=export_mime)
                else:
                    return None
            else:
                request = self.service.files().get_media(fileId=file_id)
                
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            done = False
            
            while done is False:
                status, done = downloader.next_chunk()
                
            file_io.seek(0)
            return file_io.read()
            
        except HttpError as error:
            logger.error(f"Error downloading file {file_id}: {error}")
            return None
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            doc = Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def _extract_excel_text(self, content: bytes) -> str:
        """Extract text from Excel content"""
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=None)
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"Sheet: {sheet_name}\n"
                text += sheet_df.to_string() + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting Excel text: {e}")
            return ""
    
    def _extract_text_file(self, content: bytes) -> str:
        """Extract text from plain text file"""
        try:
            return content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def _extract_csv_text(self, content: bytes) -> str:
        """Extract text from CSV content with better error handling"""
        try:
            # Try different encodings
            text_content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    text_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
        
            if text_content is None:
                logger.error("Could not decode CSV content with any encoding")
                return ""
        
            # Try to detect delimiter
            sniffer = csv.Sniffer()
            try:
                sample = text_content[:1024]  # Use first 1KB for detection
                delimiter = sniffer.sniff(sample).delimiter
            except:
                delimiter = ','  # Default fallback
        
            # Parse CSV
            csv_reader = csv.reader(io.StringIO(text_content), delimiter=delimiter)
            rows = list(csv_reader)
        
            if not rows:
                return "Empty CSV file"
        
            # Format CSV data as readable text
            text_lines = []
        
            # Add file info
            text_lines.append(f"CSV Data Analysis:")
            text_lines.append(f"Total rows: {len(rows)}")
        
            # Add headers if present
            if rows:
                headers = rows[0]
                text_lines.append(f"Columns ({len(headers)}): " + " | ".join(headers))
                text_lines.append("-" * 60)
        
            # Add sample data rows (limit to prevent huge documents)
            sample_rows = min(20, len(rows) - 1)  # Show max 20 data rows
        
            for i, row in enumerate(rows[1:sample_rows + 1], 1):
                if len(row) > 0:  # Skip empty rows
                    row_text = " | ".join(str(cell).strip() for cell in row)
                    text_lines.append(f"Row {i}: {row_text}")
        
            if len(rows) > sample_rows + 1:
                remaining = len(rows) - sample_rows - 1
                text_lines.append(f"... and {remaining} more rows")
        
            # Add basic statistics for numeric columns
            if len(rows) > 1:
                text_lines.append(f"\nData Summary:")
                text_lines.append(f"• Total data rows: {len(rows) - 1}")
                text_lines.append(f"• Columns: {len(headers) if rows else 0}")
            
                # Try to identify numeric columns
                numeric_cols = []
                if len(rows) > 1:
                    for col_idx, header in enumerate(headers):
                        try:
                            # Check first few non-empty values
                            sample_values = []
                            for row in rows[1:min(6, len(rows))]:
                                if col_idx < len(row) and row[col_idx].strip():
                                    sample_values.append(row[col_idx].strip())
                        
                            if sample_values:
                                # Try to convert to float
                                numeric_values = []
                                for val in sample_values:
                                    try:
                                        numeric_values.append(float(val))
                                    except:
                                        break
                            
                                if len(numeric_values) == len(sample_values):
                                    numeric_cols.append(header)
                        except:
                            continue
            
                if numeric_cols:
                    text_lines.append(f"• Numeric columns detected: {', '.join(numeric_cols)}")
        
            return "\n".join(text_lines)
        
        except Exception as e:
            logger.error(f"Error extracting CSV text: {e}")
            return f"Error processing CSV file: {str(e)}"
    
    def _extract_google_doc(self, content: bytes) -> str:
        """Extract text from Google Doc (exported as DOCX)"""
        return self._extract_docx_text(content)
    
    def _extract_google_sheet(self, content: bytes) -> str:
        """Extract text from Google Sheet (exported as Excel)"""
        return self._extract_excel_text(content)
    
    def extract_text_from_file(self, file_info: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a file"""
        file_id = file_info['id']
        mime_type = file_info['mimeType']
        file_name = file_info['name']
        
        # Handle CSV files by extension if mime type is not detected correctly
        if file_name.lower().endswith('.csv') and mime_type not in ['text/csv', 'application/csv']:
            mime_type = 'text/csv'
        
        if mime_type not in self.supported_types:
            logger.warning(f"Unsupported file type: {mime_type} for file {file_name}")
            return None
        
        logger.info(f"Processing file: {file_name} (type: {mime_type})")
        content = self.download_file_content(file_id, mime_type)
        
        if content is None:
            logger.error(f"Failed to download content for file: {file_name}")
            return None
        
        extractor = self.supported_types[mime_type]
        text = extractor(content)
        
        if not text.strip():
            logger.warning(f"No text extracted from file: {file_name}")
            return None
            
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunks.append(text[start:end].strip())
            start = max(start + chunk_size - overlap, end)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        return self.embedding_model.encode(texts)
    
    def get_file_hash(self, file_info: Dict[str, Any]) -> str:
        """Generate a hash for file identification"""
        file_data = f"{file_info['id']}_{file_info.get('modifiedTime', '')}"
        return hashlib.md5(file_data.encode()).hexdigest()
    
    def process_all_files(self) -> List[Dict[str, Any]]:
        """Process all files in the folder and return document chunks with metadata"""
        files = self.get_files_in_folder()
        processed_documents = []
        
        for file_info in files:
            try:
                text = self.extract_text_from_file(file_info)
                if not text:
                    continue
                
                chunks = self.chunk_text(text)
                embeddings = self.generate_embeddings(chunks)
                file_hash = self.get_file_hash(file_info)
                
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    doc = {
                        'id': f"{file_info['id']}_{i}",
                        'file_id': file_info['id'],
                        'file_name': file_info['name'],
                        'file_hash': file_hash,
                        'chunk_index': i,
                        'text': chunk,
                        'embedding': embedding,
                        'metadata': {
                            'file_name': file_info['name'],
                            'mime_type': file_info['mimeType'],
                            'modified_time': file_info.get('modifiedTime'),
                            'file_size': file_info.get('size'),
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        }
                    }
                    processed_documents.append(doc)
                    
                logger.info(f"Processed {file_info['name']}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing file {file_info['name']}: {e}")
                continue
        
        return processed_documents