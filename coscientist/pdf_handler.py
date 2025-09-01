"""
PDF Handler for Coscientist - Enhanced PDF processing with fallback mechanisms
"""

import os
import hashlib
import logging
import requests
import arxiv
import pdfplumber
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse, urljoin
from pathlib import Path
import time
import re
import tempfile
import atexit
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PDFProcessingConfig:
    """Configuration for PDF processing"""
    cache_dir: str = ""  # Empty means use temp directory
    download_timeout: int = 30
    max_size_mb: int = 50
    fallback_to_abstract: bool = True
    max_pages: int = 50
    chunk_size: int = 2000
    overlap_size: int = 200
    persistent_cache: bool = False
    cleanup_on_exit: bool = True
    max_age_hours: int = 24

class PDFHandler:
    """Enhanced PDF handler with multiple fallback mechanisms"""
    
    def __init__(self, config: PDFProcessingConfig = None):
        self.config = config or PDFProcessingConfig()
        self._temp_dir_created = False
        
        # Setup cache directory
        if self.config.cache_dir and self.config.persistent_cache:
            # Use specified persistent directory
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        else:
            # Create temporary directory
            self.cache_dir = Path(tempfile.mkdtemp(prefix="coscientist_pdf_"))
            self._temp_dir_created = True
            logger.info(f"Created temporary PDF cache: {self.cache_dir}")
            
            # Register cleanup on exit
            if self.config.cleanup_on_exit:
                atexit.register(self._cleanup_cache_on_exit)
        
        # Clean up old files if configured
        self._cleanup_old_files()
        
        # Session for HTTP requests with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate'
        })
    
    def _get_cache_path(self, url: str) -> Path:
        """Generate cache file path from URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.pdf"
    
    def _get_text_cache_path(self, url: str) -> Path:
        """Generate text cache file path from URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}_text.txt"
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL likely points to a PDF"""
        parsed = urlparse(url.lower())
        return (
            parsed.path.endswith('.pdf') or 
            'pdf' in parsed.query or
            'application/pdf' in url.lower()
        )
    
    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from URL if present"""
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        match = re.search(doi_pattern, url)
        return match.group(0) if match else None
    
    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL"""
        arxiv_pattern = r'arxiv\.org/(?:abs/|pdf/)?(\d{4}\.\d{4,5})'
        match = re.search(arxiv_pattern, url.lower())
        return match.group(1) if match else None
    
    def _download_pdf(self, url: str) -> Optional[bytes]:
        """Download PDF with proper error handling"""
        try:
            logger.info(f"Attempting to download PDF from: {url}")
            
            response = self.session.get(
                url, 
                timeout=self.config.download_timeout,
                allow_redirects=True,
                stream=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not self._is_pdf_url(url):
                logger.warning(f"Content type '{content_type}' doesn't appear to be PDF")
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.config.max_size_mb:
                    logger.warning(f"PDF too large: {size_mb:.1f}MB > {self.config.max_size_mb}MB")
                    return None
            
            # Download in chunks
            pdf_data = b''
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                pdf_data += chunk
                total_size += len(chunk)
                
                # Check size limit while downloading
                if total_size > self.config.max_size_mb * 1024 * 1024:
                    logger.warning(f"PDF download exceeded size limit during transfer")
                    return None
            
            logger.info(f"Successfully downloaded PDF: {len(pdf_data)} bytes")
            return pdf_data
            
        except requests.RequestException as e:
            logger.warning(f"Failed to download PDF from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF from {url}: {e}")
            return None
    
    def _extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF bytes using pdfplumber"""
        try:
            import io
            
            pdf_buffer = io.BytesIO(pdf_data)
            text_content = []
            
            with pdfplumber.open(pdf_buffer) as pdf:
                pages_to_process = min(len(pdf.pages), self.config.max_pages)
                logger.info(f"Processing {pages_to_process} pages from PDF")
                
                for i, page in enumerate(pdf.pages[:pages_to_process]):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text.strip())
                        
                        # Add page break marker
                        if i < pages_to_process - 1:
                            text_content.append("\n--- PAGE BREAK ---\n")
                            
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {i+1}: {e}")
                        continue
            
            full_text = '\n'.join(text_content)
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            return ""
    
    def _try_arxiv_fallback(self, url: str) -> Optional[str]:
        """Try to get paper from arXiv as fallback"""
        try:
            arxiv_id = self._extract_arxiv_id(url)
            if not arxiv_id:
                return None
            
            logger.info(f"Trying arXiv fallback for ID: {arxiv_id}")
            
            # Search arXiv
            search = arxiv.Search(id_list=[arxiv_id])
            papers = list(search.results())
            
            if not papers:
                return None
            
            paper = papers[0]
            
            # Download PDF from arXiv
            pdf_url = paper.pdf_url
            pdf_data = self._download_pdf(pdf_url)
            
            if pdf_data:
                text = self._extract_text_from_pdf(pdf_data)
                if text:
                    # Cache the result
                    cache_path = self._get_cache_path(url)
                    with open(cache_path, 'wb') as f:
                        f.write(pdf_data)
                    
                    text_cache_path = self._get_text_cache_path(url)
                    with open(text_cache_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    logger.info(f"Successfully retrieved paper via arXiv fallback")
                    return text
            
            # Fallback to abstract if full text fails
            if self.config.fallback_to_abstract:
                abstract_text = f"Title: {paper.title}\n\nAbstract: {paper.summary}\n\nAuthors: {', '.join(str(author) for author in paper.authors)}"
                logger.info("Using arXiv abstract as fallback")
                return abstract_text
                
        except Exception as e:
            logger.warning(f"arXiv fallback failed: {e}")
        
        return None
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > self.config.chunk_size and current_chunk:
                # Add current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-self.config.overlap_size//10:]  # Rough word count
                current_chunk = overlap_words + [word]
                current_size = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_size += word_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def process_pdf_url(self, url: str) -> Tuple[str, List[str], str]:
        """
        Main method to process a PDF URL
        Returns: (full_text, text_chunks, title)
        """
        try:
            # Check text cache first
            text_cache_path = self._get_text_cache_path(url)
            if text_cache_path.exists():
                logger.info(f"Loading cached text for: {url}")
                with open(text_cache_path, 'r', encoding='utf-8') as f:
                    cached_text = f.read()
                    chunks = self._chunk_text(cached_text)
                    title = cached_text.split('\n')[0] if cached_text else "Unknown Title"
                    return cached_text, chunks, title
            
            # Check PDF cache
            cache_path = self._get_cache_path(url)
            pdf_data = None
            
            if cache_path.exists():
                logger.info(f"Loading cached PDF for: {url}")
                with open(cache_path, 'rb') as f:
                    pdf_data = f.read()
            else:
                # Try to download PDF
                pdf_data = self._download_pdf(url)
                
                # If direct download failed, try arXiv
                if not pdf_data:
                    fallback_text = self._try_arxiv_fallback(url)
                    if fallback_text:
                        chunks = self._chunk_text(fallback_text)
                        title = fallback_text.split('\n')[0] if fallback_text else "Unknown Title"
                        return fallback_text, chunks, title
                
                # Cache downloaded PDF
                if pdf_data:
                    with open(cache_path, 'wb') as f:
                        f.write(pdf_data)
            
            if not pdf_data:
                logger.warning(f"Could not obtain PDF data for: {url}")
                return "", [], "Failed to load PDF"
            
            # Extract text
            text = self._extract_text_from_pdf(pdf_data)
            
            if not text:
                logger.warning(f"Could not extract text from PDF: {url}")
                return "", [], "Failed to extract text"
            
            # Cache extracted text
            with open(text_cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Create chunks
            chunks = self._chunk_text(text)
            
            # Extract title (first meaningful line)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            title = lines[0] if lines else "Unknown Title"
            
            logger.info(f"Successfully processed PDF: {len(text)} chars, {len(chunks)} chunks")
            return text, chunks, title
            
        except Exception as e:
            logger.error(f"Error processing PDF URL {url}: {e}")
            return "", [], "Error processing PDF"
    
    def _cleanup_cache_on_exit(self):
        """Cleanup cache directory on process exit"""
        if self._temp_dir_created and self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cleaned up temporary PDF cache: {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp cache on exit: {e}")
    
    def _cleanup_old_files(self):
        """Clean up old cache files based on max_age_hours"""
        if not self.cache_dir.exists() or self.config.max_age_hours <= 0:
            return
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config.max_age_hours)
            removed_count = 0
            
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                        removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old cache files older than {self.config.max_age_hours} hours")
        except Exception as e:
            logger.warning(f"Error during old file cleanup: {e}")
    
    def clear_cache(self):
        """Clear PDF cache"""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                logger.info("PDF cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.cache_dir.exists():
                return {"files": 0, "size_mb": 0}
            
            files = list(self.cache_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            return {
                "files": len(files),
                "size_mb": total_size / (1024 * 1024),
                "pdf_files": len(list(self.cache_dir.glob("*.pdf"))),
                "text_files": len(list(self.cache_dir.glob("*_text.txt")))
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}


# Global PDF handler instance
_pdf_handler = None

def get_pdf_handler(config: PDFProcessingConfig = None) -> PDFHandler:
    """Get or create global PDF handler instance"""
    global _pdf_handler
    if _pdf_handler is None:
        _pdf_handler = PDFHandler(config)
    return _pdf_handler


def process_pdf_url(url: str) -> Tuple[str, List[str], str]:
    """Convenience function to process a PDF URL"""
    handler = get_pdf_handler()
    return handler.process_pdf_url(url)