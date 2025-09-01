#!/usr/bin/env python3
"""
Test script for temporary PDF handler functionality
"""

import sys
import os
import time
sys.path.append('.')

from coscientist.pdf_handler import PDFHandler, PDFProcessingConfig, process_pdf_url
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_temp_pdf_handler():
    """Test PDF handler with temporary caching"""
    
    print("=== Testing Temporary PDF Handler ===")
    
    # Test 1: Default temporary configuration
    print("\n--- Test 1: Default Temporary Cache ---")
    handler1 = PDFHandler()
    print(f"Cache directory: {handler1.cache_dir}")
    print(f"Is temporary: {handler1._temp_dir_created}")
    print(f"Cache exists: {handler1.cache_dir.exists()}")
    
    # Test with a simple arXiv PDF
    text, chunks, title = handler1.process_pdf_url("https://arxiv.org/pdf/2301.07041.pdf")
    print(f"Processed PDF - Title: {title[:50]}...")
    print(f"Cache stats after processing: {handler1.get_cache_stats()}")
    
    # Test 2: Persistent cache configuration
    print("\n--- Test 2: Persistent Cache Configuration ---")
    persistent_config = PDFProcessingConfig(
        cache_dir="./test_persistent_cache",
        persistent_cache=True,
        cleanup_on_exit=False
    )
    handler2 = PDFHandler(persistent_config)
    print(f"Cache directory: {handler2.cache_dir}")
    print(f"Is temporary: {handler2._temp_dir_created}")
    
    # Test 3: Multiple handlers with temp caches
    print("\n--- Test 3: Multiple Temporary Handlers ---")
    handlers = []
    for i in range(3):
        h = PDFHandler()
        handlers.append(h)
        print(f"Handler {i+1} cache: {h.cache_dir}")
    
    # Test 4: Cache cleanup demonstration
    print("\n--- Test 4: Cache Cleanup Test ---")
    temp_handler = PDFHandler()
    cache_path = temp_handler.cache_dir
    print(f"Temporary cache created at: {cache_path}")
    print(f"Cache exists before cleanup: {cache_path.exists()}")
    
    # Manual cleanup
    temp_handler._cleanup_cache_on_exit()
    print(f"Cache exists after cleanup: {cache_path.exists()}")
    
    print("\n--- Cache Verification ---")
    # Verify no persistent caches remain
    import glob
    persistent_caches = glob.glob("**/pdf_cache*", recursive=True)
    print(f"Persistent cache directories found: {persistent_caches}")
    
    temp_caches = glob.glob("/tmp/coscientist_pdf_*")
    print(f"Temporary cache directories: {len(temp_caches)} found")

if __name__ == "__main__":
    test_temp_pdf_handler()