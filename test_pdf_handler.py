#!/usr/bin/env python3
"""
Test script for PDF handler functionality
"""

import sys
import os
sys.path.append('.')

from coscientist.pdf_handler import PDFHandler, PDFProcessingConfig, process_pdf_url
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_pdf_handler():
    """Test PDF handler with various URLs"""
    
    # Test URLs (mix of accessible and problematic ones)
    test_urls = [
        # arXiv PDF (should work with fallback)
        "https://arxiv.org/pdf/2301.07041.pdf",
        # Another arXiv
        "https://arxiv.org/pdf/2023.12345.pdf",  # Likely 404 for fallback testing
        # Academic PDF (might be blocked)
        "https://www.researchgate.net/profile/test/publication/test.pdf",
    ]
    
    print("=== Testing PDF Handler ===")
    
    # Create handler with custom config
    config = PDFProcessingConfig(
        cache_dir="./test_pdf_cache",
        max_size_mb=10,
        max_pages=5,  # Limit for testing
        fallback_to_abstract=True
    )
    
    handler = PDFHandler(config)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n--- Test {i}: {url} ---")
        
        try:
            text, chunks, title = handler.process_pdf_url(url)
            
            print(f"✓ Title: {title}")
            print(f"✓ Text length: {len(text)} characters")
            print(f"✓ Number of chunks: {len(chunks)}")
            
            if text:
                # Show first 200 chars
                preview = text[:200].replace('\n', ' ').strip()
                print(f"✓ Preview: {preview}...")
                
            if chunks:
                print(f"✓ First chunk length: {len(chunks[0])}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Test cache stats
    print(f"\n--- Cache Statistics ---")
    stats = handler.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    # Test convenience function
    print(f"\n--- Testing Convenience Function ---")
    try:
        text, chunks, title = process_pdf_url("https://arxiv.org/pdf/2301.07041.pdf")
        print(f"✓ Convenience function works - Title: {title}, Text length: {len(text)}")
    except Exception as e:
        print(f"✗ Convenience function error: {e}")

if __name__ == "__main__":
    test_pdf_handler()