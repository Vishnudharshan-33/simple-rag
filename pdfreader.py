"""
PDF Reader Module
-----------------
This module handles PDF file reading and text extraction.
Used in the offline phase of the RAG pipeline.
"""

import os
from pypdf import PdfReader


# ============================================================================
# OFFLINE PHASE: PDF Text Extraction
# ============================================================================

def read_pdf(pdf_path):
    """
    Reads a PDF file and extracts text from all pages.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        list: List of strings, each containing text from one page

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    reader = PdfReader(pdf_path)
    pages = [page.extract_text() for page in reader.pages]

    return pages

# ============================================================================
# ONLINE PHASE: Not applicable for this module
# ============================================================================