"""
Data Processor - Offline Pipeline
----------------------------------
This script orchestrates the offline phase of the RAG system:
1. Read PDF document
2. Split text into chunks
3. Generate embeddings
4. Store in vector database

Run this script to index a new document before querying.
"""

from typing import List
from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_pinecone

# ============================================================================
# Configuration
# ============================================================================

# Path to the PDF document to be processed
pdf_path = "./data/sample.pdf"


# ============================================================================
# Main Processing Pipeline
# ============================================================================

def run():
    """
    Executes the complete offline RAG pipeline.

    Pipeline steps:
    1. Extract text from PDF pages
    2. Split text into overlapping chunks
    3. Generate embeddings for each chunk
    4. Store chunks and embeddings in Pinecone
    """
    # Step 1: Read PDF and extract text from all pages
    pages = read_pdf(pdf_path)

    # Step 2: Split extracted text into manageable chunks
    chunks = chunk_pages(pages)

    # Step 3: Generate embedding vectors for all chunks
    embeddings = embed_chunks(chunks)

    # Step 4: Store chunks with embeddings in vector database
    store_in_pinecone(chunks, embeddings)


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    run()