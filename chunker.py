"""
Text Chunking Module
--------------------
This module splits extracted text into smaller, overlapping chunks
for efficient embedding and retrieval.
"""

from typing import List


# ============================================================================
# OFFLINE PHASE: Text Chunking
# ============================================================================

def chunk_pages(pages: List[str], chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Splits page text into overlapping chunks of specified size.

    Overlapping chunks ensure that context isn't lost at chunk boundaries,
    improving retrieval accuracy.

    Args:
        pages (List[str]): List of text from each page
        chunk_size (int): Maximum size of each chunk in characters (default: 500)
        chunk_overlap (int): Number of overlapping characters between chunks (default: 100)

    Returns:
        List[str]: List of text chunks
    """
    # Combine all pages into a single text string
    full_text = " ".join(pages)
    text_length = len(full_text)
    chunks = []

    # Handle empty text
    if text_length == 0:
        return chunks

    # Create overlapping chunks using sliding window approach
    start = 0
    while start < text_length:
        # Calculate end position for current chunk
        end = min(start + chunk_size, text_length)

        # Extract and clean chunk
        chunk = full_text[start:end].strip()

        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)

        # Stop if we've reached the end
        if end >= text_length:
            break

        # Move start position, accounting for overlap
        start = end - chunk_overlap

    return chunks

# ============================================================================
# ONLINE PHASE: Not applicable for this module
# ============================================================================