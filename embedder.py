"""
Embedding Module
----------------
This module handles text embedding generation using OpenAI's API.
Embeddings convert text into vector representations for similarity search.
"""

import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================================
# OFFLINE PHASE: Document Chunk Embedding
# ============================================================================

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of text chunks.

    Each chunk is converted into a dense vector representation that captures
    its semantic meaning, enabling similarity-based retrieval.

    Args:
        chunks (List[str]): List of text chunks to embed

    Returns:
        List[List[float]]: List of embedding vectors, one per chunk
    """
    embeddings = []

    # Generate embedding for each chunk
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model=EMBEDDING_MODEL
        )

        # Extract embedding vector from response
        embeddings.append(response.data[0].embedding)

    return embeddings


# ============================================================================
# ONLINE PHASE: User Query Embedding
# ============================================================================

def embed_user_query(query: str) -> List[float]:
    """
    Generates an embedding for a user's query.

    The query embedding is used to find similar document chunks via
    vector similarity search.

    Args:
        query (str): User's question or search query

    Returns:
        List[float]: Embedding vector for the query
    """
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )

    return response.data[0].embedding