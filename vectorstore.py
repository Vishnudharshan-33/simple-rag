"""
Vector Store Module
-------------------
This module handles storage and retrieval of embeddings using Pinecone.
Pinecone provides fast similarity search over high-dimensional vectors.
"""

import os
from typing import List
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client and index
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME"))


# ============================================================================
# OFFLINE PHASE: Store Document Embeddings
# ============================================================================

def store_in_pinecone(chunks: List[str], embeddings: List[List[float]], namespace: str = ""):
    """
    Stores text chunks and their embeddings in Pinecone vector database.

    Each chunk is stored with its embedding vector and metadata (original text).
    Uploads are batched for efficiency.

    Args:
        chunks (List[str]): List of text chunks
        embeddings (List[List[float]]): Corresponding embedding vectors
        namespace (str): Optional namespace for organizing vectors (default: "")
    """
    vectors_to_upsert = []

    # Prepare vectors with metadata
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_data = {
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i
            }
        }
        vectors_to_upsert.append(vector_data)

    # Upload vectors in batches to avoid rate limits
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)


# ============================================================================
# ONLINE PHASE: Retrieve Similar Chunks
# ============================================================================

def search_in_pinecone(query_vector: List[float], top_k: int = 3, namespace: str = ""):
    """
    Searches for the most similar chunks to a query vector.

    Uses cosine similarity to find chunks whose embeddings are closest
    to the query embedding.

    Args:
        query_vector (List[float]): Embedding vector of the user's query
        top_k (int): Number of most similar chunks to retrieve (default: 3)
        namespace (str): Optional namespace to search within (default: "")

    Returns:
        List[str]: List of matched chunk texts, ordered by similarity
    """
    # Query Pinecone for similar vectors
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    # Extract text from matched chunks
    matched_chunks = []
    for match in results.matches:
        matched_chunks.append(match.metadata.get("text", " "))

    return matched_chunks