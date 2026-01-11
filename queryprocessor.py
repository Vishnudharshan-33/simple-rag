"""
Query Processor - Online Pipeline
----------------------------------
This script handles user queries in the online phase of the RAG system:
1. Embed user query
2. Retrieve similar chunks from vector database
3. Generate answer using LLM with retrieved context

Run this script to query the indexed document.
"""

from embedder import embed_user_query
from vectorstore import search_in_pinecone
from llm import query_llm_with_context


# ============================================================================
# Main Query Pipeline
# ============================================================================

def process_user_query(query: str):
    """
    Processes a user query and generates an answer.

    Pipeline steps:
    1. Convert query to embedding vector
    2. Find most similar document chunks via vector search
    3. Generate answer using LLM with retrieved chunks as context

    Args:
        query (str): User's question about the document
    """
    # Step 1: Generate embedding for the user's query
    query_vector = embed_user_query(query)

    # Step 2: Retrieve most similar chunks from Pinecone
    matched_chunks = search_in_pinecone(query_vector)

    # Step 3: Generate answer using LLM with retrieved context
    generated_response = query_llm_with_context(query, matched_chunks)

    # Display the answer
    print(generated_response)


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    # Example query - modify this to ask different questions
    user_query = "what is text formatting"
    process_user_query(user_query)