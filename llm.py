"""
LLM Query Module
----------------
This module handles question answering using OpenAI's chat models.
The LLM generates answers based on retrieved document context.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# ONLINE PHASE: Generate Answer from Context
# ============================================================================

def query_llm_with_context(query: str, context: str):
    """
    Generates an answer to a user's question using retrieved context.

    The LLM is instructed to answer strictly based on the provided context,
    preventing hallucinations and ensuring factual accuracy.

    Args:
        query (str): User's question
        context (str): Retrieved document chunks that may contain the answer

    Returns:
        str: Generated answer based on the context
    """
    # System prompt that constrains the LLM to use only provided context
    system_context = """You are a precise, helpful document Q&A assistant.
Answer the user's question ONLY using the provided context.
If the information isn't in the context, say:
"I don't have enough information in the provided document to answer this."

Be concise, accurate, and speak naturally."""

    # Generate response using chat completion
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
        ],
        temperature=0.4  # Lower temperature for more focused, deterministic responses
    )

    return response.choices[0].message.content