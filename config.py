"""
Configuration file for Legal Contract Analyzer
Contains all constants and configuration values
"""

import os
from typing import Optional

# ===== Embedding Configuration =====
EMBEDDING_DIM = 384  # Standardized dimension for all-MiniLM-L6-v2
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXT_CHUNK = 8000  # Max characters per chunk
CHUNK_OVERLAP = 500    # Overlap between chunks

# ===== API Configuration =====
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 8000
TEMPERATURE = 0.1

# ===== Retry Configuration =====
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
RETRY_BACKOFF = 2  # exponential backoff multiplier

# ===== API Timeouts =====
HF_API_TIMEOUT = 30  # seconds
GROQ_API_TIMEOUT = 60  # seconds

# ===== Similarity Search Configuration =====
SIMILARITY_THRESHOLD = 0.85  # High similarity threshold for contract matching
PRECEDENT_SIMILARITY_THRESHOLD = 0.75  # Threshold for precedent matching
DEFAULT_TOP_K = 5  # Default number of results for similarity search

# ===== Environment Variables =====
def get_env_var(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional validation"""
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Required environment variable {name} is not set")
    return value

# API Keys and Tokens
GROQ_API_KEY = get_env_var("GROQ_API_KEY", required=True)
HF_TOKEN = get_env_var("HF_TOKEN", required=True)

# Neo4j Configuration
NEO4J_URI = get_env_var("NEO4J_URI", required=True)
NEO4J_USERNAME = get_env_var("NEO4J_USERNAME", required=True)
NEO4J_PASSWORD = get_env_var("NEO4J_PASSWORD", required=True)

# Weaviate Configuration (Optional)
WEAVIATE_URL = get_env_var("WEAVIATE_URL")
WEAVIATE_API_KEY = get_env_var("WEAVIATE_API_KEY")

# ===== Neo4j URI Fix for Aura =====
def get_neo4j_uri() -> str:
    """Get Neo4j URI with Aura compatibility fix"""
    uri = NEO4J_URI
    # Fix for Neo4j Aura: convert neo4j+s:// to neo4j+ssc:// (uses system cert store)
    if "neo4j+s://" in uri and "neo4j+ssc://" not in uri:
        uri = uri.replace("neo4j+s://", "neo4j+ssc://")
    return uri

# ===== HuggingFace API URL =====
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_EMBED_MODEL}"

# ===== Database Connection Retries =====
MAX_CONNECTION_RETRIES = 3


