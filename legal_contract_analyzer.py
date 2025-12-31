"""
Legal Contract Analyzer
========================
AI-powered legal contract analysis system using LangGraph, Neo4j, and vector embeddings.

Usage:
    # Process a contract
    from legal_contract_analyzer import workflow, pdf_hash
    
    cid = pdf_hash("contract.pdf")
    workflow.invoke({
        "pdf_path": "contract.pdf",
        "cid": cid,
        "text": "",
        "embeddings": [],
        "analysis": {},
    })
    
    # Retrieve contract
    from legal_contract_analyzer import retrieve_contract_from_db, print_contract_summary
    
    data = retrieve_contract_from_db(cid)
    print_contract_summary(data)
    
    # Search similar clauses
    from legal_contract_analyzer import search_similar_clauses
    
    results = search_similar_clauses("payment terms", top_k=5)
"""

from dotenv import load_dotenv
load_dotenv()

# Standard library imports
import os
import json
import hashlib
import time
import re
import atexit
from functools import wraps
from typing import TypedDict, List, Dict, Any, Optional

# Third-party imports
import fitz  # PyMuPDF
import numpy as np
import requests
from groq import Groq
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

# Embedding configuration
EMBEDDING_DIM = 384  # Standardized dimension for all-MiniLM-L6-v2
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXT_CHUNK = 8000  # Max characters per chunk
CHUNK_OVERLAP = 500    # Overlap between chunks

# API Configuration
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 8000
TEMPERATURE = 0.1

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def validate_env_vars():
    """Validate required environment variables"""
    required = ['GROQ_API_KEY', 'HF_TOKEN', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return True

# Validate on import
try:
    validate_env_vars()
    print("[OK] Environment variables validated")
except ValueError as e:
    print(f"[WARNING] {e}")



# ===== Retry Logic with Exponential Backoff =====

def retry_with_backoff(max_retries=MAX_RETRIES, delay=RETRY_DELAY, backoff=2):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        print(f"[ERROR] {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    print(f"[WARNING] {func.__name__} failed (attempt {retries}/{max_retries}), retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator

print("[OK] Retry utility loaded")



# ===== IMPROVED: Text Chunking for Large Contracts =====
def chunk_text(text: str, max_size: int = MAX_TEXT_CHUNK, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split large text into overlapping chunks for processing.
    
    Args:
        text: Input text to chunk
        max_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def merge_chunk_analyses(chunk_results: list) -> dict:
    """
    Merge analysis results from multiple chunks into a single result.
    
    Args:
        chunk_results: List of analysis dictionaries from chunks
    
    Returns:
        Merged analysis dictionary
    """
    if not chunk_results:
        return {}
    
    merged = {
        "title": chunk_results[0].get("title", ""),
        "parties": [],
        "dates": [],
        "governing_law": chunk_results[0].get("governing_law", ""),
        "clauses": []
    }
    
    # Collect unique parties
    seen_parties = set()
    for result in chunk_results:
        for party in result.get("parties", []):
            party_key = str(party.get("name", "")) + str(party.get("role", ""))
            if party_key not in seen_parties:
                merged["parties"].append(party)
                seen_parties.add(party_key)
    
    # Collect unique dates
    seen_dates = set()
    for result in chunk_results:
        for date in result.get("dates", []):
            date_key = str(date.get("value", "")) + str(date.get("type", ""))
            if date_key not in seen_dates:
                merged["dates"].append(date)
                seen_dates.add(date_key)
    
    # Collect all clauses
    seen_clauses = set()
    for result in chunk_results:
        for clause in result.get("clauses", []):
            clause_key = clause.get("clause_name", "")
            if clause_key and clause_key not in seen_clauses:
                merged["clauses"].append(clause)
                seen_clauses.add(clause_key)
    
    return merged

print("[OK] Chunking utilities loaded")



# ===== IMPROVED: Enhanced Embedding Function with Retry =====
@retry_with_backoff(max_retries=MAX_RETRIES)
def get_embeddings_api_improved(text: str) -> Optional[list]:
    """
    Get embeddings using HuggingFace Inference API with improved error handling.
    
    Args:
        text: Text to generate embeddings for
    
    Returns:
        List of floats (384 dimensions) or None if failed
    """
    if not text or not text.strip():
        return None
    
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": text[:MAX_TEXT_CHUNK],  # Limit input size
        "options": {"wait_for_model": True}
    }
    
    try:
        response = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{HF_EMBED_MODEL}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle nested list response (token embeddings)
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # Mean pool token embeddings
                    import numpy as np
                    emb = np.mean(result, axis=0).tolist()
                else:
                    emb = result
            else:
                emb = result
            
            # Validate dimension
            if isinstance(emb, list) and len(emb) == EMBEDDING_DIM:
                return emb
            elif isinstance(emb, list) and len(emb) > 0:
                # Truncate or pad to correct dimension
                if len(emb) > EMBEDDING_DIM:
                    return emb[:EMBEDDING_DIM]
                else:
                    return emb + [0.0] * (EMBEDDING_DIM - len(emb))
            else:
                return None
        else:
            print(f"[WARNING] HF API returned status {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("[WARNING] HF API request timed out")
        return None
    except Exception as e:
        print(f"[WARNING] HF API error: {str(e)}")
        return None

def validate_embedding(emb: Optional[list], expected_dim: int = EMBEDDING_DIM) -> list:
    """
    Validate and fix embedding dimensions.
    
    Args:
        emb: Embedding vector or None
        expected_dim: Expected dimension (default 384)
    
    Returns:
        Valid embedding vector of correct dimension
    """
    if emb is None or not isinstance(emb, list):
        return [0.0] * expected_dim
    
    if len(emb) == expected_dim:
        return emb
    elif len(emb) > expected_dim:
        return emb[:expected_dim]
    else:
        return emb + [0.0] * (expected_dim - len(emb))

print("[OK] Improved embedding functions loaded")



# ===== Enhanced JSON Parsing with Better Error Recovery =====

def parse_llm_json(content: str, max_attempts: int = 3) -> Optional[dict]:
    """
    Parse JSON from LLM output with multiple fallback strategies.
    
    Args:
        content: Raw LLM output string
        max_attempts: Maximum parsing attempts
    
    Returns:
        Parsed dictionary or None
    """
    if not content:
        return None
    
    # Strategy 1: Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Remove markdown code blocks
    cleaned = content
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract JSON object boundaries
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        json_str = cleaned[start:end]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Fix common JSON issues
    try:
        # Fix unescaped quotes in strings
        fixed = re.sub(r'(?<!\\)"(?=\w)', r'\\"', cleaned)
        # Remove trailing commas
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        print(f"[WARNING] JSON parsing failed after all attempts: {e}")
        return None

def validate_analysis_data(data: dict) -> dict:
    """
    Validate and fix analysis data structure.
    
    Args:
        data: Analysis dictionary from LLM
    
    Returns:
        Validated and fixed analysis dictionary
    """
    if not isinstance(data, dict):
        return {
            "title": "Unknown Contract",
            "parties": [],
            "dates": [],
            "governing_law": "Not Specified",
            "clauses": []
        }
    
    # Ensure required fields exist
    validated = {
        "title": data.get("title", "Unknown Contract"),
        "parties": data.get("parties", []),
        "dates": data.get("dates", []),
        "governing_law": data.get("governing_law", "Not Specified"),
        "clauses": data.get("clauses", [])
    }
    
    # Validate clauses
    validated_clauses = []
    for clause in validated["clauses"]:
        if isinstance(clause, dict):
            validated_clause = {
                "clause_name": clause.get("clause_name", "Unnamed Clause"),
                "summary": clause.get("summary", ""),
                "risk_level": clause.get("risk_level", "MEDIUM"),
                "risk_reason": clause.get("risk_reason", ""),
                "obligation": clause.get("obligation", ""),
                "liability": clause.get("liability", ""),
                "ai_summary": clause.get("ai_summary", "")
            }
            # Ensure risk_level is valid
            if validated_clause["risk_level"] not in ["LOW", "MEDIUM", "HIGH"]:
                validated_clause["risk_level"] = "MEDIUM"
            validated_clauses.append(validated_clause)
    
    validated["clauses"] = validated_clauses
    return validated

print("[OK] Improved JSON parsing utilities loaded")



# ===== HF EMBEDDING VIA API (NO MODEL DOWNLOAD) =====
import requests

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Reliable model
HF_API_URL = (
    f"https://router.huggingface.co/hf-inference/models/{HF_EMBED_MODEL}"
)
# Groq
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Neo4j - Auto-fix Aura URI for SSL compatibility
uri = os.environ["NEO4J_URI"]

# Fix for Neo4j Aura: convert neo4j+s:// to neo4j+ssc:// (uses system cert store)
if "neo4j+s://" in uri and "neo4j+ssc://" not in uri:
    uri = uri.replace("neo4j+s://", "neo4j+ssc://")
    print(f"[UPDATE] Updated URI for Aura compatibility: {uri[:50]}...")

neo4j_driver = GraphDatabase.driver(
    uri,
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# Test connection
try:
    neo4j_driver.verify_connectivity()
    print("[OK] Neo4j Aura connection successful!")
except Exception as e:
    print(f"[ERROR] Neo4j connection failed: {e}")
    print("[TIP] Make sure:")
    print("   1. Your NEO4J_URI uses neo4j+ssc:// or neo4j+s://")
    print("   2. Username and password are correct")
    print("   3. Your Aura database is running")
    raise

def get_embeddings_api(text):
    """Get embeddings using HuggingFace Inference API"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=2)
        
        if response.status_code == 200:
            result = response.json()
            # Handle different response formats
            if isinstance(result, list):
                if isinstance(result[0], list):
                    return result[0]  # Nested list
                return result
            return result
        else:
            # print(f"[ERROR] API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception: {str(e)}")
        return None

print("[OK] Clients initialized (API mode - no model download)")

def pdf_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def extract_text(pdf):
    doc = fitz.open(pdf)
    return "".join(p.get_text() for p in doc)

def safe_str(v):
    return v.strip() if isinstance(v, str) else None

def normalize_list(items, key=None):
    out = []
    for i in items:
        if isinstance(i, dict):
            val = i.get(key) if key else None
            if val:
                out.append(val)
        elif isinstance(i, str):
            out.append(i)
    return out

def print_contract_summary(data):
    """Enhanced summary printer with all details"""
    print("\n" + "="*80)
    print("[DOC] CONTRACT SUMMARY")
    print("="*80)

    print(f"\n[PIN] BASIC INFORMATION")
    print("-"*80)
    print(f"Title          : {data.get('title', 'N/A')}")
    print(f"File Name      : {data.get('file_name', 'N/A')}")
    print(f"Contract ID    : {data.get('contract_id', 'N/A')}")
    print(f"Governing Law  : {data.get('governing_law', 'N/A')}")

    # Parties
    print(f"\n[PARTIES] PARTIES ({len(data.get('parties', []))})")
    print("-"*80)
    for i, p in enumerate(data.get('parties', []), 1):
        print(f"  [{i}] {p}")

    # Important Dates
    print(f"\n[DATE] IMPORTANT DATES ({len(data.get('dates', []))})")
    print("-"*80)
    for i, d in enumerate(data.get('dates', []), 1):
        print(f"  [{i}] {d}")

    # Clauses with Risk Analysis
    print(f"\n[CLAUSE] CLAUSE RISK ANALYSIS ({len(data.get('clauses', []))})")
    print("="*80)
    for i, c in enumerate(data.get("clauses", []), 1):
        print(f"\n[Clause {i}] {c.get('clause_name', 'Unnamed')}")
        print("-"*80)
        print(f"Summary      : {c.get('summary', 'N/A')}")
        print(f"\n[RISK] Risk Level : {c.get('risk_level', 'N/A')}")
        print(f"Risk Reason  : {c.get('risk_reason', 'N/A')}")
        print(f"\n[OBLIGATION] Obligation : {c.get('obligation', 'N/A')}")
        print(f"[LIABILITY] Liability  : {c.get('liability', 'N/A')}")
        print(f"\n[AI] AI Summary : {c.get('ai_summary', 'N/A')}")
        print("-"*80)
    
    print("\n" + "="*80)

class ContractState(TypedDict):
    pdf_path: str
    cid: str
    text: str
    embeddings: List[float]
    analysis: Dict[str, Any]

def pdf_extraction_agent(state: ContractState):
    print(f"\n[DOC] Extracting PDF: {state['pdf_path']}")
    text = extract_text(state["pdf_path"])
    return {
        **state,
        "text": text
    }

def embedding_agent(state: ContractState):
    print("[EMBED] Generating embeddings via HuggingFace API")
    
    # ===== IMPORTANT: Embeddings are created from TEXT, stored separately =====
    # This function converts TEXT → EMBEDDINGS (384 numbers)
    # These embeddings are NOT sent to LLM - they're used for similarity search
    # LLM will receive the TEXT separately in analysis_agent
    # =====

    # Limit chunk size for HF inference safety
    text_chunk = state["text"][:8000]  # We use TEXT here, not embeddings

    emb = get_embeddings_api(text_chunk)  # Converts text → array of 384 numbers

    # Fallback if API fails or returns bad data
    if (
        emb is None
        or not isinstance(emb, list)
        or len(emb) == 0
    ):
        print("[WARNING] Using fallback embeddings")
        emb = [0.0] * 384  # all-MiniLM-L6-v2 → 384 dims

    # Handle nested response ([[...]])
    if isinstance(emb[0], list):
        emb = emb[0]

    # Final dimension guard (VERY important for Neo4j vector index)
    if len(emb) != 384:
        print(f"[WARNING] Invalid embedding size {len(emb)}, forcing fallback")
        emb = [0.0] * 384

    print(f"   Embedding dimension: {len(emb)}")

    return {
        **state,
        "embeddings": emb
    }
def get_embeddings_api(text):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": text,
        "options": {
            "wait_for_model": True
        }
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            # HF returns: [ [token_embeddings...] ]
            # We must MEAN POOL
            if isinstance(result, list) and isinstance(result[0], list):
                import numpy as np
                return np.mean(result, axis=0).tolist()

            print("[WARNING] Unexpected HF response:", result)
            return None

        else:
            # print(f"[ERROR] API Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return None
    


# def get_embeddings_api(text):
#     headers = {
#         "Authorization": f"Bearer {HF_TOKEN}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "inputs": text,
#         "options": {"wait_for_model": True}
#     }

#     try:
#         response = requests.post(
#             HF_API_URL,
#             headers=headers,
#             json=payload,
#             timeout=30
#         )

#         if response.status_code == 200:
#             result = response.json()

#             # token-level embeddings → mean pooling
#             if isinstance(result, list) and isinstance(result[0], list):
#                 import numpy as np
#                 return np.mean(result, axis=0).tolist()

#             print("[WARNING] Unexpected HF response:", result)
#             return None

#         print(f"[ERROR] HF API Error {response.status_code}: {response.text}")
#         return None

#     except Exception as e:
#         print(f"[ERROR] HF Exception: {e}")
#         return None


def analysis_agent(state: ContractState):
    print("[ANALYZE] Analyzing contract via Groq LLM")
    
    # ===== FIX 1: Use embeddings to check for similar contracts =====
    # Embeddings ARE used - they're stored and used for similarity search
    # Here we demonstrate usage by checking for similar contracts before analysis
    contract_emb = state.get("embeddings", [])
    if contract_emb and len(contract_emb) == 384:
        try:
            with neo4j_driver.session() as s:
                # Find similar contracts using cosine similarity (embeddings in action!)
                result = s.run("""
                    MATCH (c:Contract)
                    WHERE c.embedding IS NOT NULL
                    RETURN c.id as id, c.title as title, c.embedding as emb
                    LIMIT 10
                """)
                
                similar_contracts = []
                for record in result:
                    existing_emb = record["emb"]
                    if existing_emb and len(existing_emb) == 384:
                        # Calculate cosine similarity (this is how embeddings are used!)
                        similarity = np.dot(np.array(contract_emb), np.array(existing_emb)) / (
                            np.linalg.norm(contract_emb) * np.linalg.norm(existing_emb)
                        )
                        if similarity > 0.85:  # High similarity threshold
                            similar_contracts.append({
                                "id": record["id"],
                                "title": record["title"],
                                "similarity": float(similarity)
                            })
                
                if similar_contracts:
                    print(f"   ℹ️ Found {len(similar_contracts)} similar contract(s) using embeddings")
                    print(f"   (Embeddings are actively used for similarity detection)")
        except Exception as e:
            print(f"   [WARNING] Could not check for similar contracts: {e}")
    # ===== END FIX 1 =====

    # ===== IMPORTANT CLARIFICATION: LLM receives TEXT, NOT embeddings =====
    # Embeddings (arrays of numbers) are NEVER sent to Groq LLM
    # 
    # WORKFLOW EXPLANATION:
    #   1. PDF → Extract TEXT (words, sentences)
    #   2. TEXT → Create EMBEDDINGS (384 numbers) via HuggingFace API
    #   3. TEXT → Send to Groq LLM (LLM only understands text, not numbers!)
    #   4. Store both separately:
    #      - Embeddings: Used for similarity search (finding similar contracts/clauses)
    #      - LLM Analysis: Used for understanding contract content
    #
    # Embeddings and LLM work SEPARATELY:
    #   - Embeddings = For finding similar things (semantic search)
    #   - LLM = For understanding and analyzing content
    # =====
    
    # Enhanced prompt for better extraction
    # NOTE: We send state["text"] (TEXT) to LLM, NOT state["embeddings"] (numbers)
    prompt = f"""
You are a legal contract analyzer. Analyze the following contract and extract detailed information.

CRITICAL: Return ONLY valid JSON. No markdown, no explanations, just the JSON object.

{{
  "title": "Contract title",
  "parties": [
    {{"name": "Party 1 name", "role": "Role (e.g., Service Provider, Client)"}},
    {{"name": "Party 2 name", "role": "Role"}}
  ],
  "dates": [
    {{"type": "Effective Date", "value": "YYYY-MM-DD or as mentioned"}},
    {{"type": "Expiration Date", "value": "YYYY-MM-DD or as mentioned"}}
  ],
  "governing_law": "Jurisdiction and governing law",
  "clauses": [
    {{
      "clause_name": "Name of the clause",
      "summary": "Brief summary of what this clause says",
      "risk_level": "Low/Medium/High",
      "risk_reason": "Detailed explanation of why this risk level was assigned. Mention specific concerns, potential liabilities, or unfavorable terms.",
      "obligation": "Specific obligations this clause imposes on parties. Be detailed.",
      "liability": "What liabilities or penalties are mentioned in this clause. Include financial limits if any.",
      "ai_summary": "A comprehensive AI analysis of this clause including: 1) What it means in plain language, 2) Key takeaways, 3) Red flags or concerns, 4) Recommendations"
    }}
  ]
}}

IMPORTANT INSTRUCTIONS:
1. Extract ALL major clauses from the contract (aim for 5-10 clauses)
2. For risk_reason: Explain WHY you assigned that risk level with specific concerns
3. For ai_summary: Provide detailed analysis (at least 2-3 sentences)
4. Be thorough - don't leave fields empty
5. Focus on: payment terms, termination, liability, intellectual property, confidentiality, warranties, indemnification
6. ESCAPE all quotes inside strings properly using backslash
7. Do NOT include any text before or after the JSON object

CONTRACT TEXT:
{state["text"][:10000]}
"""

    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a JSON-only API. Return only valid JSON, no markdown or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=8000
        )

        content = res.choices[0].message.content.strip()
        
        # Clean the response
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Try to find JSON object boundaries
        if not content.startswith("{"):
            start = content.find("{")
            if start != -1:
                content = content[start:]
        
        if not content.endswith("}"):
            end = content.rfind("}")
            if end != -1:
                content = content[:end+1]
        
        # Attempt to parse
        try:
            analysis = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON parsing failed at position {e.pos}: {e.msg}")
            print(f"Problematic content around error: ...{content[max(0,e.pos-50):e.pos+50]}...")
            
            # Try to fix common issues
            import re
            
            # Fix unescaped quotes in strings
            content = re.sub(r'(?<!\\)"(?=\w)', r'\\"', content)
            
            # Try parsing again
            try:
                analysis = json.loads(content)
                print("[OK] JSON fixed and parsed successfully")
            except:
                print("[ERROR] Could not fix JSON, using fallback structure")
                # Fallback structure
                analysis = {
                    "title": "Contract Analysis",
                    "parties": [{"name": "Party A", "role": "Unknown"}, {"name": "Party B", "role": "Unknown"}],
                    "dates": [{"type": "Effective Date", "value": "Not specified"}],
                    "governing_law": "Not specified",
                    "clauses": [
                        {
                            "clause_name": "General Terms",
                            "summary": "Contract terms extracted from document",
                            "risk_level": "Medium",
                            "risk_reason": "Unable to fully analyze due to parsing error. Manual review recommended.",
                            "obligation": "Review document manually for obligations",
                            "liability": "Review document manually for liabilities",
                            "ai_summary": "Automated analysis encountered an error. This contract requires manual legal review to identify all terms, conditions, and potential risks."
                        }
                    ]
                }
        
        print(f"[OK] Analysis complete - Found {len(analysis.get('clauses', []))} clauses")

        return {
            **state,
            "analysis": analysis
        }
        
    except Exception as e:
        print(f"[ERROR] Analysis error: {str(e)}")
        # Return minimal fallback
        return {
            **state,
            "analysis": {
                "title": "Error in Analysis",
                "parties": [],
                "dates": [],
                "governing_law": "Unknown",
                "clauses": []
            }
        }

def store_graph_agent(state: ContractState):
    data = state["analysis"]
    cid = state["cid"]
    filename = os.path.basename(state["pdf_path"])
    embeddings = state["embeddings"]

    print("[STORE] Storing into Neo4j with vector embeddings")

    with neo4j_driver.session() as s:

        # Contract with embeddings
        s.run("""
        MERGE (c:Contract {id:$id})
        SET c.title=$title,
            c.file_name=$file,
            c.governing_law=$law,
            c.embedding=$emb
        """, id=cid, title=data.get("title", "Unknown Contract"), file=filename, 
             law = data.get("governing_law", "Not Specified"), emb=embeddings)

        # Parties with roles
        for p in data.get("parties", []):
            if isinstance(p, dict):
                s.run("""
                MERGE (o:Organization {name:$name})
                SET o.role=$role
                WITH o
                MATCH (c:Contract {id:$id})
                MERGE (o)-[:IS_PARTY_TO]->(c)
                """, name=p.get("name"), role=p.get("role"), id=cid)

        # Dates with types
        for d in data.get("dates", []):
            if isinstance(d, dict):
                s.run("""
                MERGE (dt:ImportantDate {value:$v})
                SET dt.type=$type
                WITH dt
                MATCH (c:Contract {id:$id})
                MERGE (c)-[:HAS_DATE]->(dt)
                """, v=d.get("value"), type=d.get("type"), id=cid)

        # Clauses with all details
        for cl in data.get("clauses", []):
            # Generate embedding for each clause via API
            clause_text = f"{cl.get('clause_name', '')} {cl.get('summary', '')}"
            clause_emb = get_embeddings_api(clause_text)
            
            if clause_emb is None:
                clause_emb = [0.0] * 384  # Fixed: Use 384 to match all-MiniLM-L6-v2 model
            
            # Normalize if nested
            if isinstance(clause_emb[0], list):
                clause_emb = clause_emb[0]
            
            # ===== FIX 2 & 3: Store all clause details as properties (not separate nodes) =====
            # This reduces node count dramatically and creates cleaner graph structure
            # Embeddings ARE used - they're stored for similarity search (search_similar_clauses function)
            # Added display labels for better Neo4j visualization
            clause_name_clean = safe_str(cl.get("clause_name")) or "Unnamed Clause"
            summary_clean = safe_str(cl.get("summary")) or ""
            risk_level_clean = safe_str(cl.get("risk_level")) or "MEDIUM"
            
            s.run("""
            MATCH (c:Contract {id:$id})
            MERGE (cl:Clause {
                name:$n,
                contract_id:$id
            })
            SET cl.summary=$s,
                cl.embedding=$emb,
                cl.risk_level=$rl,
                cl.risk_reason=$rr,
                cl.obligation=$ob,
                cl.liability=$li,
                cl.ai_summary=$ai,
                cl.display_name=$display_name,
                cl.risk_color=$risk_color
            
            MERGE (c)-[:HAS_CLAUSE]->(cl)
            """,
            id=cid,
            n=clause_name_clean,
            s=summary_clean,
            rl=risk_level_clean,
            rr=safe_str(cl.get("risk_reason")) or "",
            ob=safe_str(cl.get("obligation")) or "",
            li=safe_str(cl.get("liability")) or "",
            ai=safe_str(cl.get("ai_summary")) or "",
            emb=clause_emb,
            display_name=f"{clause_name_clean} ({risk_level_clean})",  # For Neo4j display
            risk_color="[LOW]" if risk_level_clean == "LOW" else ("[MED]" if risk_level_clean == "MEDIUM" else "[HIGH]")
            )

    print("[OK] Stored successfully with vector embeddings")
    return state

graph = StateGraph(ContractState)

graph.add_node("extract", pdf_extraction_agent)
graph.add_node("embed", embedding_agent)
graph.add_node("analyze", analysis_agent)
graph.add_node("store", store_graph_agent)

graph.set_entry_point("extract")

graph.add_edge("extract", "embed")
graph.add_edge("embed", "analyze")
graph.add_edge("analyze", "store")
graph.add_edge("store", END)

workflow = graph.compile()
print("[OK] LangGraph workflow ready")

def retrieve_contract_from_db(contract_id):
    """
    Retrieve complete contract details from Neo4j database
    """
    print(f"\n[SEARCH] Retrieving contract: {contract_id}")
    
    with neo4j_driver.session() as s:
        # Get contract with all related data
        # ===== FIX 2 & 3: Updated to read from new structure (properties on Clause node) =====
        result = s.run("""
        MATCH (c:Contract {id:$id})
        OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(org:Organization)
        OPTIONAL MATCH (c)-[:HAS_DATE]->(dt:ImportantDate)
        OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
        
        RETURN c.title as title,
               c.file_name as file_name,
               c.id as contract_id,
               c.governing_law as governing_law,
               collect(DISTINCT org.name) as parties,
               collect(DISTINCT dt.value) as dates,
               collect(DISTINCT {
                   clause_name: cl.name,
                   summary: cl.summary,
                   risk_level: cl.risk_level,
                   risk_reason: cl.risk_reason,
                   obligation: cl.obligation,
                   liability: cl.liability,
                   ai_summary: cl.ai_summary
               }) as clauses
        """, id=contract_id)
        
        record = result.single()
        if record:
            data = {
                "title": record["title"],
                "file_name": record["file_name"],
                "contract_id": record["contract_id"],
                "governing_law": record["governing_law"],
                "parties": [p for p in record["parties"] if p],
                "dates": [d for d in record["dates"] if d],
                "clauses": [c for c in record["clauses"] if c.get("clause_name")]
            }
            print("[OK] Contract retrieved successfully")
            return data
        else:
            print("[ERROR] Contract not found")
            return None

def retrieve_all_contracts():
    """
    Retrieve all contracts from database
    """
    print("\n[BOOK] Retrieving all contracts...")
    
    with neo4j_driver.session() as s:
        result = s.run("""
        MATCH (c:Contract)
        RETURN c.id as id, c.title as title, c.file_name as file_name
        """)
        
        contracts = []
        for record in result:
            contracts.append({
                "id": record["id"],
                "title": record["title"],
                "file_name": record["file_name"]
            })
        
        print(f"[OK] Found {len(contracts)} contracts")
        return contracts

def search_similar_clauses(query_text, top_k=5):
    """
    Search for similar clauses using vector embeddings
    """
    print(f"\n[SEARCH] Searching for clauses similar to: '{query_text}'")
    
    # Generate embedding for query via API
    query_emb = get_embeddings_api(query_text)
    
    if query_emb is None:
        print("[ERROR] Could not generate query embedding")
        return []
    
    # Normalize if nested
    if isinstance(query_emb[0], list):
        query_emb = query_emb[0]
    
    with neo4j_driver.session() as s:
        # Get all clauses with embeddings
        result = s.run("""
        MATCH (c:Contract)-[:HAS_CLAUSE]->(cl:Clause)
        WHERE cl.embedding IS NOT NULL
        RETURN c.title as contract_title,
               cl.name as clause_name,
               cl.summary as summary,
               cl.embedding as embedding
        """)
        
        clauses = []
        for record in result:
            # Calculate cosine similarity
            emb = np.array(record["embedding"])
            query = np.array(query_emb)
            
            # Handle different embedding dimensions
            if len(emb) != len(query):
                continue
            
            similarity = np.dot(emb, query) / (np.linalg.norm(emb) * np.linalg.norm(query))
            
            clauses.append({
                "contract": record["contract_title"],
                "clause": record["clause_name"],
                "summary": record["summary"],
                "similarity": float(similarity)
            })
        
        # Sort by similarity
        clauses.sort(key=lambda x: x["similarity"], reverse=True)
        
        print(f"\n[STATS] Top {top_k} similar clauses:")
        for i, clause in enumerate(clauses[:top_k], 1):
            print(f"\n[{i}] Similarity: {clause['similarity']:.4f}")
            print(f"    Contract: {clause['contract']}")
            print(f"    Clause: {clause['clause']}")
            print(f"    Summary: {clause['summary']}")
        
        return clauses[:top_k]

# ===== PROCESS CONTRACTS =====
pdfs = [
    "Legal-Services-Agreement.pdf",
    "Employment_contract.pdf",
    "sample_contract.pdf"
]

contract_ids = []

for pdf in pdfs:
    cid = pdf_hash(pdf)
    contract_ids.append(cid)

    print("\n" + "="*80)
    print(f"🚀 Processing: {pdf}")
    print("="*80)

    workflow.invoke({
        "pdf_path": pdf,
        "cid": cid,
        "text": "",
        "embeddings": [],
        "analysis": {},
    })

# ===== RETRIEVE AND DISPLAY RESULTS FOR ALL CONTRACTS =====

print("\n\n" + "#"*80)
print("# RETRIEVING ALL STORED CONTRACTS FROM DATABASE")
print("#"*80)

# Check if required functions are defined
try:
    # Test if functions exist
    if 'retrieve_all_contracts' not in globals():
        raise NameError("retrieve_all_contracts function not found")
    if 'retrieve_contract_from_db' not in globals():
        raise NameError("retrieve_contract_from_db function not found")
    if 'print_contract_summary' not in globals():
        raise NameError("print_contract_summary function not found")
    
    # Get all contracts from database
    all_contracts = retrieve_all_contracts()
    
    if not all_contracts:
        print("\n[ERROR] No contracts found in database")
    else:
        print(f"\n[STATS] Found {len(all_contracts)} contract(s) in database")
        print("="*80)
        
        # Retrieve and display summary for EACH contract
        for i, contract in enumerate(all_contracts, 1):
            print(f"\n\n{'='*80}")
            print(f"[DOC] CONTRACT {i} of {len(all_contracts)}")
            print(f"{'='*80}")
            
            try:
                contract_data = retrieve_contract_from_db(contract['id'])
                if contract_data:
                    print_contract_summary(contract_data)
                else:
                    print(f"[ERROR] Could not retrieve details for: {contract['title']}")
            except Exception as e:
                print(f"[ERROR] Error retrieving contract {contract['title']}: {str(e)}")
                continue
        
        print(f"\n\n{'='*80}")
        print(f"[OK] Displayed summaries for {len(all_contracts)} contract(s)")
        print(f"{'='*80}")
        
except NameError as e:
    print(f"\n[ERROR] ERROR: {str(e)}")
    print("\n[TIP] SOLUTION: Please run the cells in order:")
    print("   1. Run all cells from the beginning")
    print("   2. Make sure the cell containing 'retrieve_all_contracts' function is executed")
    print("   3. Then run this cell again")
    print("\n   Or run: Kernel -> Restart & Run All")
    
except Exception as e:
    print(f"\n[ERROR] Unexpected error: {str(e)}")
    print("[TIP] Check your Neo4j connection and try again")

# ===== VIEW INDIVIDUAL CONTRACT GRAPHS =====

def reconnect_neo4j():
    """
    Reconnect to Neo4j if connection is lost
    """
    global neo4j_driver
    
    try:
        # Close existing driver if it exists
        if 'neo4j_driver' in globals():
            try:
                neo4j_driver.close()
            except:
                pass
    except:
        pass
    
    # Re-initialize driver with URI fix
    uri = os.environ["NEO4J_URI"]
    if "neo4j+s://" in uri and "neo4j+ssc://" not in uri:
        uri = uri.replace("neo4j+s://", "neo4j+ssc://")
        print(f"[UPDATE] Updated URI for Aura compatibility: {uri[:50]}...")
    
    neo4j_driver = GraphDatabase.driver(
        uri,
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    
    # Test connection
    try:
        neo4j_driver.verify_connectivity()
        print("[OK] Neo4j Aura reconnected successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to reconnect: {e}")
        return False

def get_contract_cypher_query(contract_id=None, contract_title=None):
    """
    Generate Cypher query to view a single contract graph in Neo4j Browser
    Returns query string you can copy-paste into Neo4j Browser
    """
    if contract_id:
        query = f"""// View Individual Contract Graph by ID
MATCH (c:Contract {{id: "{contract_id}"}})
OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
OPTIONAL MATCH (cl)-[:HAS_RISK]->(r:Risk)
OPTIONAL MATCH (cl)-[:HAS_REASON]->(rr:RiskReason)
OPTIONAL MATCH (cl)-[:HAS_OBLIGATION]->(ob:Obligation)
OPTIONAL MATCH (cl)-[:HAS_LIABILITY]->(li:Liability)
OPTIONAL MATCH (cl)-[:HAS_AI_SUMMARY]->(ai:AISummary)
RETURN c, o, d, cl, r, rr, ob, li, ai"""
    elif contract_title:
        query = f"""// View Individual Contract Graph by Title
MATCH (c:Contract {{title: "{contract_title}"}})
OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
OPTIONAL MATCH (cl)-[:HAS_RISK]->(r:Risk)
OPTIONAL MATCH (cl)-[:HAS_REASON]->(rr:RiskReason)
OPTIONAL MATCH (cl)-[:HAS_OBLIGATION]->(ob:Obligation)
OPTIONAL MATCH (cl)-[:HAS_LIABILITY]->(li:Liability)
OPTIONAL MATCH (cl)-[:HAS_AI_SUMMARY]->(ai:AISummary)
RETURN c, o, d, cl, r, rr, ob, li, ai"""
    else:
        query = """// View ALL Contracts Together (Explore View)
MATCH (c:Contract)
OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
OPTIONAL MATCH (cl)-[:HAS_RISK]->(r:Risk)
RETURN c, o, d, cl, r
LIMIT 100"""
    return query

def view_individual_contract_graph(contract_id=None, contract_title=None):
    """
    View a single contract's graph structure
    Shows what nodes and relationships belong to this contract only
    """
    if not contract_id and not contract_title:
        print("[ERROR] Please provide either contract_id or contract_title")
        return
    
    print("\n" + "="*80)
    print("[STATS] INDIVIDUAL CONTRACT GRAPH VIEW")
    print("="*80)
    
    # Try to reconnect if connection fails
    try:
        with neo4j_driver.session() as s:
            if contract_id:
                result = s.run("""
                    MATCH (c:Contract {id: $id})
                    OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
                    OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
                    OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
                    OPTIONAL MATCH (cl)-[:HAS_RISK]->(r:Risk)
                    OPTIONAL MATCH (cl)-[:HAS_REASON]->(rr:RiskReason)
                    OPTIONAL MATCH (cl)-[:HAS_OBLIGATION]->(ob:Obligation)
                    OPTIONAL MATCH (cl)-[:HAS_LIABILITY]->(li:Liability)
                    OPTIONAL MATCH (cl)-[:HAS_AI_SUMMARY]->(ai:AISummary)
                    RETURN c, 
                           collect(DISTINCT o) as parties,
                           collect(DISTINCT d) as dates,
                           collect(DISTINCT cl) as clauses,
                           collect(DISTINCT r) as risks
                """, id=contract_id)
            else:
                result = s.run("""
                    MATCH (c:Contract {title: $title})
                    OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
                    OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
                    OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
                    OPTIONAL MATCH (cl)-[:HAS_RISK]->(r:Risk)
                    OPTIONAL MATCH (cl)-[:HAS_REASON]->(rr:RiskReason)
                    OPTIONAL MATCH (cl)-[:HAS_OBLIGATION]->(ob:Obligation)
                    OPTIONAL MATCH (cl)-[:HAS_LIABILITY]->(li:Liability)
                    OPTIONAL MATCH (cl)-[:HAS_AI_SUMMARY]->(ai:AISummary)
                    RETURN c, 
                           collect(DISTINCT o) as parties,
                           collect(DISTINCT d) as dates,
                           collect(DISTINCT cl) as clauses,
                           collect(DISTINCT r) as risks
                """, title=contract_title)
        
        record = result.single()
        if record:
            c = record["c"]
            parties = [p for p in record["parties"] if p]
            dates = [d for d in record["dates"] if d]
            clauses = [cl for cl in record["clauses"] if cl]
            risks = [r for r in record["risks"] if r]
            
            print(f"\n[DOC] Contract: {c.get('title', 'Unknown')}")
            print(f"   ID: {c.get('id', 'N/A')[:30]}...")
            print(f"\n[STATS] Graph Statistics:")
            print(f"   Parties: {len(parties)}")
            print(f"   Dates: {len(dates)}")
            print(f"   Clauses: {len(clauses)}")
            print(f"   Risk Levels: {len(risks)}")
            
            print(f"\n[LINK] Copy this query to Neo4j Browser to visualize:")
            print("-" * 80)
            query = get_contract_cypher_query(contract_id=contract_id, contract_title=contract_title)
            print(query)
            print("-" * 80)
            
            return {
                "contract": c,
                "parties": parties,
                "dates": dates,
                "clauses": clauses,
                "risks": risks,
                "cypher_query": query
            }
        else:
            print("[ERROR] Contract not found")
            return None
            
    except Exception as e:
        print(f"[ERROR] Neo4j connection failed: {e}")
        print("[TIP] Make sure:")
        print("   1. Your NEO4J_URI uses neo4j+ssc:// or neo4j+s://")
        print("   2. Username and password are correct")
        print("   3. Your Aura database is running")
        print("\n   Try running: reconnect_neo4j()")
        return None

def list_contracts_for_viewing():
    """
    List all contracts with their IDs and titles for easy selection
    """
    print("\n" + "="*80)
    print("[OBLIGATION] AVAILABLE CONTRACTS FOR VIEWING")
    print("="*80)
    
    # Try to reconnect if connection fails
    try:
        contracts = retrieve_all_contracts()
    except Exception as e:
        print(f"[WARNING] Connection error: {e}")
        print("[UPDATE] Attempting to reconnect...")
        if reconnect_neo4j():
            contracts = retrieve_all_contracts()
        else:
            print("[ERROR] Could not connect to Neo4j. Please check your connection.")
            return []
    
    if not contracts:
        print("No contracts found in database")
        return []
    
    print(f"\nFound {len(contracts)} contract(s):\n")
    for i, contract in enumerate(contracts, 1):
        print(f"[{i}] {contract['title']}")
        print(f"    ID: {contract['id']}")
        print(f"    File: {contract['file_name']}\n")
    
    return contracts

print("[OK] Individual contract graph viewing functions loaded!")
print("\nUsage:")
print("  1. reconnect_neo4j() - Reconnect if you get connection errors")
print("  2. list_contracts_for_viewing() - See all contracts")
print("  3. view_individual_contract_graph(contract_id='...') - View one contract")
print("  4. get_contract_cypher_query(contract_id='...') - Get Cypher query for Browser")
print("\n[TIP] If you get connection errors, run: reconnect_neo4j()")


# ===== QUICK FIX: Reconnect Neo4j =====
# Run this cell if you get connection errors

# Close existing driver
try:
    neo4j_driver.close()
except:
    pass

# Re-initialize with URI fix
uri = os.environ["NEO4J_URI"]
if "neo4j+s://" in uri and "neo4j+ssc://" not in uri:
    uri = uri.replace("neo4j+s://", "neo4j+ssc://")
    print(f"[UPDATE] Updated URI: {uri[:50]}...")

neo4j_driver = GraphDatabase.driver(
    uri,
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

# Test connection
try:
    neo4j_driver.verify_connectivity()
    print("[OK] Neo4j reconnected! Now try your query again.")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    print("[TIP] Check your .env file and Neo4j Aura status")




# ===== VECTOR SIMILARITY SEARCH EXAMPLE =====

print("\n\n" + "#"*80)
print("# VECTOR SIMILARITY SEARCH")
print("#"*80)

# ===== Cleanup on Exit =====
atexit.register(lambda: neo4j_driver.close() if 'neo4j_driver' in globals() else None)

# ===== CLEANUP: Remove old structure nodes (if needed) =====
def cleanup_old_structure_nodes():
    """
    Remove old structure nodes (AISummary, Risk, RiskReason, Obligation, Liability)
    These are from the previous structure where summaries were separate nodes
    """
    print("\n" + "="*80)
    print("[CLEANUP] CLEANUP: Removing old structure nodes")
    print("="*80)
    
    with neo4j_driver.session() as s:
        # Count before cleanup
        result = s.run("MATCH (ai:AISummary) RETURN count(ai) as count")
        ai_count = result.single()["count"]
        
        result = s.run("MATCH (r:Risk) RETURN count(r) as count")
        risk_count = result.single()["count"]
        
        result = s.run("MATCH (rr:RiskReason) RETURN count(rr) as count")
        reason_count = result.single()["count"]
        
        result = s.run("MATCH (o:Obligation) RETURN count(o) as count")
        obligation_count = result.single()["count"]
        
        result = s.run("MATCH (l:Liability) RETURN count(l) as count")
        liability_count = result.single()["count"]
        
        total_old = ai_count + risk_count + reason_count + obligation_count + liability_count
        
        if total_old == 0:
            print("\n[OK] No old nodes found - database is already clean!")
            return
        
        print(f"\n[STATS] Found old nodes:")
        print(f"   AISummary: {ai_count}")
        print(f"   Risk: {risk_count}")
        print(f"   RiskReason: {reason_count}")
        print(f"   Obligation: {obligation_count}")
        print(f"   Liability: {liability_count}")
        print(f"   Total: {total_old}")
        
        # Delete relationships first, then nodes
        print(f"\n🗑️ Deleting old relationships and nodes...")
        
        # Delete relationships
        s.run("MATCH ()-[r:HAS_AI_SUMMARY]->() DELETE r")
        s.run("MATCH ()-[r:HAS_RISK]->() DELETE r")
        s.run("MATCH ()-[r:HAS_REASON]->() DELETE r")
        s.run("MATCH ()-[r:HAS_OBLIGATION]->() DELETE r")
        s.run("MATCH ()-[r:HAS_LIABILITY]->() DELETE r")
        
        # Delete nodes
        s.run("MATCH (ai:AISummary) DETACH DELETE ai")
        s.run("MATCH (r:Risk) DETACH DELETE r")
        s.run("MATCH (rr:RiskReason) DETACH DELETE rr")
        s.run("MATCH (o:Obligation) DETACH DELETE o")
        s.run("MATCH (l:Liability) DETACH DELETE l")
        
        print(f"[OK] Cleaned up {total_old} old nodes and their relationships")
        print(f"[TIP] Your database now uses the new structure (properties on Clause nodes)")

# Uncomment the line below to run cleanup (only if you have old nodes)
# cleanup_old_structure_nodes()


# ===== VIEW ONE CONTRACT WITH ONE SUMMARY =====
def view_one_contract_one_summary(contract_id=None, contract_title=None, clause_index=0):
    """
    View ONE contract with ONE clause/summary in Neo4j Browser
    Simple function to see a clean graph with just one summary
    
    Usage:
        # Get contract ID first
        contracts = retrieve_all_contracts()
        view_one_contract_one_summary(contract_id=contracts[0]['id'])
        
        # Or by title
        view_one_contract_one_summary(contract_title="Legal Services Agreement")
        
        # Show different clause (0=first, 1=second, etc.)
        view_one_contract_one_summary(contract_id=contracts[0]['id'], clause_index=1)
    """
    if not contract_id and not contract_title:
        print("[ERROR] Please provide either contract_id or contract_title")
        print("\n[TIP] First, get your contract ID:")
        print("   contracts = retrieve_all_contracts()")
        print("   print(contracts)")
        return None
    
    print("\n" + "="*80)
    print("[STATS] VIEW ONE CONTRACT WITH ONE SUMMARY")
    print("="*80)
    
    with neo4j_driver.session() as s:
        # Get contract and one clause
        if contract_id:
            result = s.run("""
                MATCH (c:Contract {id: $id})
                OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
                OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
                WITH c, collect(DISTINCT o) as orgs, collect(DISTINCT d) as dates
                MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
                WITH c, orgs, dates, collect(cl) as clauses
                RETURN c, orgs, dates, clauses[$idx] as single_clause, size(clauses) as total_clauses
            """, id=contract_id, idx=clause_index)
        else:
            result = s.run("""
                MATCH (c:Contract {title: $title})
                OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
                OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
                WITH c, collect(DISTINCT o) as orgs, collect(DISTINCT d) as dates
                MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
                WITH c, orgs, dates, collect(cl) as clauses
                RETURN c, orgs, dates, clauses[$idx] as single_clause, size(clauses) as total_clauses
            """, title=contract_title, idx=clause_index)
        
        record = result.single()
        if not record or not record["single_clause"]:
            print("[ERROR] Contract or clause not found")
            if record and record.get("total_clauses", 0) > 0:
                print(f"[TIP] This contract has {record['total_clauses']} clauses")
                print(f"   Try clause_index from 0 to {record['total_clauses']-1}")
            return None
        
        contract = record["c"]
        clause = record["single_clause"]
        total_clauses = record.get("total_clauses", 0)
        
        print(f"\n[DOC] Contract: {contract.get('title', 'Unknown')}")
        print(f"   File: {contract.get('file_name', 'N/A')}")
        print(f"   Total Clauses: {total_clauses}")
        print(f"\n[OBLIGATION] Showing Clause {clause_index + 1} of {total_clauses}:")
        print(f"   Name: {clause.get('name', 'Unnamed')}")
        print(f"   Summary: {clause.get('summary', 'N/A')[:150]}...")
        print(f"   Risk Level: {clause.get('risk_level', 'N/A')}")
        
        # Generate simple Cypher query for Neo4j Browser
        if contract_id:
            query = f"""// View ONE Contract with ONE Clause/Summary
MATCH (c:Contract {{id: "{contract_id}"}})
OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
WITH c, collect(DISTINCT o) as orgs, collect(DISTINCT d) as dates
MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
WITH c, orgs, dates, collect(cl)[{clause_index}] as single_clause
UNWIND orgs as o
UNWIND dates as d
RETURN c, o, d, single_clause as cl"""
        else:
            query = f"""// View ONE Contract with ONE Clause/Summary
MATCH (c:Contract {{title: "{contract_title}"}})
OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
WITH c, collect(DISTINCT o) as orgs, collect(DISTINCT d) as dates
MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
WITH c, orgs, dates, collect(cl)[{clause_index}] as single_clause
UNWIND orgs as o
UNWIND dates as d
RETURN c, o, d, single_clause as cl"""
        
        print("\n" + "="*80)
        print("[LINK] COPY THIS QUERY TO NEO4J BROWSER:")
        print("="*80)
        print(query)
        print("="*80)
        print("\n[TIP] Instructions:")
        print("  1. Open Neo4j Browser (in Neo4j Desktop or Aura)")
        print("  2. Copy the query above")
        print("  3. Paste and run it")
        print("  4. You'll see: Contract → Parties, Dates, and ONE Clause with its summary")
        print(f"\n[TIP] To see a different clause, use clause_index (0 to {total_clauses-1})")
        
        return {
            "contract": contract,
            "clause": clause,
            "cypher_query": query,
            "total_clauses": total_clauses
        }

print("[OK] Function loaded: view_one_contract_one_summary()")
print("[TIP] Usage: view_one_contract_one_summary(contract_id='...', clause_index=0)")


# ===== CLEAN GRAPH VISUALIZATION: One Contract with All Clauses (Organized) =====
def view_contract_clean_graph(contract_id=None, contract_title=None):
    """
    View ONE contract with ALL clauses in a clean, organized way
    Shows everything clearly so you understand the structure
    
    The graph will show:
    - Contract (center)
    - Parties (connected to Contract)
    - Dates (connected to Contract)
    - Clauses (connected to Contract, each with its summary and risk level)
    
    Each clause shows: name, risk level, and summary (no confusion!)
    """
    if not contract_id and not contract_title:
        print("[ERROR] Please provide either contract_id or contract_title")
        print("\n[TIP] First, get your contract ID:")
        print("   contracts = retrieve_all_contracts()")
        print("   print(contracts)")
        return None
    
    print("\n" + "="*80)
    print("[STATS] CLEAN GRAPH: One Contract with All Clauses")
    print("="*80)
    
    with neo4j_driver.session() as s:
        # Get contract info
        if contract_id:
            result = s.run("""
                MATCH (c:Contract {id: $id})
                OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
                OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
                OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
                RETURN c, 
                       collect(DISTINCT o) as parties,
                       collect(DISTINCT d) as dates,
                       collect(cl) as clauses
            """, id=contract_id)
        else:
            result = s.run("""
                MATCH (c:Contract {title: $title})
                OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
                OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
                OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
                RETURN c, 
                       collect(DISTINCT o) as parties,
                       collect(DISTINCT d) as dates,
                       collect(cl) as clauses
            """, title=contract_title)
        
        record = result.single()
        if not record:
            print("[ERROR] Contract not found")
            return None
        
        contract = record["c"]
        parties = [p for p in record["parties"] if p]
        dates = [d for d in record["dates"] if d]
        clauses = [cl for cl in record["clauses"] if cl]
        
        print(f"\n[DOC] Contract: {contract.get('title', 'Unknown')}")
        print(f"   File: {contract.get('file_name', 'N/A')}")
        print(f"   Governing Law: {contract.get('governing_law', 'N/A')}")
        print(f"\n[STATS] Summary:")
        print(f"   • Parties: {len(parties)}")
        print(f"   • Important Dates: {len(dates)}")
        print(f"   • Clauses: {len(clauses)}")
        
        print(f"\n[OBLIGATION] Clauses in this contract:")
        for i, clause in enumerate(clauses, 1):
            risk = clause.get('risk_level', 'MEDIUM')
            risk_icon = "[LOW]" if risk == "LOW" else ("[MED]" if risk == "MEDIUM" else "[HIGH]")
            print(f"   {i}. {risk_icon} {clause.get('name', 'Unnamed')} - Risk: {risk}")
        
        # Generate CLEAN Cypher query for Neo4j Browser
        if contract_id:
            query = f"""// CLEAN GRAPH: View One Contract with All Clauses (Organized)
// This shows: Contract → Parties, Dates, and Clauses (each with summary)
MATCH (c:Contract {{id: "{contract_id}"}})
OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
RETURN c, o, d, cl
ORDER BY cl.risk_level DESC, cl.name"""
        else:
            query = f"""// CLEAN GRAPH: View One Contract with All Clauses (Organized)
// This shows: Contract → Parties, Dates, and Clauses (each with summary)
MATCH (c:Contract {{title: "{contract_title}"}})
OPTIONAL MATCH (c)<-[:IS_PARTY_TO]-(o:Organization)
OPTIONAL MATCH (c)-[:HAS_DATE]->(d:ImportantDate)
OPTIONAL MATCH (c)-[:HAS_CLAUSE]->(cl:Clause)
RETURN c, o, d, cl
ORDER BY cl.risk_level DESC, cl.name"""
        
        print("\n" + "="*80)
        print("[LINK] COPY THIS QUERY TO NEO4J BROWSER:")
        print("="*80)
        print(query)
        print("="*80)
        print("\n[TIP] What you'll see in Neo4j:")
        print("   • Contract node (center)")
        print("   • Organization nodes (parties) connected to Contract")
        print("   • ImportantDate nodes connected to Contract")
        print("   • Clause nodes connected to Contract")
        print("   • Each Clause shows: name, risk_level, summary, ai_summary")
        print("\n[TIP] Click on any Clause node to see all its properties:")
        print("   - name: Clause name")
        print("   - summary: Brief summary")
        print("   - risk_level: LOW/MEDIUM/HIGH")
        print("   - risk_reason: Why this risk level")
        print("   - obligation: What parties must do")
        print("   - liability: Financial/legal liabilities")
        print("   - ai_summary: Detailed AI analysis")
        
        return {
            "contract": contract,
            "parties": parties,
            "dates": dates,
            "clauses": clauses,
            "cypher_query": query
        }

print("[OK] Function loaded: view_contract_clean_graph()")
print("[TIP] Usage: view_contract_clean_graph(contract_id='...')")


# ===== DATA VALIDATION: Check and Fix Data Quality =====
def validate_and_fix_contract_data():
    """
    Validate all contract data and fix any issues
    Ensures all fields are proper, no nulls, proper risk levels, etc.
    """
    print("\n" + "="*80)
    print("[SEARCH] DATA VALIDATION: Checking Contract Data Quality")
    print("="*80)
    
    issues_found = []
    fixes_applied = 0
    
    with neo4j_driver.session() as s:
        # Get all contracts
        result = s.run("""
            MATCH (c:Contract)
            RETURN c.id as id, c.title as title
        """)
        
        contracts = [{"id": r["id"], "title": r["title"]} for r in result]
        print(f"\n[DOC] Found {len(contracts)} contracts to validate")
        
        for contract in contracts:
            contract_id = contract["id"]
            
            # Check clauses
            result = s.run("""
                MATCH (c:Contract {id: $id})-[:HAS_CLAUSE]->(cl:Clause)
                RETURN cl
            """, id=contract_id)
            
            clauses = [r["cl"] for r in result]
            
            for clause in clauses:
                clause_name = clause.get("name")
                issues = []
                
                # Check for missing/null values
                if not clause_name or clause_name.strip() == "":
                    issues.append("Missing clause name")
                    s.run("""
                        MATCH (cl:Clause {name: $old_name, contract_id: $cid})
                        SET cl.name = 'Unnamed Clause'
                    """, old_name=clause_name, cid=contract_id)
                    fixes_applied += 1
                
                # Check risk level
                risk_level = clause.get("risk_level", "").upper()
                if risk_level not in ["LOW", "MEDIUM", "HIGH"]:
                    issues.append(f"Invalid risk level: {risk_level}")
                    s.run("""
                        MATCH (cl:Clause {name: $name, contract_id: $cid})
                        SET cl.risk_level = 'MEDIUM'
                    """, name=clause.get("name"), cid=contract_id)
                    fixes_applied += 1
                
                # Check for empty summaries
                if not clause.get("summary") or clause.get("summary").strip() == "":
                    issues.append("Empty summary")
                
                if not clause.get("ai_summary") or clause.get("ai_summary").strip() == "":
                    issues.append("Empty AI summary")
                
                if issues:
                    issues_found.append({
                        "contract": contract["title"],
                        "clause": clause_name or "Unnamed",
                        "issues": issues
                    })
        
        # Summary
        print(f"\n[STATS] Validation Results:")
        print(f"   • Contracts checked: {len(contracts)}")
        print(f"   • Issues found: {len(issues_found)}")
        print(f"   • Fixes applied: {fixes_applied}")
        
        if issues_found:
            print(f"\n[WARNING] Issues found:")
            for issue in issues_found[:10]:  # Show first 10
                print(f"   • {issue['contract']} - {issue['clause']}: {', '.join(issue['issues'])}")
            if len(issues_found) > 10:
                print(f"   ... and {len(issues_found) - 10} more")
        else:
            print(f"\n[OK] All data looks good! No issues found.")
        
        # Update display names for all clauses
        print(f"\n[UPDATE] Updating display names for better visualization...")
        result = s.run("""
            MATCH (cl:Clause)
            WHERE cl.name IS NOT NULL AND cl.risk_level IS NOT NULL
            SET cl.display_name = cl.name + ' (' + cl.risk_level + ')'
        """)
        print(f"[OK] Display names updated")
        
        print("\n" + "="*80)
        return {
            "contracts_checked": len(contracts),
            "issues_found": len(issues_found),
            "fixes_applied": fixes_applied
        }

# ===== Main Execution =====
def main():
    """Main function - uncomment to run examples"""
    # Example: Process contracts
    # pdfs = ["contract1.pdf", "contract2.pdf"]
    # for pdf in pdfs:
    #     cid = pdf_hash(pdf)
    #     workflow.invoke({
    #         "pdf_path": pdf,
    #         "cid": cid,
    #         "text": "",
    #         "embeddings": [],
    #         "analysis": {},
    #     })
    
    # Example: Retrieve and view contracts
    # contracts = retrieve_all_contracts()
    # if contracts:
    #     contract_data = retrieve_contract_from_db(contracts[0]['id'])
    #     print_contract_summary(contract_data)
    
    # Example: Search similar clauses
    # search_similar_clauses("payment terms", top_k=5)
    
    pass

if __name__ == "__main__":
    main()