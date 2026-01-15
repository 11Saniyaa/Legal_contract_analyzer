# Legal Contract Analysis System 📄⚖️

AI-powered legal contract analysis system using LangGraph, Neo4j, and vector embeddings.

## 🌟 Features

* PDF contract processing with automated text extraction
* AI-powered clause analysis using Groq LLM (Llama 3.1)
* Risk assessment (LOW/MEDIUM/HIGH) for each clause
* Neo4j graph database storage with relationships
* Weaviate vector database for fast semantic search
* Precedent matching across contracts
* Web interface (Streamlit) for easy contract management
* Export to Excel and PDF formats
* Risk dashboard with interactive visualizations
* Graph visualization with Cypher queries

## 🏗️ Architecture

LangGraph workflow with four agents:

1. **PDF Extraction** \- Extracts text from PDFs
2. **Embedding** \- Generates vector embeddings (HuggingFace)
3. **Analysis** \- Analyzes contracts with Groq LLM
4. **Storage** \- Stores data in Neo4j (graph) and Weaviate (vectors)

## 🚀 Quick Start

### Prerequisites

* Python 3.8+
* Docker (for Weaviate - optional but recommended)
* Neo4j Database (Neo4j Aura recommended)
* API Keys: Groq API Key, HuggingFace Token

### Installation

1. **Clone and setup**:

```bash
git clone https://github.com/11Saniyaa/Legal_contract_analyzer.git
cd legal_contract_analyzer
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# or: source .venv/bin/activate  # Linux/Mac
```

2. **Start Weaviate (optional but recommended)**:

```bash
docker run -d -p 8080:8080 \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  semitechnologies/weaviate:latest
```

Verify Weaviate is running: `curl http://localhost:8080/v1/.well-known/ready`

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Configure environment** \- Create `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
WEAVIATE_URL=http://localhost:8080  # Optional: for vector search
```

**Note**: If `WEAVIATE_URL` is not set, the system will use Neo4j for similarity search (slower but works without Weaviate).

5. **Run the application**:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
WEAVIATE_URL=http://localhost:8080  # Optional: for fast vector search
```

### Weaviate Setup (Optional)

Weaviate provides fast vector search and precedent matching. To use it:

1. **Start Weaviate with Docker**:

```bash
docker run -d -p 8080:8080 \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  semitechnologies/weaviate:latest
```

2. **Add to `.env`**:

```env
WEAVIATE_URL=http://localhost:8080
```

3. **Benefits**:  
   * Faster similarity search (optimized vector operations)  
   * Precedent matching across contracts  
   * Better scalability for large datasets  
   * Automatic fallback to Neo4j if Weaviate unavailable

### Models

* **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
* **LLM**: Groq `llama-3.1-8b-instant`

### Retry Configuration

Adjust retry behavior in `legal_contract_analyzer.py`:

```python
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
```

## 🛡️ Error Handling

* Retry logic with exponential backoff for API calls
* Fallback embeddings if API fails
* JSON parsing with multiple fallback strategies
* Automatic risk level normalization
* Neo4j connection reconnection handling

## ⚠️ Disclaimer

This tool is for informational purposes only and does not constitute legal advice. Always consult with a qualified attorney for legal matters.

---

**Built with ❤️ for legal professionals and contract analysts**
