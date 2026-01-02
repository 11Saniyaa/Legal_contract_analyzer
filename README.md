# Legal Contract Analysis System 📄⚖️

AI-powered legal contract analysis system using LangGraph, Neo4j, and vector embeddings.

## 🌟 Features

- PDF contract processing with automated text extraction
- AI-powered clause analysis using Groq LLM (Llama 3.1)
- Risk assessment (LOW/MEDIUM/HIGH) for each clause
- Neo4j graph database storage with relationships
- Web interface (Streamlit) for easy contract management
- Export to Excel and PDF formats
- Risk dashboard with interactive visualizations
- Graph visualization with Cypher queries

## 🏗️ Architecture

LangGraph workflow with four agents:
1. **PDF Extraction** - Extracts text from PDFs
2. **Embedding** - Generates vector embeddings (HuggingFace)
3. **Analysis** - Analyzes contracts with Groq LLM
4. **Storage** - Stores data in Neo4j graph database

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Neo4j Database (Neo4j Aura recommended)
- API Keys: Groq API Key, HuggingFace Token

### Installation

1. **Clone and setup**:
```bash
git clone https://github.com/11Saniyaa/Legal_contract_analyzer.git
cd legal_contract_analyzer
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# or: source .venv/bin/activate  # Linux/Mac
```

2. **Install dependencies**:
```bash
pip install -r requirement.txt
```

3. **Configure environment** - Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

4. **Run the application**:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Embedding Model

The system uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) via HuggingFace API. To change models, update in `legal_contract_analyzer.py`:

```python
HF_EMBED_MODEL = "sentence-transformers/your-preferred-model"
EMBEDDING_DIM = 384  # Update if using different model
```

### LLM Model

Currently uses Groq's `llama-3.1-8b-instant`. Modify in `legal_contract_analyzer.py`:

```python
GROQ_MODEL = "llama-3.1-8b-instant"  # Change as needed
```

### Retry Configuration

Adjust retry behavior in `legal_contract_analyzer.py`:

```python
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
```

## 🛡️ Error Handling

- Retry logic with exponential backoff for API calls
- Fallback embeddings if API fails
- JSON parsing with multiple fallback strategies
- Automatic risk level normalization
- Neo4j connection reconnection handling

## 🔧 Configuration

### Environment Variables (.env)
```env
GROQ_API_KEY=your_key
HF_TOKEN=your_token
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Models
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **LLM**: Groq `llama-3.1-8b-instant`

## ⚠️ Disclaimer

This tool is for informational purposes only and does not constitute legal advice. Always consult with a qualified attorney for legal matters.

---

**Built with ❤️ for legal professionals and contract analysts**
