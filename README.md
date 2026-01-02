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

## 📖 Usage

### Web Interface (Recommended)

**Available Pages**:
- **Upload & Process**: Upload and analyze PDF contracts
- **View Contracts**: Browse contracts with export options (Excel/PDF)
- **Risk Dashboard**: Interactive risk distribution visualizations
- **Graph Visualization**: Generate Neo4j Cypher queries

### Programmatic Usage

```python
from legal_contract_analyzer import workflow, pdf_hash, retrieve_contract_from_db

# Process contract
cid = pdf_hash("contract.pdf")
workflow.invoke({
    "pdf_path": "contract.pdf",
    "cid": cid,
    "text": "",
    "embeddings": [],
    "analysis": {},
})

# Retrieve contract
contract_data = retrieve_contract_from_db(cid)
```

## 🔍 Extracted Information

For each contract:
- **Basic Info**: Title, Contract ID, File Name, Governing Law
- **Parties**: Names and roles
- **Important Dates**: Effective, expiration, and other critical dates
- **Clause Analysis**: Name, summary, risk level (LOW/MEDIUM/HIGH), risk reason, obligations, liabilities, AI summary

## 📊 Risk Levels

- **HIGH**: Unfavorable termination, unlimited liability, strict penalties, one-sided terms
- **MEDIUM**: Standard legal language, moderate obligations, typical industry terms
- **LOW**: Favorable terms, standard protections, reasonable conditions

## 📈 Features

- **Export**: Download contract summaries as Excel or PDF
- **Risk Dashboard**: Interactive charts showing risk distribution
- **Graph Visualization**: Generate Neo4j Cypher queries
- **Vector Search**: Find similar clauses using embeddings

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

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for informational purposes only and does not constitute legal advice. Always consult with a qualified attorney for legal matters.

## 🚀 Quick Start Guide

1. **Clone the repository**:
```bash
git clone https://github.com/11Saniyaa/Legal_contract_analyzer.git
cd legal_contract_analyzer
```

2. **Set up virtual environment** (recommended):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# or
source .venv/bin/activate   # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirement.txt
```

4. **Configure environment**:
   - Create `.env` file with your API keys and Neo4j credentials
   - See Configuration section above for details

5. **Run the application**:
```bash
streamlit run app.py
```

6. **Access the interface**:
   - Open browser to `http://localhost:8501`
   - Upload a PDF contract and start analyzing!

## 🙏 Acknowledgments

- **LangGraph** - Workflow orchestration
- **Neo4j** - Graph database
- **Groq** - Fast LLM inference
- **HuggingFace** - Embedding models
- **PyMuPDF** - PDF text extraction
- **Streamlit** - Web interface framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **ReportLab** - PDF generation

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for legal professionals and contract analysts**
