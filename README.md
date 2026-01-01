# Legal Contract Analysis System 📄⚖️

An AI-powered legal contract analysis system that extracts, analyzes, and stores contract information using LangGraph workflows, Neo4j graph database, and vector embeddings for semantic search.

## 🌟 Features

- **PDF Contract Processing**: Automated extraction of text from legal contract PDFs
- **AI-Powered Analysis**: Uses Groq LLM (Llama 3.1) to extract and analyze contract clauses
- **Risk Assessment**: Automatic risk level classification (Low/Medium/High) for each clause
- **Graph Database Storage**: Structured storage in Neo4j with relationships between entities
- **Vector Embeddings**: Semantic search capabilities using HuggingFace embeddings
- **Comprehensive Extraction**: Identifies parties, dates, obligations, liabilities, and governing law
- **Similarity Search**: Find similar clauses across multiple contracts

## 🏗️ Architecture

The system uses a **LangGraph workflow** with four main agents:

1. **PDF Extraction Agent**: Extracts text content from PDF files
2. **Embedding Agent**: Generates vector embeddings using HuggingFace API
3. **Analysis Agent**: Performs detailed contract analysis using Groq LLM
4. **Storage Agent**: Stores structured data in Neo4j graph database

## 📊 Data Model

### Neo4j Graph Structure

```
Contract
├── Properties: id, title, file_name, governing_law, embedding
├── Relationships:
    ├── IS_PARTY_TO ← Organization (name, role)
    ├── HAS_DATE → ImportantDate (value, type)
    └── HAS_CLAUSE → Clause
        ├── Properties: name, summary, embedding
        └── Relationships:
            ├── HAS_RISK → Risk (level)
            ├── HAS_REASON → RiskReason (text)
            ├── HAS_OBLIGATION → Obligation (text)
            ├── HAS_LIABILITY → Liability (text)
            └── HAS_AI_SUMMARY → AISummary (text)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Neo4j Database (local or cloud instance - Neo4j Aura recommended)
- API Keys:
  - Groq API Key
  - HuggingFace API Token

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal-contract-analysis.git
cd legal-contract-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your credentials:
```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
NEO4J_URI=bolt://localhost:7687
# OR for Neo4j Aura (recommended):
# NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Required Packages

```txt
python-dotenv
pymupdf
groq
neo4j
langgraph
sentence-transformers
numpy
requests
streamlit
```

## 📖 Usage

### Basic Contract Processing

```python
# Process a single contract
from workflow import workflow

result = workflow.invoke({
    "pdf_path": "path/to/contract.pdf",
    "cid": "unique_contract_id",
    "text": "",
    "embeddings": [],
    "analysis": {},
})
```

### Retrieve Contract Data

```python
from utils import retrieve_contract_from_db, print_contract_summary

# Retrieve and display contract
contract_data = retrieve_contract_from_db("contract_id")
print_contract_summary(contract_data)
```

### Semantic Search

```python
from utils import search_similar_clauses

# Find similar clauses
results = search_similar_clauses(
    "payment terms and conditions", 
    top_k=5
)
```

## 🔍 Extracted Information

For each contract, the system extracts:

- **Basic Information**
  - Title
  - Contract ID
  - File Name
  - Governing Law

- **Parties**
  - Party Names
  - Roles (Service Provider, Client, etc.)

- **Important Dates**
  - Effective Date
  - Expiration Date
  - Other critical dates

- **Clause Analysis** (for each clause)
  - Clause Name
  - Summary
  - Risk Level (Low/Medium/High)
  - Risk Reason (detailed explanation)
  - Obligations
  - Liabilities
  - AI-Generated Summary

## 📈 Example Output

```
================================================================================
📄 CONTRACT SUMMARY
================================================================================

📌 BASIC INFORMATION
--------------------------------------------------------------------------------
Title          : Legal Services Agreement
Contract ID    : 1de79b4...
Governing Law  : California law

👥 PARTIES (2)
--------------------------------------------------------------------------------
  [1] Law Firm (Service Provider)
  [2] Client

⚖️ CLAUSE RISK ANALYSIS (20)
================================================================================

[Clause 1] Payment Terms
--------------------------------------------------------------------------------
Summary      : Outlines payment structure and billing frequency
🚨 Risk Level : High
Risk Reason  : Contains automatic payment provisions with limited dispute window
📋 Obligation : Client must pay within 15 days of invoice
💼 Liability  : Late fees of 1.5% per month on overdue amounts
```

## 🔧 Configuration

### Embedding Model

The system uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) via HuggingFace API. To change models, update:

```python
HF_EMBED_MODEL = "sentence-transformers/your-preferred-model"
```

### LLM Model

Currently uses Groq's `llama-3.1-8b-instant`. Modify in the analysis agent:

```python
model="llama-3.1-8b-instant"  # Change as needed
```

## 🛡️ Error Handling

The system includes comprehensive error handling:

- Fallback embeddings if API fails (384-dim zero vectors)
- JSON parsing error recovery
- Missing field detection and default values
- Dimension validation for vector operations

## 📊 Advanced Features

### Vector Similarity Search

Uses cosine similarity to find semantically similar clauses:

```python
similarity = np.dot(embedding1, embedding2) / 
             (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
```

### Multi-Contract Analysis

Process multiple contracts in batch:

```python
pdfs = [
    "contract1.pdf",
    "contract2.pdf",
    "contract3.pdf"
]

for pdf in pdfs:
    cid = pdf_hash(pdf)
    workflow.invoke({
        "pdf_path": pdf,
        "cid": cid,
        "text": "",
        "embeddings": [],
        "analysis": {},
    })
```

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

## 🙏 Acknowledgments

- **LangGraph** - Workflow orchestration
- **Neo4j** - Graph database
- **Groq** - Fast LLM inference
- **HuggingFace** - Embedding models
- **PyMuPDF** - PDF text extraction

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for legal professionals and contract analysts**
