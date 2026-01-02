# Legal Contract Analysis System 📄⚖️

An AI-powered legal contract analysis system that extracts, analyzes, and stores contract information using LangGraph workflows, Neo4j graph database, and vector embeddings for semantic search.

## 🌟 Features

- **PDF Contract Processing**: Automated extraction of text from legal contract PDFs
- **AI-Powered Analysis**: Uses Groq LLM (Llama 3.1) to extract and analyze contract clauses
- **Risk Assessment**: Automatic risk level classification (LOW/MEDIUM/HIGH) for each clause
- **Graph Database Storage**: Structured storage in Neo4j with relationships between entities
- **Vector Embeddings**: Semantic search capabilities using HuggingFace embeddings
- **Comprehensive Extraction**: Identifies parties, dates, obligations, liabilities, and governing law
- **Web Interface**: User-friendly Streamlit web application for easy contract management
- **Export Functionality**: Export contract summaries to Excel and PDF formats
- **Risk Dashboard**: Interactive visualizations showing risk distribution across contracts
- **Graph Visualization**: Generate Cypher queries for Neo4j graph visualization

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
streamlit
pymupdf
groq
neo4j
langgraph
sentence-transformers
python-dotenv
numpy
requests
pandas
openpyxl
reportlab
plotly
```

Install all dependencies:
```bash
pip install -r requirement.txt
```

## 📖 Usage

### Web Interface (Recommended)

The easiest way to use the system is through the Streamlit web interface:

1. **Start the application**:
```bash
# Activate virtual environment (if using)
.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate    # Linux/Mac

# Run Streamlit app
streamlit run app.py
```

2. **Access the interface**: Open your browser to `http://localhost:8501`

3. **Available Pages**:
   - **Upload & Process**: Upload PDF contracts and process them
   - **View Contracts**: Browse and view stored contracts with detailed analysis
   - **Risk Dashboard**: Visualize risk distribution with interactive charts
   - **Graph Visualization**: Generate Cypher queries for Neo4j graph viewing

### Programmatic Usage

#### Basic Contract Processing

```python
from legal_contract_analyzer import workflow, pdf_hash

# Process a single contract
cid = pdf_hash("path/to/contract.pdf")
result = workflow.invoke({
    "pdf_path": "path/to/contract.pdf",
    "cid": cid,
    "text": "",
    "embeddings": [],
    "analysis": {},
})
```

#### Retrieve Contract Data

```python
from legal_contract_analyzer import retrieve_contract_from_db, print_contract_summary

# Retrieve and display contract
contract_data = retrieve_contract_from_db("contract_id")
print_contract_summary(contract_data)
```

#### Get All Contracts

```python
from legal_contract_analyzer import retrieve_all_contracts

# Get list of all contracts
contracts = retrieve_all_contracts()
for contract in contracts:
    print(f"{contract['title']} - {contract['file_name']}")
```

## 🔍 Extracted Information

For each contract, the system extracts:

- **Basic Information**
  - Title
  - Contract ID (hash-based unique identifier)
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
  - Risk Level (LOW/MEDIUM/HIGH)
  - Risk Reason (detailed explanation of risk assessment)
  - Obligations (specific obligations imposed)
  - Liabilities (financial/legal liabilities mentioned)
  - AI-Generated Summary (comprehensive analysis)

## 📊 Risk Level Determination

Risk levels are assigned by the AI based on the following criteria:

- **HIGH Risk**: Unfavorable termination clauses, unlimited liability, strict penalties, one-sided terms
- **MEDIUM Risk**: Standard legal language, moderate obligations, typical industry terms
- **LOW Risk**: Favorable terms, standard protections, reasonable conditions

The AI analyzes each clause and provides a detailed `risk_reason` explaining why a particular risk level was assigned.

## 📈 Example Output

### Console Output

```
================================================================================
[DOC] CONTRACT SUMMARY
================================================================================

[PIN] BASIC INFORMATION
--------------------------------------------------------------------------------
Title          : Legal Services Agreement
Contract ID    : 1de79b4...
Governing Law  : California law

[PARTIES] PARTIES (2)
--------------------------------------------------------------------------------
  [1] Law Firm (Service Provider)
  [2] Client

[CLAUSE] CLAUSE RISK ANALYSIS (20)
================================================================================

[Clause 1] Payment Terms
--------------------------------------------------------------------------------
Summary      : Outlines payment structure and billing frequency

[RISK] Risk Level : HIGH
Risk Reason  : Contains automatic payment provisions with limited dispute window

[OBLIGATION] Obligation : Client must pay within 15 days of invoice
[LIABILITY] Liability  : Late fees of 1.5% per month on overdue amounts

[AI] AI Summary : This clause establishes a strict payment schedule...
```

### Web Interface Features

- **Interactive Risk Dashboard**: Visual charts showing risk distribution
- **Export Options**: Download contract summaries as Excel or PDF
- **Graph Visualization**: Generate Neo4j queries for visual contract relationships

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

The system includes comprehensive error handling:

- **Retry Logic**: Exponential backoff for API calls (Groq, HuggingFace)
- **Fallback Embeddings**: 384-dim zero vectors if embedding API fails
- **JSON Parsing**: Multiple fallback strategies for malformed LLM responses
- **Missing Fields**: Automatic detection and default values
- **Dimension Validation**: Ensures vector operations use correct dimensions
- **Neo4j Connection**: Automatic reconnection handling for Aura instances
- **Risk Level Normalization**: Automatic conversion to uppercase (LOW/MEDIUM/HIGH)

## 📊 Advanced Features

### Export Functionality

Export contract summaries in multiple formats:

- **Excel Export**: 
  - Contract Info sheet with basic details
  - Clauses sheet with all clause analysis
  - Downloadable via Streamlit interface

- **PDF Export**:
  - Formatted contract analysis report
  - Professional layout with tables
  - Includes all clause details

### Risk Dashboard

Interactive visualizations showing:

- **Overall Risk Distribution**: Pie and bar charts showing risk level percentages
- **Risk by Contract**: Stacked bar charts comparing risk across contracts
- **High-Risk Clauses Table**: List of all HIGH-risk clauses for quick review
- **Summary Statistics**: Aggregated metrics across all contracts

### Graph Visualization

Generate Cypher queries for Neo4j Browser:

- View contract relationships visually
- See connections between contracts, parties, dates, and clauses
- Export queries for use in Neo4j Desktop or Aura

### Vector Similarity Search

Uses cosine similarity to find semantically similar clauses:

```python
from legal_contract_analyzer import search_similar_clauses

# Find similar clauses
results = search_similar_clauses("payment terms", top_k=5)
```

### Multi-Contract Analysis

Process multiple contracts in batch:

```python
from legal_contract_analyzer import workflow, pdf_hash

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
