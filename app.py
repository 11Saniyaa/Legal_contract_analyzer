"""
Legal Contract Analyzer - Web Interface
Streamlit web application for legal contract analysis
"""

import streamlit as st
import os
import sys
import tempfile
import io
from contextlib import redirect_stdout, redirect_stderr

# Suppress print statements from imported module
class SuppressOutput:
    def __init__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
    
    def __enter__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        return self
    
    def __exit__(self, *args):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

# Import functions (suppress initial print statements)
with SuppressOutput():
    from legal_contract_analyzer import (
        workflow, pdf_hash, retrieve_all_contracts, retrieve_contract_from_db,
        search_similar_clauses, view_contract_clean_graph, fix_all_risk_levels,
        validate_and_fix_contract_data
    )

# Page configuration
st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚖️ Legal Contract Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page:",
        ["Upload & Process", "View Contracts", "Search Clauses", "Graph Visualization"]
    )
    st.markdown("---")
    st.info("💡 Make sure your .env file is configured with API keys")

# Main content based on selected page
if page == "Upload & Process":
    st.header("📄 Upload and Process Contract")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF contract",
        type=['pdf'],
        help="Upload a legal contract PDF file to analyze"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        if st.button("🚀 Process Contract", type="primary"):
            with st.spinner("Processing contract... This may take a few minutes."):
                try:
                    # Generate contract ID
                    cid = pdf_hash(tmp_path)
                    
                    # Process contract with progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Capture output
                    output_capture = io.StringIO()
                    
                    with redirect_stdout(output_capture), redirect_stderr(output_capture):
                        status_text.text("📄 Extracting text from PDF...")
                        progress_bar.progress(25)
                        
                        status_text.text("🔢 Generating embeddings...")
                        progress_bar.progress(50)
                        
                        status_text.text("🧠 Analyzing with AI...")
                        progress_bar.progress(75)
                        
                        result = workflow.invoke({
                            "pdf_path": tmp_path,
                            "cid": cid,
                            "text": "",
                            "embeddings": [],
                            "analysis": {},
                        })
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    st.success(f"✅ Contract processed successfully!")
                    st.info(f"Contract ID: {cid[:30]}...")
                    
                    # Show summary
                    contract_data = retrieve_contract_from_db(cid)
                    if contract_data:
                        st.session_state['last_contract_id'] = cid
                        st.session_state['last_contract_data'] = contract_data
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing contract: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

elif page == "View Contracts":
    st.header("📚 View Stored Contracts")
    
    # Get all contracts
    try:
        with SuppressOutput():
            contracts = retrieve_all_contracts()
        
        if not contracts:
            st.warning("No contracts found in database.")
            st.info("Upload and process a contract first!")
        else:
            st.success(f"Found {len(contracts)} contract(s) in database")
            
            # Contract selector
            contract_options = {f"{c['file_name']} - {c['title']}": c['id'] for c in contracts}
            selected_contract = st.selectbox(
                "Select a contract to view:",
                options=list(contract_options.keys())
            )
            
            if selected_contract:
                contract_id = contract_options[selected_contract]
                
                if st.button("View Contract Details"):
                    with st.spinner("Loading contract details..."):
                        with SuppressOutput():
                            contract_data = retrieve_contract_from_db(contract_id)
                        
                        if contract_data:
                            # Display contract information
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📌 Basic Information")
                                st.write(f"**Title:** {contract_data.get('title', 'N/A')}")
                                st.write(f"**File Name:** {contract_data.get('file_name', 'N/A')}")
                                st.write(f"**Contract ID:** {contract_data.get('contract_id', 'N/A')[:30]}...")
                                st.write(f"**Governing Law:** {contract_data.get('governing_law', 'N/A')}")
                            
                            with col2:
                                st.subheader("📊 Statistics")
                                st.metric("Parties", len(contract_data.get('parties', [])))
                                st.metric("Important Dates", len(contract_data.get('dates', [])))
                                st.metric("Clauses", len(contract_data.get('clauses', [])))
                            
                            # Parties
                            if contract_data.get('parties'):
                                st.subheader("👥 Parties")
                                for i, party in enumerate(contract_data['parties'], 1):
                                    st.write(f"{i}. {party}")
                            
                            # Dates
                            if contract_data.get('dates'):
                                st.subheader("📅 Important Dates")
                                for i, date in enumerate(contract_data['dates'], 1):
                                    st.write(f"{i}. {date}")
                            
                            # Clauses
                            if contract_data.get('clauses'):
                                st.subheader("⚖️ Clause Analysis")
                                
                                for i, clause in enumerate(contract_data['clauses'], 1):
                                    with st.expander(f"Clause {i}: {clause.get('clause_name', 'Unnamed')}"):
                                        # Normalize risk level to uppercase for display
                                        risk_level_raw = clause.get('risk_level', 'MEDIUM')
                                        if isinstance(risk_level_raw, str):
                                            risk_level = risk_level_raw.strip().upper()
                                            if risk_level not in ["LOW", "MEDIUM", "HIGH"]:
                                                risk_level = "MEDIUM"  # Default fallback
                                        else:
                                            risk_level = "MEDIUM"
                                        
                                        # Risk level badge
                                        if risk_level == "HIGH":
                                            st.markdown(f'<p class="risk-high">🚨 Risk Level: {risk_level}</p>', unsafe_allow_html=True)
                                        elif risk_level == "MEDIUM":
                                            st.markdown(f'<p class="risk-medium">⚠️ Risk Level: {risk_level}</p>', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f'<p class="risk-low">✅ Risk Level: {risk_level}</p>', unsafe_allow_html=True)
                                        
                                        st.write(f"**Summary:** {clause.get('summary', 'N/A')}")
                                        st.write(f"**Risk Reason:** {clause.get('risk_reason', 'N/A')}")
                                        st.write(f"**Obligation:** {clause.get('obligation', 'N/A')}")
                                        st.write(f"**Liability:** {clause.get('liability', 'N/A')}")
                                        st.write(f"**AI Summary:** {clause.get('ai_summary', 'N/A')}")
                                        
    except Exception as e:
        st.error(f"Error loading contracts: {str(e)}")

elif page == "Search Clauses":
    st.header("🔍 Search Similar Clauses")
    
    st.write("Search for clauses similar to your query across all contracts in the database.")
    
    query = st.text_input(
        "Enter search query:",
        placeholder="e.g., payment terms, liability, termination clause"
    )
    
    top_k = st.slider("Number of results:", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching for similar clauses..."):
                try:
                    with SuppressOutput():
                        results = search_similar_clauses(query, top_k=top_k)
                    
                    if results:
                        st.success(f"Found {len(results)} similar clause(s)")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i}: {result['clause']} (Similarity: {result['similarity']:.4f})"):
                                st.write(f"**Contract:** {result['contract']}")
                                st.write(f"**Clause:** {result['clause']}")
                                st.write(f"**Summary:** {result['summary']}")
                                st.write(f"**Similarity Score:** {result['similarity']:.4f}")
                    else:
                        st.warning("No similar clauses found.")
                except Exception as e:
                    st.error(f"Error searching: {str(e)}")
        else:
            st.warning("Please enter a search query.")

elif page == "Graph Visualization":
    st.header("📊 Graph Visualization")
    
    st.write("View contract graphs in Neo4j Browser")
    
    try:
        with SuppressOutput():
            contracts = retrieve_all_contracts()
        
        if not contracts:
            st.warning("No contracts found. Process a contract first!")
        else:
            contract_options = {f"{c['file_name']} - {c['title']}": c['id'] for c in contracts}
            selected_contract = st.selectbox(
                "Select a contract:",
                options=list(contract_options.keys())
            )
            
            if selected_contract:
                contract_id = contract_options[selected_contract]
                
                if st.button("Generate Cypher Query"):
                    with st.spinner("Generating query..."):
                        try:
                            with SuppressOutput():
                                result = view_contract_clean_graph(contract_id=contract_id)
                            
                            if result:
                                st.success("✅ Query generated!")
                                
                                st.subheader("📋 Contract Information")
                                st.json({
                                    "Title": result['contract'].get('title'),
                                    "File": result['contract'].get('file_name'),
                                    "Parties": len(result['parties']),
                                    "Dates": len(result['dates']),
                                    "Clauses": len(result['clauses'])
                                })
                                
                                st.subheader("🔗 Cypher Query for Neo4j Browser")
                                st.code(result['cypher_query'], language="cypher")
                                
                                st.info("💡 Copy the query above and paste it into Neo4j Browser to visualize the graph")
                        except Exception as e:
                            st.error(f"Error generating query: {str(e)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

elif page == "Database Tools":
    st.header("🔧 Database Tools")
    
    st.write("Tools to fix and validate data in the database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fix Risk Levels")
        st.write("Normalize all risk levels in the database to uppercase (LOW, MEDIUM, HIGH)")
        if st.button("Fix All Risk Levels", type="primary"):
            with st.spinner("Fixing risk levels..."):
                try:
                    with SuppressOutput():
                        fixed_count = fix_all_risk_levels()
                    st.success(f"✅ Fixed {fixed_count} risk levels!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Validate All Data")
        st.write("Check and fix all contract data quality issues")
        if st.button("Run Data Validation", type="primary"):
            with st.spinner("Validating data..."):
                try:
                    with SuppressOutput():
                        result = validate_and_fix_contract_data()
                    st.success(f"✅ Validation complete!")
                    st.json({
                        "Issues Found": result.get('issues_found', 0),
                        "Fixes Applied": result.get('fixes_applied', 0)
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Legal Contract Analyzer** | Built with LangGraph, Neo4j, and Streamlit")

