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
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

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
        view_contract_clean_graph
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

# Helper functions for export
def export_to_excel(contract_data):
    """Export contract data to Excel format"""
    data = {
        'Clause Name': [],
        'Summary': [],
        'Risk Level': [],
        'Risk Reason': [],
        'Obligation': [],
        'Liability': [],
        'AI Summary': []
    }
    
    for clause in contract_data.get('clauses', []):
        risk_level = clause.get('risk_level', 'MEDIUM')
        if isinstance(risk_level, str):
            risk_level = risk_level.strip().upper()
        
        data['Clause Name'].append(clause.get('clause_name', 'N/A'))
        data['Summary'].append(clause.get('summary', 'N/A'))
        data['Risk Level'].append(risk_level)
        data['Risk Reason'].append(clause.get('risk_reason', 'N/A'))
        data['Obligation'].append(clause.get('obligation', 'N/A'))
        data['Liability'].append(clause.get('liability', 'N/A'))
        data['AI Summary'].append(clause.get('ai_summary', 'N/A'))
    
    df = pd.DataFrame(data)
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Contract info sheet
        info_df = pd.DataFrame({
            'Field': ['Title', 'File Name', 'Contract ID', 'Governing Law', 'Parties', 'Important Dates'],
            'Value': [
                contract_data.get('title', 'N/A'),
                contract_data.get('file_name', 'N/A'),
                contract_data.get('contract_id', 'N/A')[:30] + '...',
                contract_data.get('governing_law', 'N/A'),
                ', '.join(contract_data.get('parties', [])),
                ', '.join(contract_data.get('dates', []))
            ]
        })
        info_df.to_excel(writer, sheet_name='Contract Info', index=False)
        
        # Clauses sheet
        df.to_excel(writer, sheet_name='Clauses', index=False)
    
    output.seek(0)
    return output

def export_to_pdf(contract_data):
    """Export contract data to PDF format"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30
    )
    story.append(Paragraph("Contract Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Contract Information
    story.append(Paragraph("Contract Information", styles['Heading2']))
    info_data = [
        ['Field', 'Value'],
        ['Title', contract_data.get('title', 'N/A')],
        ['File Name', contract_data.get('file_name', 'N/A')],
        ['Contract ID', contract_data.get('contract_id', 'N/A')[:30] + '...'],
        ['Governing Law', contract_data.get('governing_law', 'N/A')],
        ['Parties', ', '.join(contract_data.get('parties', []))],
        ['Important Dates', ', '.join(contract_data.get('dates', []))]
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))
    story.append(PageBreak())
    
    # Clauses
    story.append(Paragraph("Clause Analysis", styles['Heading2']))
    for i, clause in enumerate(contract_data.get('clauses', []), 1):
        risk_level = clause.get('risk_level', 'MEDIUM')
        if isinstance(risk_level, str):
            risk_level = risk_level.strip().upper()
        
        story.append(Paragraph(f"Clause {i}: {clause.get('clause_name', 'Unnamed')}", styles['Heading3']))
        story.append(Spacer(1, 6))
        
        clause_data = [
            ['Field', 'Value'],
            ['Summary', clause.get('summary', 'N/A')],
            ['Risk Level', risk_level],
            ['Risk Reason', clause.get('risk_reason', 'N/A')],
            ['Obligation', clause.get('obligation', 'N/A')],
            ['Liability', clause.get('liability', 'N/A')],
            ['AI Summary', clause.get('ai_summary', 'N/A')]
        ]
        clause_table = Table(clause_data, colWidths=[1.5*inch, 4.5*inch])
        clause_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(clause_table)
        story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Title
st.markdown('<h1 class="main-header">⚖️ Legal Contract Analyzer</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page:",
        ["Upload & Process", "View Contracts", "Risk Dashboard", "Graph Visualization"]
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
                            st.session_state['current_contract_data'] = contract_data
                            
                            # Export buttons
                            col_export1, col_export2 = st.columns(2)
                            with col_export1:
                                excel_data = export_to_excel(contract_data)
                                st.download_button(
                                    label="📊 Export to Excel",
                                    data=excel_data,
                                    file_name=f"contract_{contract_data.get('file_name', 'export')}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            with col_export2:
                                pdf_data = export_to_pdf(contract_data)
                                st.download_button(
                                    label="📄 Export to PDF",
                                    data=pdf_data,
                                    file_name=f"contract_{contract_data.get('file_name', 'export')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf"
                                )
                            
                            st.markdown("---")
                            
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

elif page == "Risk Dashboard":
    st.header("📊 Risk Dashboard")
    
    # Header with refresh button and timestamp
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    with col_header1:
        st.write("Visualize risk distribution across all contracts")
    with col_header2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            # Clear cache and rerun
            if 'dashboard_last_updated' in st.session_state:
                del st.session_state['dashboard_last_updated']
            st.rerun()
    with col_header3:
        if 'dashboard_last_updated' in st.session_state:
            st.caption(f"Last updated: {st.session_state['dashboard_last_updated']}")
        else:
            st.session_state['dashboard_last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"Last updated: {st.session_state['dashboard_last_updated']}")
    
    try:
        with SuppressOutput():
            contracts = retrieve_all_contracts()
        
        if not contracts:
            st.warning("No contracts found. Process a contract first!")
        else:
            # Collect all clauses from all contracts
            all_clauses = []
            contract_clause_map = {}
            
            with st.spinner("Loading contract data..."):
                for contract in contracts:
                    with SuppressOutput():
                        contract_data = retrieve_contract_from_db(contract['id'])
                    
                    if contract_data and contract_data.get('clauses'):
                        for clause in contract_data['clauses']:
                            risk_level = clause.get('risk_level', 'MEDIUM')
                            if isinstance(risk_level, str):
                                risk_level = risk_level.strip().upper()
                            
                            clause_info = {
                                'Contract': contract_data.get('title', 'Unknown'),
                                'Clause': clause.get('clause_name', 'Unnamed'),
                                'Risk Level': risk_level,
                                'Summary': clause.get('summary', 'N/A')
                            }
                            all_clauses.append(clause_info)
                            
                            if contract['id'] not in contract_clause_map:
                                contract_clause_map[contract['id']] = {
                                    'title': contract_data.get('title', 'Unknown'),
                                    'clauses': []
                                }
                            contract_clause_map[contract['id']]['clauses'].append(clause_info)
            
            # Update timestamp after loading
            st.session_state['dashboard_last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if not all_clauses:
                st.warning("No clauses found in contracts.")
            else:
                df = pd.DataFrame(all_clauses)
                
                # Risk distribution pie chart
                st.subheader("🎯 Overall Risk Distribution")
                risk_counts = df['Risk Level'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Level Distribution",
                        color_discrete_map={
                            'HIGH': '#d32f2f',
                            'MEDIUM': '#f57c00',
                            'LOW': '#388e3c'
                        }
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Export pie chart
                    col_export1, col_export2 = st.columns(2)
                    with col_export1:
                        pie_html = fig_pie.to_html()
                        st.download_button(
                            label="📥 Export as HTML",
                            data=pie_html,
                            file_name=f"risk_pie_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    with col_export2:
                        try:
                            pie_png = fig_pie.to_image(format="png")
                            st.download_button(
                                label="📥 Export as PNG",
                                data=pie_png,
                                file_name=f"risk_pie_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        except:
                            st.info("PNG export requires kaleido. Install: pip install kaleido")
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        x=risk_counts.index,
                        y=risk_counts.values,
                        title="Risk Level Count",
                        labels={'x': 'Risk Level', 'y': 'Number of Clauses'},
                        color=risk_counts.index,
                        color_discrete_map={
                            'HIGH': '#d32f2f',
                            'MEDIUM': '#f57c00',
                            'LOW': '#388e3c'
                        }
                    )
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Export bar chart
                    col_export3, col_export4 = st.columns(2)
                    with col_export3:
                        bar_html = fig_bar.to_html()
                        st.download_button(
                            label="📥 Export as HTML",
                            data=bar_html,
                            file_name=f"risk_bar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    with col_export4:
                        try:
                            bar_png = fig_bar.to_image(format="png")
                            st.download_button(
                                label="📥 Export as PNG",
                                data=bar_png,
                                file_name=f"risk_bar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        except:
                            st.info("PNG export requires kaleido")
                
                # Risk distribution by contract
                st.subheader("📋 Risk Distribution by Contract")
                
                contract_risk_data = []
                for contract_id, contract_info in contract_clause_map.items():
                    contract_df = pd.DataFrame(contract_info['clauses'])
                    risk_counts = contract_df['Risk Level'].value_counts().to_dict()
                    contract_risk_data.append({
                        'Contract': contract_info['title'],
                        'HIGH': risk_counts.get('HIGH', 0),
                        'MEDIUM': risk_counts.get('MEDIUM', 0),
                        'LOW': risk_counts.get('LOW', 0),
                        'Total': len(contract_info['clauses'])
                    })
                
                contract_risk_df = pd.DataFrame(contract_risk_data)
                
                if len(contract_risk_df) > 0:
                    fig_contract = go.Figure()
                    fig_contract.add_trace(go.Bar(
                        name='HIGH',
                        x=contract_risk_df['Contract'],
                        y=contract_risk_df['HIGH'],
                        marker_color='#d32f2f'
                    ))
                    fig_contract.add_trace(go.Bar(
                        name='MEDIUM',
                        x=contract_risk_df['Contract'],
                        y=contract_risk_df['MEDIUM'],
                        marker_color='#f57c00'
                    ))
                    fig_contract.add_trace(go.Bar(
                        name='LOW',
                        x=contract_risk_df['Contract'],
                        y=contract_risk_df['LOW'],
                        marker_color='#388e3c'
                    ))
                    fig_contract.update_layout(
                        barmode='stack',
                        title="Risk Distribution by Contract",
                        xaxis_title="Contract",
                        yaxis_title="Number of Clauses",
                        height=400
                    )
                    st.plotly_chart(fig_contract, use_container_width=True)
                    
                    # Export contract chart
                    col_export5, col_export6 = st.columns(2)
                    with col_export5:
                        contract_html = fig_contract.to_html()
                        st.download_button(
                            label="📥 Export Chart as HTML",
                            data=contract_html,
                            file_name=f"risk_by_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    with col_export6:
                        try:
                            contract_png = fig_contract.to_image(format="png")
                            st.download_button(
                                label="📥 Export Chart as PNG",
                                data=contract_png,
                                file_name=f"risk_by_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        except:
                            st.info("PNG export requires kaleido")
                    
                    # Summary table
                    st.subheader("📊 Summary Statistics")
                    st.dataframe(contract_risk_df, use_container_width=True)
                    
                    # Export summary table as CSV
                    csv_data = contract_risk_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Summary Table as CSV",
                        data=csv_data,
                        file_name=f"risk_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # High-risk clauses table
                high_risk_clauses = df[df['Risk Level'] == 'HIGH']
                if len(high_risk_clauses) > 0:
                    st.subheader("🚨 High-Risk Clauses")
                    st.dataframe(
                        high_risk_clauses[['Contract', 'Clause', 'Summary']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Export high-risk clauses as CSV
                    high_risk_csv = high_risk_clauses[['Contract', 'Clause', 'Summary']].to_csv(index=False)
                    st.download_button(
                        label="📥 Export High-Risk Clauses as CSV",
                        data=high_risk_csv,
                        file_name=f"high_risk_clauses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

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

# Footer
st.markdown("---")
st.markdown("**Legal Contract Analyzer** | Built with LangGraph, Neo4j, and Streamlit")

