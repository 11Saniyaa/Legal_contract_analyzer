"""
Test script to verify contracts are being stored correctly in Neo4j
Run this after uploading a contract to check if it was stored
"""

import os
from dotenv import load_dotenv
load_dotenv()

from legal_contract_analyzer import (
    neo4j_driver, 
    retrieve_all_contracts, 
    retrieve_contract_from_db
)

def test_contract_storage():
    """Test if contracts are being stored in Neo4j"""
    
    print("=" * 80)
    print("🔍 TESTING CONTRACT STORAGE")
    print("=" * 80)
    
    # 1. Check Neo4j connection
    print("\n[1] Checking Neo4j connection...")
    if neo4j_driver is None:
        print("❌ ERROR: Neo4j driver is not connected!")
        print("   Check your .env file: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
        return False
    
    try:
        neo4j_driver.verify_connectivity()
        print("✅ Neo4j connection: OK")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False
    
    # 2. Count all contracts
    print("\n[2] Counting contracts in database...")
    try:
        with neo4j_driver.session() as s:
            result = s.run("MATCH (c:Contract) RETURN count(c) as count")
            count = result.single()["count"]
            print(f"✅ Found {count} contract(s) in database")
    except Exception as e:
        print(f"❌ Error counting contracts: {e}")
        return False
    
    # 3. List all contracts
    print("\n[3] Listing all contracts...")
    try:
        contracts = retrieve_all_contracts()
        if not contracts:
            print("⚠️  No contracts found in database")
            print("   Upload a contract first using the Streamlit app")
            return False
        
        print(f"✅ Found {len(contracts)} contract(s):")
        for i, contract in enumerate(contracts, 1):
            print(f"\n   [{i}] {contract['title']}")
            print(f"       File: {contract['file_name']}")
            print(f"       ID: {contract['id'][:50]}...")
    except Exception as e:
        print(f"❌ Error retrieving contracts: {e}")
        return False
    
    # 4. Check contract details
    print("\n[4] Checking contract details...")
    if contracts:
        test_contract_id = contracts[0]['id']
        try:
            contract_data = retrieve_contract_from_db(test_contract_id)
            if contract_data:
                print(f"✅ Contract details retrieved successfully:")
                print(f"   Title: {contract_data.get('title', 'N/A')}")
                print(f"   File: {contract_data.get('file_name', 'N/A')}")
                print(f"   Parties: {len(contract_data.get('parties', []))}")
                print(f"   Dates: {len(contract_data.get('dates', []))}")
                print(f"   Clauses: {len(contract_data.get('clauses', []))}")
                
                # Check clauses have all required fields
                clauses = contract_data.get('clauses', [])
                if clauses:
                    print(f"\n   Sample clause:")
                    sample = clauses[0]
                    print(f"      Name: {sample.get('clause_name', 'N/A')}")
                    print(f"      Risk Level: {sample.get('risk_level', 'N/A')}")
                    print(f"      Has Summary: {'Yes' if sample.get('summary') else 'No'}")
                    print(f"      Has AI Summary: {'Yes' if sample.get('ai_summary') else 'No'}")
            else:
                print(f"❌ Could not retrieve contract details")
                return False
        except Exception as e:
            print(f"❌ Error retrieving contract details: {e}")
            return False
    
    # 5. Check Neo4j graph structure
    print("\n[5] Checking Neo4j graph structure...")
    try:
        with neo4j_driver.session() as s:
            # Count nodes
            result = s.run("MATCH (c:Contract) RETURN count(c) as count")
            contract_count = result.single()["count"]
            
            result = s.run("MATCH (o:Organization) RETURN count(o) as count")
            org_count = result.single()["count"]
            
            result = s.run("MATCH (cl:Clause) RETURN count(cl) as count")
            clause_count = result.single()["count"]
            
            result = s.run("MATCH (d:ImportantDate) RETURN count(d) as count")
            date_count = result.single()["count"]
            
            print(f"✅ Graph structure:")
            print(f"   Contracts: {contract_count}")
            print(f"   Organizations: {org_count}")
            print(f"   Clauses: {clause_count}")
            print(f"   Important Dates: {date_count}")
            
            # Check relationships
            result = s.run("MATCH ()-[r:IS_PARTY_TO]->() RETURN count(r) as count")
            party_rels = result.single()["count"]
            
            result = s.run("MATCH ()-[r:HAS_CLAUSE]->() RETURN count(r) as count")
            clause_rels = result.single()["count"]
            
            result = s.run("MATCH ()-[r:HAS_DATE]->() RETURN count(r) as count")
            date_rels = result.single()["count"]
            
            print(f"\n   Relationships:")
            print(f"   IS_PARTY_TO: {party_rels}")
            print(f"   HAS_CLAUSE: {clause_rels}")
            print(f"   HAS_DATE: {date_rels}")
            
    except Exception as e:
        print(f"❌ Error checking graph structure: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Contracts are being stored correctly!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    test_contract_storage()

