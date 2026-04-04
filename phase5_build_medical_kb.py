"""
Phase 5: Build Medical Knowledge Base
"""

import sys
from pathlib import Path
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.medical_kb import MedicalKnowledgeBase, create_medical_knowledge_base

def test_queries(kb: MedicalKnowledgeBase):
    """Test various queries against the knowledge base."""
    
    print("\n" + "="*60)
    print("🔍 TESTING KNOWLEDGE BASE QUERIES")
    print("="*60)
    
    test_cases = [
        ("Drug interaction", "Is it safe to take Metformin with Ibuprofen?"),
        ("Disease info", "What are the symptoms of Type 2 Diabetes?"),
        ("Treatment", "How is hypertension treated?"),
        ("Medication", "What is Metformin used for?"),
        ("Lab values", "What is a normal HbA1c level?"),
        ("Guideline", "What is the first-line treatment for diabetes?"),
    ]
    
    for name, query in test_cases:
        print(f"\n📋 {name}:")
        print(f"   Query: {query}")
        results = kb.query(query, top_k=2)
        
        if results:
            for i, r in enumerate(results, 1):
                doc_type = r['metadata'].get('type', 'unknown')
                print(f"   Result {i} [{doc_type}]: {r['document'][:120]}...")
        else:
            print(f"   No results found")

def interactive_query(kb: MedicalKnowledgeBase):
    """Interactive query mode."""
    
    print("\n" + "="*60)
    print("💬 INTERACTIVE QUERY MODE")
    print("="*60)
    print("Type 'exit' to quit, 'summary' for KB summary\n")
    
    while True:
        query = input("🔍 Ask a medical question: ").strip()
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'summary':
            summary = kb.get_summary()
            print(f"\n📊 Knowledge Base Summary:")
            print(f"   Total entries: {summary['total_entries']}")
            for doc_type, count in summary['by_type'].items():
                print(f"   - {doc_type}: {count}")
            continue
        
        results = kb.query(query, top_k=3)
        
        if results:
            print(f"\n📚 Found {len(results)} relevant entries:\n")
            for i, r in enumerate(results, 1):
                print(f"[{i}] Type: {r['metadata'].get('type', 'unknown')}")
                print(f"    {r['document']}")
                print()
        else:
            print("\n❌ No relevant medical knowledge found.\n")

def main():
    """Main execution for Phase 5."""
    
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 5: MEDICAL KNOWLEDGE BASE")
    print("="*60)
    print("Building external medical reference for grounded answers")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Create/populate medical knowledge base")
    print("  2. Test queries")
    print("  3. Interactive query mode")
    print("  4. Show knowledge base summary")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        confirm = input("\nCreate/populate medical knowledge base? (y/n): ")
        if confirm.lower() == 'y':
            kb = create_medical_knowledge_base()
            test_queries(kb)
        else:
            print("Cancelled.")
            
    elif choice == "2":
        # Try to load existing KB
        try:
            kb = MedicalKnowledgeBase("data/medical_kb")
            test_queries(kb)
        except Exception as e:
            print(f"❌ Knowledge base not found. Please run option 1 first.")
            
    elif choice == "3":
        try:
            kb = MedicalKnowledgeBase("data/medical_kb")
            interactive_query(kb)
        except Exception as e:
            print(f"❌ Knowledge base not found. Please run option 1 first.")
            
    elif choice == "4":
        try:
            kb = MedicalKnowledgeBase("data/medical_kb")
            summary = kb.get_summary()
            print(f"\n📊 Knowledge Base Summary:")
            print(f"   Total entries: {summary['total_entries']}")
            print(f"\n   By type:")
            for doc_type, count in summary['by_type'].items():
                print(f"     - {doc_type}: {count}")
        except Exception as e:
            print(f"❌ Knowledge base not found. Please run option 1 first.")
            
    elif choice == "5":
        print("Exiting...")
        
    else:
        print("Invalid choice. Creating knowledge base...")
        kb = create_medical_knowledge_base()

if __name__ == "__main__":
    main()