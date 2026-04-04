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
    print("TESTING KNOWLEDGE BASE QUERIES")
    print("="*60)
    
    test_cases = [
        ("Drug interaction", "Is it safe to take metformin with ibuprofen?"),
        ("Disease info", "What are core management strategies for type 2 diabetes?"),
        ("Treatment guideline", "What do CDC guidelines recommend for hypertension management?"),
        ("Medication", "What are lisinopril contraindications?"),
        ("Lab values", "How should HbA1c be interpreted clinically?"),
        ("Evidence", "What does current literature say about CKD treatment progression?"),
    ]
    
    for name, query in test_cases:
        print(f"\n{name}:")
        print(f"   Query: {query}")
        results = kb.query(query, top_k=2)
        
        if results:
            for i, r in enumerate(results, 1):
                doc_type = r['metadata'].get('type', 'unknown')
                source = r['metadata'].get('source', 'unknown')
                score = r.get('score', 0.0)
                print(f"   Result {i} [{doc_type} | {source} | score={score:.3f}]: {r['document'][:120]}...")
        else:
            print(f"   No results found")

def interactive_query(kb: MedicalKnowledgeBase):
    """Interactive query mode."""
    
    print("\n" + "="*60)
    print("INTERACTIVE QUERY MODE")
    print("="*60)
    print("Type 'exit' to quit, 'summary' for KB summary\n")
    
    while True:
        query = input("Ask a medical question: ").strip()
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'summary':
            summary = kb.get_summary()
            print(f"\nKnowledge Base Summary:")
            print(f"   Total entries: {summary['total_entries']}")
            print(f"   By source: {summary.get('by_source', {})}")
            for doc_type, count in summary['by_type'].items():
                print(f"   - {doc_type}: {count}")
            continue
        
        results = kb.query(query, top_k=3)
        
        if results:
            print(f"\nFound {len(results)} relevant entries:\n")
            for i, r in enumerate(results, 1):
                print(
                    f"[{i}] Type: {r['metadata'].get('type', 'unknown')} | "
                    f"Source: {r['metadata'].get('source', 'unknown')} | "
                    f"Score: {r.get('score', 0.0):.3f}"
                )
                print(f"    {r['document']}")
                print()
        else:
            print("\nNo relevant medical knowledge found.\n")

def main():
    """Main execution for Phase 5."""
    
    print("\n" + "="*60)
    print("GRAPHMED - PHASE 5: MEDICAL KNOWLEDGE BASE")
    print("="*60)
    print("Building external medical reference for grounded answers")
    print("="*60)
    print("Source priority: DrugBank > CDC > PubMed > MedlinePlus")
    
    print("\nOptions:")
    print("  1. Build/update medical KB (priority pipeline)")
    print("  2. Rebuild KB from scratch (drops old collection)")
    print("  3. Test queries")
    print("  4. Interactive query mode")
    print("  5. Show knowledge base summary")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        confirm = input("\nBuild/update medical KB using priority pipeline? (y/n): ")
        if confirm.lower() == 'y':
            kb = create_medical_knowledge_base(rebuild=False)
            test_queries(kb)
        else:
            print("Cancelled.")

    elif choice == "2":
        confirm = input("\nRebuild from scratch (delete existing KB collection)? (y/n): ")
        if confirm.lower() == 'y':
            kb = create_medical_knowledge_base(rebuild=True)
            test_queries(kb)
        else:
            print("Cancelled.")

    elif choice == "3":
        # Try to load existing KB
        try:
            kb = MedicalKnowledgeBase("data/medical_kb")
            test_queries(kb)
        except Exception as e:
            print(f"Knowledge base not found. Please run option 1 first.")

    elif choice == "4":
        try:
            kb = MedicalKnowledgeBase("data/medical_kb")
            interactive_query(kb)
        except Exception as e:
            print(f"Knowledge base not found. Please run option 1 first.")

    elif choice == "5":
        try:
            kb = MedicalKnowledgeBase("data/medical_kb")
            summary = kb.get_summary()
            print(f"\nKnowledge Base Summary:")
            print(f"   Total entries: {summary['total_entries']}")
            print(f"   By source: {summary.get('by_source', {})}")
            print(f"\n   By type:")
            for doc_type, count in summary['by_type'].items():
                print(f"     - {doc_type}: {count}")
        except Exception as e:
            print(f"Knowledge base not found. Please run option 1 first.")

    elif choice == "6":
        print("Exiting...")
        
    else:
        print("Invalid choice. Building/updating knowledge base...")
        kb = create_medical_knowledge_base(rebuild=False)

if __name__ == "__main__":
    main()