"""
Phase 4: Build vector memory stores for all patients
"""

import os
import sys
from pathlib import Path
import warnings
import logging


def _disable_chroma_telemetry_noise() -> None:
    """Disable telemetry calls that may fail due to PostHog API mismatch."""
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_TELEMETRY"] = "False"

    try:
        import posthog  # type: ignore

        def _capture_noop(*args, **kwargs):
            return None

        posthog.capture = _capture_noop
    except Exception:
        pass


# Suppress ALL ChromaDB telemetry warnings
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMA_OTEL_EXPORTER_ENDPOINT"] = ""
os.environ["ANONYMIZED_TELEMETRY"] = "False"

_disable_chroma_telemetry_noise()

# Disable all logging from chromadb
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pathlib import Path
from src.memory import build_memory_for_all_patients, GlobalMemoryManager, PatientMemoryStore

def test_retrieval(patient_id: str = "P001"):
    """Test memory retrieval for a specific patient."""
    
    print(f"\n{'='*60}")
    print(f"Testing Memory Retrieval for {patient_id}")
    print(f"{'='*60}")
    
    try:
        store = PatientMemoryStore(patient_id, "data/chroma_db")
        
        # Test different query styles including narrative context.
        test_queries = [
            ("fatigue", "Search for fatigue"),
            ("shortness of breath", "Search for breathing issues"),
            ("chest pain", "Search for cardiac symptoms"),
            ("patient seems worried and symptoms are getting worse", "Search for emotional progression"),
            ("doctor plans referral and close monitoring", "Search for clinician intent")
        ]
        
        for query, description in test_queries:
            print(f"\n🔍 {description}: '{query}'")
            results = store.retrieve_similar(query, top_k=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    visit_id = result['metadata'].get('visit_id', 'Unknown')
                    date = result['metadata'].get('date', 'Unknown')
                    has_uncertainty = result['metadata'].get('has_uncertainty_signal', False)
                    has_emotion = result['metadata'].get('has_emotion_signal', False)
                    has_intent = result['metadata'].get('has_clinician_intent', False)
                    print(f"   {i}. Visit {visit_id} ({date})")
                    print(
                        f"      Signals -> emotion: {has_emotion}, "
                        f"uncertainty: {has_uncertainty}, intent: {has_intent}"
                    )
                    print(f"      Preview: {result['document'][:100]}...")
            else:
                print(f"   No results found")
    except Exception as e:
        print(f"Error testing retrieval: {e}")

def show_memory_summary():
    """Show summary of all memory stores."""
    
    print(f"\n{'='*60}")
    print(f"📊 MEMORY STORE SUMMARY")
    print(f"{'='*60}")
    
    try:
        manager = GlobalMemoryManager("data/chroma_db")
        patients = manager.list_patients()
        
        print(f"\nPatients with memory stores: {len(patients)}")
        
        for patient_id in patients:
            try:
                store = manager.get_patient_store(patient_id)
                info = store.get_collection_info()
                print(f"  • {patient_id}: {info['count']} visits stored")
            except Exception as e:
                print(f"  • {patient_id}: Error - {e}")
    except Exception as e:
        print(f"Error getting summary: {e}")

def main():
    """Main execution for Phase 4."""
    
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 4: VECTOR MEMORY STORE")
    print("="*60)
    print("Building semantic memory from full patient visit narratives")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Build memory for all available patients")
    print("  2. Build memory for first 5 patients (test)")
    print("  3. Test retrieval for P001")
    print("  4. Show memory store summary")
    print("  5. Test semantic search across all patients")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        confirm = input("\nBuild memory for all available patients? (y/n): ")
        if confirm.lower() == 'y':
            results = build_memory_for_all_patients(verbose=True)
            print(f"\n✅ Built memory for {len(results)} patients")
            show_memory_summary()
            test_retrieval("P001")
        else:
            print("Cancelled.")
            
    elif choice == "2":
        print("\nBuilding memory for first 5 patients...")
        results = build_memory_for_all_patients(limit=5, verbose=True)
        show_memory_summary()
        test_retrieval("P001")
        
    elif choice == "3":
        test_retrieval("P001")
        
    elif choice == "4":
        show_memory_summary()
        
    elif choice == "5":
        print("\n🔍 Testing semantic search across all patients...")
        manager = GlobalMemoryManager("data/chroma_db")
        
        query = input("Enter search query (e.g., 'patient with fatigue and chest pain'): ")
        
        if query:
            results = manager.global_search(query, top_k=2)
            
            if results:
                print(f"\n📋 Search Results for: '{query}'")
                print("-"*60)
                for patient_id, patient_results in results.items():
                    print(f"\nPatient {patient_id}:")
                    for i, result in enumerate(patient_results, 1):
                        visit_id = result['metadata'].get('visit_id', 'Unknown')
                        date = result['metadata'].get('date', 'Unknown')
                        print(f"   {i}. Visit {visit_id} ({date})")
                        print(f"      {result['document'][:100]}...")
            else:
                print("No results found.")
        else:
            print("No query entered.")
        
    elif choice == "6":
        print("Exiting...")
        
    else:
        print("Invalid choice. Building memory for first 5 patients...")
        build_memory_for_all_patients(limit=5, verbose=True)

if __name__ == "__main__":
    main()