"""
Phase 6: Run GraphMed Direct ReAct Agent
"""

import sys
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from src.agent_direct import DirectReActAgent


def interactive_mode():
    """Interactive mode."""
    
    print("\n" + "="*60)
    print("💬 GRAPHMED INTERACTIVE MODE")
    print("="*60)
    print("\nAvailable patients: P001-P010")
    print("Type 'exit' to quit\n")
    
    try:
        agent = DirectReActAgent(max_steps=5)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    while True:
        patient_id = input("👤 Patient ID (P001-P010): ").strip().upper()
        
        if patient_id.lower() == 'exit':
            break
        
        if patient_id not in [f"P{i:03d}" for i in range(1, 11)]:
            print(f"❌ Invalid. Use P001-P010")
            continue
        
        query = input("❓ Question: ").strip()
        
        if query.lower() == 'exit':
            break
        
        if not query:
            continue
        
        try:
            agent.invoke(patient_id, query)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def demo_mode():
    """Demo mode."""
    
    print("\n" + "="*60)
    print("🎬 DEMO MODE")
    print("="*60)
    
    try:
        agent = DirectReActAgent(max_steps=5)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    queries = [
        ("P001", "What medications is this patient taking?"),
        ("P001", "What conditions does this patient have?"),
        ("P001", "Check if Metformin and Ibuprofen interact"),
    ]
    
    for i, (pid, q) in enumerate(queries, 1):
        print(f"\n{'─'*50}")
        print(f"Demo {i}/{len(queries)}")
        if i > 1:
            input("\nPress Enter...")
        agent.invoke(pid, q)


def main():
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 6: REACT AGENT")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Interactive mode")
    print("  2. Demo mode")
    print("  3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        interactive_mode()
    elif choice == "2":
        demo_mode()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()