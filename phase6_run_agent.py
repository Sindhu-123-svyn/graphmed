"""
Phase 6: Run GraphMed ReAct Agent with LangGraph
"""

import sys
from pathlib import Path
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))

from src.agent import GraphMedAgent


def interactive_mode():
    """Interactive mode."""
    
    print("\n" + "="*60)
    print("GRAPHMED INTERACTIVE MODE")
    print("="*60)
    print("\nAvailable patients: P001-P100")
    print("Type 'exit' to quit, 'switch' to change patient\n")
    
    try:
        agent = GraphMedAgent()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    current_patient = "P001"
    session_id = agent.start_session(current_patient)
    print(f"Session started for {current_patient}")
    
    while True:
        user_input = input(f"[{current_patient}] Question: ").strip()

        if user_input.lower() == 'exit':
            break

        if user_input.lower() == "switch":
            patient_id = input("Enter Patient ID (P001-P100): ").strip().upper()
            if patient_id not in [f"P{i:03d}" for i in range(1, 101)]:
                print("Invalid patient id. Use P001-P100")
                continue
            current_patient = patient_id
            session_id = agent.start_session(current_patient)
            print(f"Switched to {current_patient}. New session started.")
            continue

        if not user_input:
            continue

        try:
            result = agent.ask(session_id, user_input, max_iterations=6)
            print("\n" + "=" * 60)
            print("GraphMed Response")
            print("=" * 60)
            print(result["answer"])
            print("-" * 60)
            print(f"Turn: {result['turn_count']}")
            print(f"Tools used this turn: {result.get('turn_tools_called', [])}")
            print(f"Tools used so far: {result['tools_called']}")

            trace = result.get("compact_trace", {})
            if trace:
                print("\nReasoning Trace (Compact)")
                print("-" * 60)
                print(f"Fallback used: {trace.get('used_fallback', False)}")
                print(f"KB required for this query: {trace.get('kb_required', False)}")
                print(f"Graph update recommendation: {trace.get('graph_update_recommendation', 'Unknown')}")

                steps = trace.get("steps", [])
                if steps:
                    print("Steps:")
                    for step in steps:
                        print(f"  {step}")

                preview = trace.get("evidence_preview", {})
                if preview:
                    print("Evidence blocks:")
                    if preview.get("graph"):
                        print(f"  [GRAPH] {preview['graph']}")
                    if preview.get("memory_query"):
                        print(f"  [MEMORY_QUERY] {preview['memory_query']}")
                    if preview.get("memory_change"):
                        print(f"  [MEMORY_CHANGE_CHECK] {preview['memory_change']}")
                    if preview.get("clinical_delta"):
                        print(f"  [CLINICAL_DELTA] {preview['clinical_delta']}")
                    if preview.get("medical_kb"):
                        print(f"  [MEDICAL_KB] {preview['medical_kb']}")

                citations = trace.get("citation_preview", [])
                if citations:
                    print("Citation preview:")
                    for c in citations:
                        print(f"  - {c}")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def demo_mode():
    """Demo mode."""
    
    print("\n" + "="*60)
    print("DEMO MODE")
    print("="*60)
    
    try:
        agent = GraphMedAgent()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    session_id = agent.start_session("P001")
    
    queries = [
        ("P001", "What medications is this patient taking?"),
        ("P001", "What conditions does this patient have?"),
        ("P001", "Check if metformin and ibuprofen interact"),
        ("P001", "Based on previous context, summarize key clinical risks."),
    ]
    
    for i, (pid, q) in enumerate(queries, 1):
        print(f"\n{'─'*50}")
        print(f"Demo {i}/{len(queries)}")
        if i > 1:
            input("\nPress Enter...")
        if pid != "P001":
            session_id = agent.start_session(pid)
        result = agent.ask(session_id, q, max_iterations=6)
        print(result["answer"])


def main():
    print("\n" + "="*60)
    print("GRAPHMED - PHASE 6: REACT AGENT (LANGGRAPH)")
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