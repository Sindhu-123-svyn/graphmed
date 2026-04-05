"""
Phase 8: Test Graph Evolution & Conflict Resolution
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.graph_evolution import simulate_patient_evolution, evolve_patient_graph
from src.graph import PatientKnowledgeGraph


def test_core_operations():
    """Verify Addition, Update, and Conflict operations in one scenario."""

    print("\n" + "=" * 60)
    print("TESTING CORE OPERATIONS (ADD / UPDATE / CONFLICT)")
    print("=" * 60)

    visits = [
        {
            "date": "2024-01-10",
            "extracted": {
                "conditions": ["Type 2 Diabetes"],
                "medications": ["Metformin 500mg"],
                "symptoms": ["Fatigue"],
                "lab_values": {"HbA1c": 7.2},
            },
        },
        {
            "date": "2024-03-10",
            "extracted": {
                "conditions": ["Type 2 Diabetes"],
                "medications": ["Metformin 1000mg"],
                "symptoms": ["Fatigue"],
                "lab_values": {"HbA1c": 8.0},
            },
        },
        {
            "date": "2024-06-10",
            "extracted": {
                "conditions": ["No diabetes"],
                "medications": ["No medications"],
                "symptoms": [],
                "lab_values": {},
            },
        },
    ]

    history = simulate_patient_evolution("CORE_OPS_TEST", visits)
    graph = PatientKnowledgeGraph.load("data/graphs/CORE_OPS_TEST_graph.json")

    add_count = 0
    update_count = 0
    conflict_count = 0
    for visit in history:
        for result in visit.get("results", []):
            op = str(result.get("operation", "")).upper()
            if op == "ADD":
                add_count += 1
            elif op == "UPDATE":
                update_count += 1
            elif op == "CONFLICT":
                conflict_count += 1

    print("\nCore operation totals:")
    print(f"  ADD: {add_count}")
    print(f"  UPDATE: {update_count}")
    print(f"  CONFLICT: {conflict_count}")
    print(f"  Graph conflicts flagged: {len(graph.get_conflicts())}")

    return {
        "add": add_count,
        "update": update_count,
        "conflict": conflict_count,
        "graph_conflicts": len(graph.get_conflicts()),
    }


def test_conflict_detection():
    """Test conflict detection during evolution."""
    
    print("\n" + "="*60)
    print("🔍 TESTING CONFLICT DETECTION")
    print("="*60)
    
    # Create a visit with conflicting information
    conflicting_visits = [
        {
            "date": "2024-01-15",
            "extracted": {
                "conditions": ["Type 2 Diabetes"],
                "medications": ["Metformin 500mg"],
                "symptoms": ["Fatigue"],
                "lab_values": {"HbA1c": 7.2}
            }
        },
        {
            "date": "2024-06-15",
            "extracted": {
                "conditions": ["No diabetes"],  # Conflict!
                "medications": ["No medications"],  # Conflict!
                "symptoms": [],
                "lab_values": {}
            }
        }
    ]
    
    print("\n📋 Simulating conflicting visits...")
    history = simulate_patient_evolution("CONFLICT_TEST", conflicting_visits)
    
    # Check for conflicts
    graph = PatientKnowledgeGraph.load("data/graphs/CONFLICT_TEST_graph.json")
    conflicts = graph.get_conflicts()
    
    print(f"\n📊 Conflicts detected: {len(conflicts)}")
    for conflict in conflicts:
        print(f"   - {conflict['name']} ({conflict['type']}): {conflict.get('conflict_note', 'Unknown')}")
    
    return conflicts


def test_lab_trend_tracking():
    """Test lab value trend tracking."""
    
    print("\n" + "="*60)
    print("📈 TESTING LAB TREND TRACKING")
    print("="*60)
    
    # Create visits with trending lab values
    trend_visits = [
        {
            "date": "2024-01-15",
            "extracted": {
                "conditions": ["Diabetes"],
                "lab_values": {"HbA1c": 7.2, "eGFR": 72}
            }
        },
        {
            "date": "2024-04-15",
            "extracted": {
                "lab_values": {"HbA1c": 7.8, "eGFR": 68}
            }
        },
        {
            "date": "2024-07-15",
            "extracted": {
                "lab_values": {"HbA1c": 8.1, "eGFR": 55}
            }
        },
        {
            "date": "2024-10-15",
            "extracted": {
                "lab_values": {"HbA1c": 7.9, "eGFR": 50}
            }
        }
    ]
    
    print("\n📋 Simulating lab value trends...")
    history = simulate_patient_evolution("TREND_TEST", trend_visits)
    
    # Show trends
    graph = PatientKnowledgeGraph.load("data/graphs/TREND_TEST_graph.json")
    
    print("\n📊 Lab Trends:")
    for lab in ["HbA1c", "eGFR"]:
        trend = graph.get_lab_trend(lab)
        if trend:
            values = [f"{t['value']}" for t in trend]
            print(f"   {lab}: {' → '.join(values)}")
    
    return graph


def test_real_patient_evolution():
    """Test evolution on a real patient from our dataset."""
    
    print("\n" + "="*60)
    print("🏥 TESTING REAL PATIENT EVOLUTION")
    print("="*60)
    
    # Load a real patient's data
    patient_id = "P001"
    processed_path = Path(f"data/patients_processed/{patient_id}.json")
    
    if not processed_path.exists():
        print(f"❌ Patient {patient_id} data not found")
        return
    
    with open(processed_path, 'r') as f:
        patient_data = json.load(f)
    
    visits = patient_data.get('visits', [])
    print(f"\n📋 Patient {patient_id} has {len(visits)} visits")
    
    # Simulate evolution
    history = simulate_patient_evolution(patient_id, visits)
    
    # Show final summary
    final_graph = PatientKnowledgeGraph.load(f"data/graphs/{patient_id}_graph.json")
    summary = final_graph.summary()
    
    print(f"\n📊 Final Graph Summary for {patient_id}:")
    print(f"   Total Nodes: {summary['total_nodes']}")
    print(f"   Total Edges: {summary['total_edges']}")
    print(f"   Conflicts: {summary['conflicts']}")
    
    # Show conflicts if any
    conflicts = final_graph.get_conflicts()
    if conflicts:
        print(f"\n⚠️ Conflicts detected:")
        for conflict in conflicts:
            print(f"   - {conflict['name']}: {conflict.get('conflict_note', '')[:100]}")
    
    return history


def main():
    """Main execution."""
    
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 8: GRAPH EVOLUTION")
    print("="*60)
    print("Temporal evolution, conflict detection, and confidence decay")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Test core operations (ADD/UPDATE/CONFLICT)")
    print("  2. Test conflict detection")
    print("  3. Test lab trend tracking")
    print("  4. Test real patient evolution (P001)")
    print("  5. Run all tests")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        test_core_operations()

    elif choice == "2":
        test_conflict_detection()
        
    elif choice == "3":
        test_lab_trend_tracking()
        
    elif choice == "4":
        test_real_patient_evolution()
        
    elif choice == "5":
        print("\nRunning all tests...")
        test_core_operations()
        test_conflict_detection()
        test_lab_trend_tracking()
        test_real_patient_evolution()
        
    elif choice == "6":
        print("Exiting...")
        
    else:
        print("Invalid choice. Running all tests...")
        test_core_operations()
        test_conflict_detection()
        test_lab_trend_tracking()
        test_real_patient_evolution()


if __name__ == "__main__":
    main()