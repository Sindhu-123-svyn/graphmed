"""
Phase 3: Build knowledge graphs for all patients
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import build_graphs_for_all_patients, PatientKnowledgeGraph
import json

def show_graph_summary(patient_id: str = "P001"):
    """Show summary of a built graph."""
    
    graph_path = Path(f"data/graphs/{patient_id}_graph.json")
    
    if not graph_path.exists():
        print(f"❌ Graph for {patient_id} not found")
        return
    
    # Load graph
    pkg = PatientKnowledgeGraph.load(str(graph_path))
    
    # Show summary
    print(f"\n{'='*60}")
    print(f"Graph Summary: {patient_id}")
    print(f"{'='*60}")
    
    summary = pkg.summary()
    print(f"Total Nodes: {summary['total_nodes']}")
    print(f"Total Edges: {summary['total_edges']}")
    print(f"Conflicts: {summary['conflicts']}")
    
    print(f"\nNode Types:")
    for node_type, count in summary['node_types'].items():
        print(f"  {node_type}: {count}")
    
    # Show sample entities
    print(f"\nSample Entities:")
    conditions = pkg.get_entities_by_type('CONDITION')
    for node, data in conditions[:5]:
        print(f"  🏥 {data.get('name')} (added: {data.get('added')}, confidence: {data.get('confidence'):.2f})")
    
    medications = pkg.get_entities_by_type('MEDICATION')
    for node, data in medications[:5]:
        print(f"  💊 {data.get('name')} (added: {data.get('added')})")
    
    # Show lab trends if available
    labs = pkg.get_entities_by_type('LAB_VALUE')
    if labs:
        print(f"\nLab Value Trends:")
        for node, data in labs[:3]:
            lab_name = data.get('name')
            trend = pkg.get_lab_trend(lab_name)
            if trend:
                values = [f"{t['value']}" for t in trend[-3:]]
                print(f"  🔬 {lab_name}: {' → '.join(values)}")
    
    return pkg

def main():
    """Main execution for Phase 3."""
    
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 3: KNOWLEDGE GRAPH CONSTRUCTION")
    print("="*60)
    print("Building temporally-aware patient knowledge graphs")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Build graphs for all processed patients (10 patients)")
    print("  2. Build graphs for first 5 patients (test)")
    print("  3. Show summary for P001")
    print("  4. Show summary for all patients")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        confirm = input("\nBuild graphs for all 10 patients? (y/n): ")
        if confirm.lower() == 'y':
            build_graphs_for_all_patients(limit=10)
            show_graph_summary("P001")
        else:
            print("Cancelled.")
            
    elif choice == "2":
        print("\nBuilding graphs for first 5 patients...")
        build_graphs_for_all_patients(limit=5)
        show_graph_summary("P001")
        
    elif choice == "3":
        show_graph_summary("P001")
        
    elif choice == "4":
        print("\nShowing summaries for all patients...")
        graph_files = list(Path("data/graphs").glob("*_graph.json"))
        for gf in graph_files:
            patient_id = gf.stem.replace("_graph", "")
            show_graph_summary(patient_id)
            print("-" * 40)
            
    elif choice == "5":
        print("Exiting...")
        
    else:
        print("Invalid choice. Building graphs for first 5 patients...")
        build_graphs_for_all_patients(limit=5)

if __name__ == "__main__":
    main()