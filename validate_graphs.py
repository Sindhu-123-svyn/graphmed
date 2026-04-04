"""
Validate all patient knowledge graphs
"""

import json
from pathlib import Path
from src.graph import PatientKnowledgeGraph

def validate_all_graphs():
    """Validate and display summary for all patient graphs."""
    
    graphs_dir = Path("data/graphs")
    
    if not graphs_dir.exists():
        print("❌ graphs directory not found")
        return
    
    graph_files = sorted(graphs_dir.glob("*_graph.json"))
    
    if not graph_files:
        print("❌ No graph files found")
        return
    
    print("\n" + "="*80)
    print("📊 PATIENT KNOWLEDGE GRAPHS - VALIDATION REPORT")
    print("="*80)
    
    total_stats = {
        'patients': 0,
        'total_nodes': 0,
        'total_edges': 0,
        'total_conditions': 0,
        'total_medications': 0,
        'total_symptoms': 0,
        'total_labs': 0
    }
    
    all_patients_data = []
    
    for graph_file in graph_files:
        try:
            # Load graph
            pkg = PatientKnowledgeGraph.load(str(graph_file))
            summary = pkg.summary()
            
            # Count entity types
            conditions = pkg.get_entities_by_type('CONDITION')
            medications = pkg.get_entities_by_type('MEDICATION')
            symptoms = pkg.get_entities_by_type('SYMPTOM')
            labs = pkg.get_entities_by_type('LAB_VALUE')
            
            patient_data = {
                'patient_id': pkg.patient_id,
                'total_nodes': summary['total_nodes'],
                'total_edges': summary['total_edges'],
                'conditions': len(conditions),
                'medications': len(medications),
                'symptoms': len(symptoms),
                'labs': len(labs),
                'condition_list': [data.get('name') for node, data in conditions],
                'medication_list': [data.get('name') for node, data in medications[:5]],  # First 5
                'symptom_list': [data.get('name') for node, data in symptoms],
                'has_conflicts': summary['conflicts'] > 0
            }
            
            all_patients_data.append(patient_data)
            
            # Update totals
            total_stats['patients'] += 1
            total_stats['total_nodes'] += summary['total_nodes']
            total_stats['total_edges'] += summary['total_edges']
            total_stats['total_conditions'] += len(conditions)
            total_stats['total_medications'] += len(medications)
            total_stats['total_symptoms'] += len(symptoms)
            total_stats['total_labs'] += len(labs)
            
        except Exception as e:
            print(f"❌ Error loading {graph_file.name}: {e}")
    
    # Display individual patient summaries
    print("\n📋 INDIVIDUAL PATIENT SUMMARIES")
    print("-"*80)
    print(f"{'Patient':<10} {'Nodes':<6} {'Edges':<6} {'Conditions':<10} {'Meds':<8} {'Symptoms':<10} {'Labs':<6}")
    print("-"*80)
    
    for data in all_patients_data:
        print(f"{data['patient_id']:<10} "
              f"{data['total_nodes']:<6} "
              f"{data['total_edges']:<6} "
              f"{data['conditions']:<10} "
              f"{data['medications']:<8} "
              f"{data['symptoms']:<10} "
              f"{data['labs']:<6}")
    
    # Display totals
    print("-"*80)
    print(f"{'TOTAL':<10} "
          f"{total_stats['total_nodes']:<6} "
          f"{total_stats['total_edges']:<6} "
          f"{total_stats['total_conditions']:<10} "
          f"{total_stats['total_medications']:<8} "
          f"{total_stats['total_symptoms']:<10} "
          f"{total_stats['total_labs']:<6}")
    
    # Display detailed view for selected patients
    print("\n" + "="*80)
    print("🔍 DETAILED VIEW - SELECTED PATIENTS")
    print("="*80)
    
    # Show P001 details
    p001_data = next((p for p in all_patients_data if p['patient_id'] == 'P001'), None)
    if p001_data:
        print("\n📌 PATIENT P001:")
        print(f"   Total Nodes: {p001_data['total_nodes']}")
        print(f"   Total Edges: {p001_data['total_edges']}")
        print(f"   Conditions ({p001_data['conditions']}): {', '.join(p001_data['condition_list'])}")
        print(f"   Medications ({p001_data['medications']}): {', '.join(p001_data['medication_list'])}")
        if p001_data['symptom_list']:
            print(f"   Symptoms: {', '.join(p001_data['symptom_list'])}")
    
    # Show patient with most conditions
    if all_patients_data:
        max_conditions = max(all_patients_data, key=lambda x: x['conditions'])
        print(f"\n📌 PATIENT WITH MOST CONDITIONS: {max_conditions['patient_id']}")
        print(f"   Conditions ({max_conditions['conditions']}): {', '.join(max_conditions['condition_list'])}")
    
    # Show patient with most medications
    max_meds = max(all_patients_data, key=lambda x: x['medications'])
    print(f"\n📌 PATIENT WITH MOST MEDICATIONS: {max_meds['patient_id']}")
    print(f"   Medications ({max_meds['medications']}): {', '.join(max_meds['medication_list'])}")
    
    # Show patients with symptoms
    patients_with_symptoms = [p for p in all_patients_data if p['symptoms'] > 0]
    if patients_with_symptoms:
        print(f"\n📌 PATIENTS WITH SYMPTOMS ({len(patients_with_symptoms)}):")
        for p in patients_with_symptoms:
            print(f"   {p['patient_id']}: {', '.join(p['symptom_list'])}")
    
    # Overall summary
    print("\n" + "="*80)
    print("📈 OVERALL SUMMARY")
    print("="*80)
    print(f"Total Patients: {total_stats['patients']}")
    print(f"Total Nodes: {total_stats['total_nodes']}")
    print(f"Total Edges: {total_stats['total_edges']}")
    print(f"Average Nodes per Patient: {total_stats['total_nodes']/total_stats['patients']:.1f}")
    print(f"Average Edges per Patient: {total_stats['total_edges']/total_stats['patients']:.1f}")
    
    print(f"\nEntity Distribution:")
    print(f"  🏥 Conditions: {total_stats['total_conditions']}")
    print(f"  💊 Medications: {total_stats['total_medications']}")
    print(f"  🤒 Symptoms: {total_stats['total_symptoms']}")
    print(f"  🔬 Lab Values: {total_stats['total_labs']}")
    
    return all_patients_data, total_stats

def show_patient_graph_details(patient_id: str = "P001"):
    """Show detailed graph structure for a specific patient."""
    
    graph_path = Path(f"data/graphs/{patient_id}_graph.json")
    
    if not graph_path.exists():
        print(f"❌ Graph for {patient_id} not found")
        return
    
    pkg = PatientKnowledgeGraph.load(str(graph_path))
    
    print(f"\n{'='*80}")
    print(f"🔬 DETAILED GRAPH STRUCTURE: {patient_id}")
    print(f"{'='*80}")
    
    # Get all entities by type
    conditions = pkg.get_entities_by_type('CONDITION')
    medications = pkg.get_entities_by_type('MEDICATION')
    symptoms = pkg.get_entities_by_type('SYMPTOM')
    labs = pkg.get_entities_by_type('LAB_VALUE')
    
    print(f"\n🏥 CONDITIONS ({len(conditions)}):")
    for node, data in conditions:
        print(f"   • {data.get('name')} (added: {data.get('added')}, confidence: {data.get('confidence', 1.0):.2f})")
    
    print(f"\n💊 MEDICATIONS ({len(medications)}):")
    for node, data in medications[:10]:  # Show first 10
        print(f"   • {data.get('name')} (added: {data.get('added')})")
    if len(medications) > 10:
        print(f"   ... and {len(medications) - 10} more")
    
    print(f"\n🤒 SYMPTOMS ({len(symptoms)}):")
    for node, data in symptoms:
        print(f"   • {data.get('name')} (added: {data.get('added')})")
    
    print(f"\n🔬 LAB VALUES ({len(labs)}):")
    for node, data in labs:
        value = data.get('value', 'N/A')
        print(f"   • {data.get('name')}: {value} (added: {data.get('added')})")
    
    # Show some relationships
    print(f"\n🔗 SAMPLE RELATIONSHIPS:")
    edges_shown = 0
    for src, dst, data in list(pkg.G.edges(data=True))[:10]:
        src_name = pkg.G.nodes[src].get('name', src)
        dst_name = pkg.G.nodes[dst].get('name', dst)
        relation = data.get('relation', 'unknown')
        print(f"   • {src_name} --{relation}--> {dst_name}")
    if pkg.G.number_of_edges() > 10:
        print(f"   ... and {pkg.G.number_of_edges() - 10} more relationships")

def main():
    """Main validation function."""
    
    print("\n" + "="*80)
    print("🏥 GRAPHMED - KNOWLEDGE GRAPH VALIDATION")
    print("="*80)
    
    print("\nOptions:")
    print("  1. Show summary of all 10 patients")
    print("  2. Show detailed graph for P001")
    print("  3. Show detailed graph for specific patient")
    print("  4. Show all graphs (complete validation)")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        validate_all_graphs()
        
    elif choice == "2":
        validate_all_graphs()
        show_patient_graph_details("P001")
        
    elif choice == "3":
        patient_id = input("Enter patient ID (e.g., P001): ").strip().upper()
        validate_all_graphs()
        show_patient_graph_details(patient_id)
        
    elif choice == "4":
        # Full validation with all details
        data, stats = validate_all_graphs()
        
        # Show detailed view for all patients with conditions
        print("\n" + "="*80)
        print("🔬 ALL PATIENTS - CONDITIONS DETAIL")
        print("="*80)
        for patient in data:
            if patient['condition_list']:
                print(f"\n{patient['patient_id']}: {', '.join(patient['condition_list'])}")
        
        # Show detailed view for P001 specifically
        show_patient_graph_details("P001")
        
    elif choice == "5":
        print("Exiting...")
        
    else:
        print("Invalid choice. Showing summary...")
        validate_all_graphs()

if __name__ == "__main__":
    main()