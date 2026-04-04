"""
Phase 2: Process all patients through extraction pipeline
"""

import os
import json
from pathlib import Path
from src.extraction import process_patient_visits, extract_entities
import time

def process_all_patients(input_dir: str = "data/patients", 
                         output_dir: str = "data/patients_processed",
                         use_llm: bool = True,
                         limit: int = None,
                         rate_limit_sleep: float = 0.5):
    """
    Process all patient files and add extracted entities.
    
    Args:
        input_dir: Directory with raw patient JSON files
        output_dir: Directory to save processed files
        use_llm: Whether to use LLM for extraction
        limit: Maximum number of patients to process
        rate_limit_sleep: Delay between patients when using LLM
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all patient files
    patient_files = sorted(Path(input_dir).glob("*.json"))
    
    if limit is not None:
        patient_files = patient_files[:limit]

    if not patient_files:
        print(f"No patient files found in {input_dir}")
        return {
            "processed": 0,
            "total": 0,
            "errors": [],
            "total_relationships": 0,
            "total_procedures": 0
        }
    
    print(f"\n{'='*60}")
    print(f"📊 PHASE 2: NLP EXTRACTION PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Patients: {len(patient_files)}")
    print(f"Using LLM: {use_llm}")
    print(f"{'='*60}\n")
    
    processed = 0
    errors = []
    total_relationships = 0
    total_procedures = 0
    
    for file_path in patient_files:
        try:
            # Load patient data
            with open(file_path, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            print(f"Processing {patient_data['patient_id']}...", end=" ")
            
            # Process visits
            processed_data = process_patient_visits(patient_data, use_llm=use_llm)
            
            # Save processed data
            output_path = Path(output_dir) / file_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            processed += 1
            relationship_count = sum(len(v.get('extracted', {}).get('relationships', [])) for v in processed_data['visits'])
            procedure_count = sum(len(v.get('extracted', {}).get('procedures', [])) for v in processed_data['visits'])
            total_relationships += relationship_count
            total_procedures += procedure_count
            print(
                f"✅ ({len(processed_data['visits'])} visits, "
                f"{sum(len(v.get('extracted', {}).get('conditions', [])) for v in processed_data['visits'])} conditions, "
                f"{relationship_count} relationships)"
            )
            
            # Small delay to respect rate limits if using LLM
            if use_llm:
                time.sleep(rate_limit_sleep)
                
        except Exception as e:
            errors.append((file_path.name, str(e)))
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"📈 EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {processed}/{len(patient_files)}")
    print(f"❌ Errors: {len(errors)}")
    print(f"🔗 Total relationships extracted: {total_relationships}")
    print(f"🛠️  Total procedures extracted: {total_procedures}")
    if errors:
        print("Errors:")
        for err_file, err_msg in errors[:5]:
            print(f"  - {err_file}: {err_msg}")
    print(f"📁 Location: {output_dir}")
    print(f"{'='*60}")

    return {
        "processed": processed,
        "total": len(patient_files),
        "errors": errors,
        "total_relationships": total_relationships,
        "total_procedures": total_procedures
    }


def quick_test_extraction(use_llm: bool = False):
    """Quick extraction smoke test with a representative clinical note."""
    note = (
        "Patient with Type 2 Diabetes on Metformin 500mg reports fatigue and shortness "
        "of breath. HbA1c 8.1 and BP 150/95. Plan to increase metformin and monitor kidneys."
    )
    print("\nQuick test note:")
    print(f"  {note}")
    result = extract_entities(note, use_llm=use_llm)
    print("\nExtraction result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

def show_sample_extraction(patient_id: str = "P001"):
    """Show a sample extraction result."""
    
    # Try processed first, then raw
    file_path = Path(f"data/patients_processed/{patient_id}.json")
    if not file_path.exists():
        file_path = Path(f"data/patients/{patient_id}.json")
    
    if not file_path.exists():
        print(f"❌ Patient {patient_id} not found")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Sample Extraction: {data['patient_id']}")
    print(f"{'='*60}")
    
    for visit in data['visits']:
        print(f"\n📅 {visit['visit_id']} - {visit['date']}")
        print(f"   Note: {visit['note'][:100]}...")
        
        # Show extracted data
        if 'extracted' in visit:
            ext = visit['extracted']
            if ext.get('conditions'):
                print(f"   🏥 Conditions: {', '.join(ext['conditions'][:3])}")
            if ext.get('medications'):
                print(f"   💊 Medications: {', '.join(ext['medications'][:3])}")
            if ext.get('lab_values'):
                print(f"   🔬 Labs: {ext['lab_values']}")
            if ext.get('symptoms'):
                print(f"   🤒 Symptoms: {', '.join(ext['symptoms'][:3])}")
            if ext.get('relationships'):
                print(f"   🔗 Relationships: {ext['relationships'][:2]}")
        else:
            # Show original fields
            if visit.get('diagnoses'):
                print(f"   🏥 Conditions: {', '.join(visit['diagnoses'][:3])}")
            if visit.get('medications'):
                print(f"   💊 Medications: {', '.join(visit['medications'][:3])}")
            if visit.get('labs'):
                print(f"   🔬 Labs: {visit['labs']}")
            if visit.get('symptoms'):
                print(f"   🤒 Symptoms: {', '.join(visit['symptoms'][:3])}")

def main():
    """Main execution for Phase 2."""
    
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 2: NLP EXTRACTION PIPELINE")
    print("="*60)
    print("Extracting structured entities from clinical notes")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Quick test on sample notes")
    print("  2. Process all patients (no LLM - fast, rule-based)")
    print("  3. Process all patients (with LLM - slower, more accurate)")
    print("  4. Process first 10 patients (test batch)")
    print("  5. Show sample extraction for P001")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        print("\nRunning quick extraction test (rule-based)...")
        quick_test_extraction(use_llm=False)
        
    elif choice == "2":
        confirm = input("\nProcess all patients with rule-based extraction? (y/n): ")
        if confirm.lower() == 'y':
            process_all_patients(use_llm=False)
            show_sample_extraction()
        else:
            print("Cancelled.")
            
    elif choice == "3":
        print("\n⚠️  Using LLM will be slower and use API credits")
        confirm = input("Process all patients with LLM? (y/n): ")
        if confirm.lower() == 'y':
            process_all_patients(use_llm=True)
            show_sample_extraction()
        else:
            print("Cancelled.")
            
    elif choice == "4":
        print("\nProcessing first 10 patients (good for testing)...")
        process_all_patients(limit=10, use_llm=False)
        show_sample_extraction()
        
    elif choice == "5":
        show_sample_extraction()
        
    elif choice == "6":
        print("Exiting...")
        
    else:
        print("Invalid choice. Running test...")
        quick_test_extraction(use_llm=False)

if __name__ == "__main__":
    main()