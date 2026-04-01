"""
Reprocess patients with enhanced extraction
"""

import json
from pathlib import Path
from src.extraction import process_patient_visits

def reprocess_all_patients():
    """Reprocess all patients with enhanced extraction."""
    
    input_dir = Path("data/patients")
    output_dir = Path("data/patients_processed")
    output_dir.mkdir(exist_ok=True)
    
    patient_files = list(input_dir.glob("*.json"))
    
    print(f"\n{'='*60}")
    print("Reprocessing Patients with Enhanced Extraction")
    print(f"{'='*60}")
    print(f"Patients: {len(patient_files)}")
    print(f"{'='*60}\n")
    
    success = 0
    for file_path in patient_files:
        try:
            # Load patient data
            with open(file_path, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            print(f"Processing {patient_data['patient_id']}...", end=" ")
            
            # Process with enhanced extraction
            processed = process_patient_visits(patient_data, use_llm=False)
            
            # Save
            output_path = output_dir / file_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed, f, indent=2, ensure_ascii=False)
            
            # Count extracted entities
            total_conditions = 0
            total_symptoms = 0
            total_labs = 0
            for visit in processed['visits']:
                ext = visit.get('extracted', {})
                total_conditions += len(ext.get('conditions', []))
                total_symptoms += len(ext.get('symptoms', []))
                total_labs += len(ext.get('lab_values', {}))
            
            print(f"✅ ({total_conditions} conditions, {total_symptoms} symptoms, {total_labs} labs)")
            success += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Reprocessed {success}/{len(patient_files)} patients")
    print(f"📁 Location: {output_dir}")

if __name__ == "__main__":
    reprocess_all_patients()