"""
Debug extraction to see what entities are being captured
"""

import json
from pathlib import Path
from src.extraction import extract_entities

def debug_patient_extraction(patient_id: str = "P001"):
    """Debug extraction for a specific patient."""
    
    # Load patient data
    patient_path = Path(f"data/patients_processed/{patient_id}.json")
    
    if not patient_path.exists():
        print(f"❌ Patient {patient_id} not found")
        return
    
    with open(patient_path, 'r', encoding='utf-8') as f:
        patient_data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Debugging Extraction for {patient_id}")
    print(f"{'='*60}\n")
    
    for visit in patient_data['visits']:
        print(f"\n📅 {visit['visit_id']} - {visit['date']}")
        print(f"Note: {visit['note'][:150]}...")
        
        # Check what's in the note
        note = visit['note']
        
        # Manual check for key terms
        print(f"\n  Manual check for key terms:")
        print(f"    'diabetes' present: {'diabetes' in note.lower()}")
        print(f"    'hypertension' present: {'hypertension' in note.lower()}")
        print(f"    'fatigue' present: {'fatigue' in note.lower()}")
        print(f"    'HbA1c' present: {'hba1c' in note.lower()}")
        
        # Show what was extracted
        if 'extracted' in visit:
            ext = visit['extracted']
            print(f"\n  Extracted (from file):")
            print(f"    Conditions: {ext.get('conditions', [])}")
            print(f"    Medications: {ext.get('medications', [])}")
            print(f"    Lab values: {ext.get('lab_values', {})}")
            print(f"    Symptoms: {ext.get('symptoms', [])}")
        else:
            print(f"\n  ⚠️ No 'extracted' field found in this visit")
            
            # Run extraction fresh
            print(f"\n  Running extraction now:")
            fresh_extract = extract_entities(note, use_llm=False)
            print(f"    Conditions: {fresh_extract.get('conditions', [])}")
            print(f"    Medications: {fresh_extract.get('medications', [])}")
            print(f"    Lab values: {fresh_extract.get('lab_values', {})}")
            print(f"    Symptoms: {fresh_extract.get('symptoms', [])}")

if __name__ == "__main__":
    debug_patient_extraction("P001")