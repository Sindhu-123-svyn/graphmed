"""Validate Phase 1 generated data."""

import os
import json
from pathlib import Path

def validate_patient_data():
    """Check all patient files for completeness."""
    
    patients_dir = Path("data/patients")
    
    if not patients_dir.exists():
        print("❌ data/patients/ directory not found")
        return False
    
    patient_files = list(patients_dir.glob("*.json"))
    
    if not patient_files:
        print("❌ No patient JSON files found")
        return False
    
    print(f"\n📊 Validating {len(patient_files)} patient files...\n")
    
    valid = 0
    invalid = []
    
    for file in patient_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required fields
            if 'patient_id' not in data:
                invalid.append(f"{file.name}: missing patient_id")
                continue
                
            if 'visits' not in data:
                invalid.append(f"{file.name}: missing visits")
                continue
                
            visits = data['visits']
            if len(visits) < 3:
                invalid.append(f"{file.name}: only {len(visits)} visits (need 3+)")
                continue
            
            # Check each visit
            for i, visit in enumerate(visits):
                required = ['visit_id', 'date', 'note', 'labs', 'medications', 'diagnoses', 'symptoms']
                for field in required:
                    if field not in visit:
                        invalid.append(f"{file.name}: visit {i+1} missing {field}")
                        break
            
            valid += 1
            
        except json.JSONDecodeError:
            invalid.append(f"{file.name}: invalid JSON")
        except Exception as e:
            invalid.append(f"{file.name}: {e}")
    
    print(f"✅ Valid: {valid}/{len(patient_files)}")
    
    if invalid:
        print(f"\n⚠️  Issues found ({len(invalid)}):")
        for issue in invalid[:10]:
            print(f"   - {issue}")
    
    return valid > 0

if __name__ == "__main__":
    validate_patient_data()