"""
Phase 1: Synthetic Patient Data Generation for GraphMed
Generates realistic patient timelines with 3+ visits each
"""

import os
import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Base conditions to work with
CONDITIONS = [
    "Type 2 Diabetes",
    "Hypertension",
    "Hyperlipidemia",
    "Coronary Artery Disease",
    "Chronic Kidney Disease",
    "Heart Failure",
    "COPD",
    "Asthma",
    "Osteoarthritis",
    "Depression"
]

def generate_patient_timeline(patient_id: str, num_visits: int = 4) -> Dict[str, Any]:
    """
    Generate a complete patient timeline using Groq LLM.
    """
    
    prompt = f"""Generate a realistic synthetic patient medical timeline with EXACTLY {num_visits} visits.

Patient ID: {patient_id}
Number of visits: {num_visits} spanning 18 months

CRITICAL: Return ONLY valid JSON. No explanation, no markdown, no text outside JSON.

Required JSON structure:
{{
    "patient_id": "{patient_id}",
    "visits": [
        {{
            "visit_id": "V1",
            "date": "YYYY-MM-DD",
            "note": "Detailed clinical note with symptoms, examination findings, and plan (3-5 sentences)",
            "labs": {{
                "HbA1c": 7.2,
                "BP_systolic": 140,
                "BP_diastolic": 90,
                "eGFR": 72,
                "LDL": 130
            }},
            "medications": ["Medication1 dose", "Medication2 dose"],
            "diagnoses": ["Diagnosis1", "Diagnosis2"],
            "symptoms": ["Symptom1", "Symptom2"]
        }}
    ]
}}

Requirements:
1. Patient should have 2-3 chronic conditions (choose from: {', '.join(CONDITIONS[:5])})
2. Lab values should show realistic progression (worsening or improving) across visits
3. Medications should adjust based on lab trends
4. Add new symptoms/diagnoses over time as complications develop
5. Make each visit clinically realistic and coherent
6. Dates should be sequential, roughly 4-6 months apart
7. Include realistic lab values (HbA1c 6.0-9.0, BP 120-160/70-100, eGFR 45-90)

Generate for patient {patient_id} with {num_visits} visits over 18 months.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        
        # Extract JSON from response
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = result[start_idx:end_idx]
            patient_data = json.loads(json_str)
            return patient_data
        else:
            print(f"  ⚠️ No JSON found for {patient_id}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"  ⚠️ JSON parse error for {patient_id}: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️ API error for {patient_id}: {e}")
        return None

def validate_timeline(timeline: Dict[str, Any]) -> bool:
    """
    Validate that the generated timeline has required fields.
    """
    if not timeline:
        return False
    
    if 'patient_id' not in timeline:
        return False
    
    if 'visits' not in timeline or len(timeline['visits']) < 3:
        return False
    
    required_fields = ['visit_id', 'date', 'note', 'labs', 'medications', 'diagnoses', 'symptoms']
    
    for visit in timeline['visits']:
        for field in required_fields:
            if field not in visit:
                return False
    
    return True

def save_patient_data(patient_data: Dict[str, Any], output_dir: str = "data/patients") -> bool:
    """
    Save patient data to JSON file.
    """
    if not patient_data:
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{patient_data['patient_id']}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(patient_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"  ❌ Error saving {filename}: {e}")
        return False

def generate_batch_patients(
    num_patients: int = 50,
    min_visits: int = 3,
    max_visits: int = 5,
    delay_seconds: float = 0.5
) -> tuple:
    """
    Generate a batch of synthetic patients.
    
    Returns:
        tuple: (successful_count, failed_ids)
    """
    
    print(f"\n{'='*60}")
    print(f"📊 GENERATING {num_patients} SYNTHETIC PATIENTS")
    print(f"{'='*60}")
    print(f"Visits per patient: {min_visits}-{max_visits}")
    print(f"Time range: 18 months per patient")
    print(f"{'='*60}\n")
    
    successful = 0
    failed = []
    
    for i in range(1, num_patients + 1):
        patient_id = f"P{i:03d}"
        num_visits = random.randint(min_visits, max_visits)
        
        print(f"[{i}/{num_patients}] Generating {patient_id} ({num_visits} visits)...", end=" ")
        
        timeline = generate_patient_timeline(patient_id, num_visits)
        
        if timeline and validate_timeline(timeline):
            if save_patient_data(timeline):
                successful += 1
                print(f"✅ Saved")
            else:
                failed.append(patient_id)
                print(f"❌ Save failed")
        else:
            failed.append(patient_id)
            print(f"❌ Generation failed")
        
        # Rate limiting delay
        if i < num_patients:
            time.sleep(delay_seconds)
    
    print(f"\n{'='*60}")
    print(f"📈 GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{num_patients}")
    print(f"❌ Failed: {len(failed)}")
    if failed:
        print(f"   Failed IDs: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    print(f"📁 Location: data/patients/")
    print(f"{'='*60}")
    
    return successful, failed

def create_sample_patient() -> bool:
    """
    Create a handcrafted sample patient for testing.
    """
    
    sample = {
        "patient_id": "P001",
        "visits": [
            {
                "visit_id": "V1",
                "date": "2024-01-15",
                "note": "Patient presents with fatigue and increased thirst. Newly diagnosed with Type 2 Diabetes. Started on Metformin 500mg daily. Blood pressure elevated at 140/90. Will monitor blood glucose and check HbA1c in 3 months. Discussed diet and exercise modifications.",
                "labs": {
                    "HbA1c": 7.2,
                    "BP_systolic": 140,
                    "BP_diastolic": 90,
                    "eGFR": 72,
                    "LDL": 130
                },
                "medications": ["Metformin 500mg daily"],
                "diagnoses": ["Type 2 Diabetes", "Hypertension"],
                "symptoms": ["Fatigue", "Polydipsia", "Polyuria"]
            },
            {
                "visit_id": "V2",
                "date": "2024-04-20",
                "note": "Follow-up visit. HbA1c increased to 7.8 despite medication. Patient reports inconsistent medication adherence. Increased Metformin to 1000mg daily. Reinforced lifestyle modifications. Blood pressure remains elevated. Will add Lisinopril if not controlled next visit.",
                "labs": {
                    "HbA1c": 7.8,
                    "BP_systolic": 142,
                    "BP_diastolic": 88,
                    "eGFR": 70,
                    "LDL": 135
                },
                "medications": ["Metformin 1000mg daily"],
                "diagnoses": ["Type 2 Diabetes", "Hypertension"],
                "symptoms": ["Fatigue"]
            },
            {
                "visit_id": "V3",
                "date": "2024-08-10",
                "note": "HbA1c now 8.1. Started Lisinopril 10mg for hypertension. Patient reports tingling in feet - possible diabetic neuropathy. Started Gabapentin 300mg. Referred to podiatry. Discussed insulin as next step if HbA1c not controlled.",
                "labs": {
                    "HbA1c": 8.1,
                    "BP_systolic": 135,
                    "BP_diastolic": 85,
                    "eGFR": 68,
                    "LDL": 128
                },
                "medications": ["Metformin 1000mg daily", "Lisinopril 10mg daily", "Gabapentin 300mg nightly"],
                "diagnoses": ["Type 2 Diabetes", "Hypertension", "Diabetic Neuropathy"],
                "symptoms": ["Tingling in feet"]
            },
            {
                "visit_id": "V4",
                "date": "2024-12-05",
                "note": "Neuropathy symptoms improving with Gabapentin. HbA1c improved to 7.9 with good medication adherence. Blood pressure controlled at 130/82. Will continue current regimen. Patient motivated and attending diabetes education classes.",
                "labs": {
                    "HbA1c": 7.9,
                    "BP_systolic": 130,
                    "BP_diastolic": 82,
                    "eGFR": 65,
                    "LDL": 125
                },
                "medications": ["Metformin 1000mg daily", "Lisinopril 10mg daily", "Gabapentin 300mg nightly"],
                "diagnoses": ["Type 2 Diabetes", "Hypertension", "Diabetic Neuropathy"],
                "symptoms": []
            }
        ]
    }
    
    os.makedirs("data/patients", exist_ok=True)
    
    with open("data/patients/P001.json", 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample patient P001 created!")
    return True

def display_summary():
    """
    Display summary of generated data.
    """
    patients_dir = "data/patients"
    
    if not os.path.exists(patients_dir):
        print("❌ No patient data found")
        return
    
    patient_files = [f for f in os.listdir(patients_dir) if f.endswith('.json')]
    
    if not patient_files:
        print("❌ No patient JSON files found")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total patients: {len(patient_files)}")
    
    total_visits = 0
    for file in patient_files[:5]:  # Show first 5 samples
        with open(f"{patients_dir}/{file}", 'r') as f:
            data = json.load(f)
            visits = len(data.get('visits', []))
            total_visits += visits
            print(f"  {data['patient_id']}: {visits} visits")
    
    if len(patient_files) > 5:
        print(f"  ... and {len(patient_files)-5} more patients")
    
    print(f"\n📁 Location: {patients_dir}")
    print(f"{'='*60}")

def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 1: DATA PREPARATION")
    print("="*60)
    print("Creating synthetic patient timelines with 3+ visits each")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Create sample patient only (P001) - Quick test")
    print("  2. Generate 10 patients - Small test batch")
    print("  3. Generate 50 patients - Development dataset")
    print("  4. Generate 100 patients - Full dataset")
    print("  5. Show summary of existing data")
    print("  6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        create_sample_patient()
        display_summary()
        
    elif choice == "2":
        generate_batch_patients(num_patients=10, min_visits=3, max_visits=4, delay_seconds=0.5)
        display_summary()
        
    elif choice == "3":
        print("\n⚠️  This will generate 50 patients using Groq API")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            generate_batch_patients(num_patients=50, min_visits=3, max_visits=5, delay_seconds=0.5)
            display_summary()
        else:
            print("Cancelled.")
            
    elif choice == "4":
        print("\n⚠️  This will generate 100 patients using Groq API")
        print("This may take 5-10 minutes")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            generate_batch_patients(num_patients=100, min_visits=3, max_visits=5, delay_seconds=0.5)
            display_summary()
        else:
            print("Cancelled.")
            
    elif choice == "5":
        display_summary()
        
    elif choice == "6":
        print("Exiting...")
        
    else:
        print("Invalid choice. Creating sample patient...")
        create_sample_patient()
        display_summary()

if __name__ == "__main__":
    main()