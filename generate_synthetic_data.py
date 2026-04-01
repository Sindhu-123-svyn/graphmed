"""Generate synthetic patient data for GraphMed."""

import os
import json
import random
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_patient_timeline(patient_id, num_visits=4):
    """Generate a synthetic patient timeline using Groq."""
    
    prompt = f"""Generate a realistic synthetic patient medical timeline.
    
Patient ID: {patient_id}
Number of visits: {num_visits} over 18 months

Return ONLY valid JSON with this exact structure:
{{
    "patient_id": "{patient_id}",
    "visits": [
        {{
            "visit_id": "V1",
            "date": "YYYY-MM-DD",
            "note": "Clinical note with symptoms, findings, and plan",
            "labs": {{"HbA1c": 7.2, "BP": "140/90", "eGFR": 72}},
            "medications": ["Metformin 500mg"],
            "diagnoses": ["Type 2 Diabetes", "Hypertension"],
            "symptoms": ["Fatigue", "Polyuria"]
        }}
    ]
}}

Make it realistic with evolving conditions. The patient should have Type 2 Diabetes and Hypertension. Include:
- Progressing lab values over time
- Medication adjustments
- New symptoms appearing
- Possible complications

Generate {num_visits} visits spanning 18 months."""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Parse the response
        result = response.choices[0].message.content
        
        # Extract JSON from response (handle possible text around it)
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            patient_data = json.loads(json_str)
            return patient_data
        else:
            print(f"❌ Could not parse JSON for patient {patient_id}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating patient {patient_id}: {e}")
        return None

def save_patient_data(patient_data, output_dir="data/patients"):
    """Save patient data to JSON file."""
    
    if not patient_data:
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{patient_data['patient_id']}.json"
    
    with open(filename, 'w') as f:
        json.dump(patient_data, f, indent=2)
    
    print(f"✅ Saved: {filename}")
    return True

def generate_batch_patients(num_patients=50, min_visits=3, max_visits=5):
    """Generate a batch of synthetic patients."""
    
    print(f"\n{'='*50}")
    print(f"Generating {num_patients} synthetic patients")
    print(f"{'='*50}\n")
    
    successful = 0
    
    for i in range(1, num_patients + 1):
        patient_id = f"P{i:03d}"
        num_visits = random.randint(min_visits, max_visits)
        
        print(f"Generating {patient_id} ({num_visits} visits)...")
        
        patient_data = generate_patient_timeline(patient_id, num_visits)
        
        if patient_data and save_patient_data(patient_data):
            successful += 1
        else:
            print(f"⚠️  Failed to generate {patient_id}")
        
        # Small delay to avoid rate limits
        if i % 10 == 0:
            print(f"\nProgress: {i}/{num_patients} patients generated\n")
    
    print(f"\n{'='*50}")
    print(f"✅ Generation complete!")
    print(f"Successfully generated: {successful}/{num_patients} patients")
    print(f"{'='*50}")
    
    return successful

def create_sample_patient():
    """Create one sample patient to test the structure."""
    
    sample = {
        "patient_id": "P001",
        "visits": [
            {
                "visit_id": "V1",
                "date": "2024-01-15",
                "note": "Patient presents with fatigue and increased thirst. Newly diagnosed with Type 2 Diabetes. Started on Metformin 500mg daily. Will monitor blood glucose.",
                "labs": {"HbA1c": 7.2, "BP": "140/90", "eGFR": 72},
                "medications": ["Metformin 500mg"],
                "diagnoses": ["Type 2 Diabetes", "Hypertension"],
                "symptoms": ["Fatigue", "Polyuria", "Polydipsia"]
            },
            {
                "visit_id": "V2",
                "date": "2024-04-20",
                "note": "Follow-up. HbA1c increased to 7.8. Patient reports dietary non-compliance. Increased Metformin to 1000mg daily. Reinforced lifestyle modifications.",
                "labs": {"HbA1c": 7.8, "BP": "138/88", "eGFR": 70},
                "medications": ["Metformin 1000mg"],
                "diagnoses": ["Type 2 Diabetes", "Hypertension"],
                "symptoms": ["Fatigue"]
            },
            {
                "visit_id": "V3",
                "date": "2024-08-10",
                "note": "HbA1c now 8.1. Considering add-on therapy. Patient reports tingling in feet - possible neuropathy. Started Gabapentin 300mg for neuropathic pain.",
                "labs": {"HbA1c": 8.1, "BP": "142/92", "eGFR": 68},
                "medications": ["Metformin 1000mg", "Gabapentin 300mg"],
                "diagnoses": ["Type 2 Diabetes", "Hypertension", "Diabetic Neuropathy"],
                "symptoms": ["Fatigue", "Tingling in feet"]
            },
            {
                "visit_id": "V4",
                "date": "2024-12-05",
                "note": "Neuropathy symptoms improving with Gabapentin. HbA1c stable at 7.9. Blood pressure elevated. Added Lisinopril 10mg. Continue current regimen.",
                "labs": {"HbA1c": 7.9, "BP": "135/85", "eGFR": 65},
                "medications": ["Metformin 1000mg", "Gabapentin 300mg", "Lisinopril 10mg"],
                "diagnoses": ["Type 2 Diabetes", "Hypertension", "Diabetic Neuropathy"],
                "symptoms": []
            }
        ]
    }
    
    os.makedirs("data/patients", exist_ok=True)
    with open("data/patients/P001.json", 'w') as f:
        json.dump(sample, f, indent=2)
    
    print("✅ Sample patient P001 created!")
    return True

if __name__ == "__main__":
    print("\n" + "="*50)
    print("GRAPH MED - PHASE 1: DATA PREPARATION")
    print("="*50)
    
    print("\nChoose an option:")
    print("1. Create sample patient only (P001)")
    print("2. Generate 10 synthetic patients (quick test)")
    print("3. Generate 50 synthetic patients (full dataset)")
    print("4. Generate 100 synthetic patients (full dataset)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        create_sample_patient()
    elif choice == "2":
        generate_batch_patients(num_patients=10, min_visits=3, max_visits=4)
    elif choice == "3":
        generate_batch_patients(num_patients=50, min_visits=3, max_visits=5)
    elif choice == "4":
        generate_batch_patients(num_patients=100, min_visits=3, max_visits=5)
    else:
        print("Invalid choice. Creating sample patient...")
        create_sample_patient()
    
    print("\n✅ Phase 1 Data Preparation initiated!")
    print("📁 Check the 'data/patients/' folder for generated files")