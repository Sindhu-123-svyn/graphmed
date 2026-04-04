"""
Phase 7: Generate High-Quality Training Data for Conflict Classifier
Creates realistic clinical statement pairs
"""

import json
import random
from pathlib import Path

# Realistic clinical statements
REALISTIC_STATEMENTS = {
    "allergy": [
        ("No known drug allergies", "Allergy to penicillin"),
        ("No allergies reported", "Sulfa allergy documented"),
        ("Tolerates medications well", "Previous anaphylaxis to cephalosporins"),
        ("No adverse drug reactions", "Developed rash after starting amoxicillin"),
    ],
    "diabetes": [
        ("HbA1c 6.2%, well controlled", "HbA1c 9.8%, poor control"),
        ("Not diabetic", "Type 2 diabetes diagnosed 2 years ago"),
        ("Diet-controlled diabetes", "Started on insulin therapy"),
        ("No history of diabetes", "Random glucose 280 mg/dL"),
    ],
    "hypertension": [
        ("BP 118/76, well controlled", "BP 165/98, poorly controlled"),
        ("No hypertension", "Blood pressure 150/90 on two readings"),
        ("Discontinued lisinopril", "Continue lisinopril 20mg daily"),
    ],
    "medications": [
        ("Not taking any medications", "Prescribed metformin 500mg BID"),
        ("Stopped warfarin", "Continue warfarin with INR monitoring"),
        ("Medication list: none", "Currently on atorvastatin and amlodipine"),
    ],
    "symptoms": [
        ("No chest pain", "Reports chest pressure when walking"),
        ("Denies shortness of breath", "SOB on minimal exertion"),
        ("No complaints of pain", "Severe headache for 3 days"),
    ],
    "labs": [
        ("Creatinine normal", "Creatinine 2.5 mg/dL, elevated"),
        ("eGFR >90", "eGFR 45, stage 3 CKD"),
        ("LDL within target", "LDL 190, very high"),
    ]
}

CONSISTENT_PAIRS = [
    ("Patient has diabetes", "Type 2 diabetes mellitus"),
    ("Taking metformin", "Prescribed metformin 500mg"),
    ("Blood pressure 130/85", "BP 130/85, controlled"),
    ("No fever", "Afebrile"),
    ("HbA1c 7.2%", "A1C 7.2, slightly elevated"),
    ("Chest pain reported", "Complains of chest discomfort"),
    ("On lisinopril", "Taking lisinopril 10mg daily"),
    ("Allergy to penicillin", "Penicillin allergy in chart"),
    ("No known allergies", "No allergies documented"),
    ("Diabetes well-controlled", "Good glycemic control"),
    ("Hypertension diagnosed", "High blood pressure history"),
    ("Discontinued statin", "Stopped taking atorvastatin"),
]

CONFLICT_PAIRS = [
    ("No known drug allergies", "Allergy to penicillin documented"),
    ("Not diabetic", "Diagnosed with Type 2 diabetes"),
    ("BP well controlled", "Blood pressure 165/95, uncontrolled"),
    ("No medications", "Currently taking metformin 500mg"),
    ("Denies chest pain", "Reports chest pressure on exertion"),
    ("No history of hypertension", "BP 150/90 on admission"),
    ("Not taking warfarin", "INR 3.5 on warfarin therapy"),
    ("No kidney disease", "eGFR 45, stage 3 CKD"),
    ("Labs normal", "Creatinine 2.8, elevated"),
    ("No symptoms reported", "Complains of severe headache"),
    ("Discontinued metformin", "Continue metformin 1000mg daily"),
    ("No allergies", "Sulfa allergy in record"),
    ("Healthy patient", "Multiple chronic conditions"),
    ("Never smoked", "Current heavy smoker"),
    ("No family history", "Father had heart attack at 50"),
]


def generate_training_data(output_dir: str = "data/conflict_data"):
    """Generate high-quality training dataset."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("🏥 PHASE 7: GENERATING HIGH-QUALITY TRAINING DATA")
    print("="*60)
    
    all_pairs = []
    
    # Add consistent pairs
    for stmt_a, stmt_b in CONSISTENT_PAIRS:
        all_pairs.append({
            "id": f"consist_{len(all_pairs)}",
            "statement_a": stmt_a,
            "statement_b": stmt_b,
            "label": 0,
            "label_name": "CONSISTENT"
        })
    
    # Add conflict pairs
    for stmt_a, stmt_b in CONFLICT_PAIRS:
        all_pairs.append({
            "id": f"conflict_{len(all_pairs)}",
            "statement_a": stmt_a,
            "statement_b": stmt_b,
            "label": 1,
            "label_name": "CONFLICT"
        })
    
    # Generate variations
    for _ in range(50):
        # Random consistent pair
        stmt = random.choice(CONSISTENT_PAIRS)
        all_pairs.append({
            "id": f"consist_var_{len(all_pairs)}",
            "statement_a": stmt[0],
            "statement_b": stmt[1],
            "label": 0,
            "label_name": "CONSISTENT"
        })
    
    for _ in range(50):
        # Random conflict pair
        stmt = random.choice(CONFLICT_PAIRS)
        all_pairs.append({
            "id": f"conflict_var_{len(all_pairs)}",
            "statement_a": stmt[0],
            "statement_b": stmt[1],
            "label": 1,
            "label_name": "CONFLICT"
        })
    
    # Shuffle
    random.shuffle(all_pairs)
    
    # Split
    train_size = int(0.7 * len(all_pairs))
    val_size = int(0.15 * len(all_pairs))
    
    train_data = all_pairs[:train_size]
    val_data = all_pairs[train_size:train_size + val_size]
    test_data = all_pairs[train_size + val_size:]
    
    # Save
    with open(output_path / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_path / "val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    with open(output_path / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    # Count labels
    train_conflict = sum(1 for x in train_data if x['label'] == 1)
    train_consistent = len(train_data) - train_conflict
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total pairs: {len(all_pairs)}")
    print(f"   - CONFLICT: {len([x for x in all_pairs if x['label']==1])}")
    print(f"   - CONSISTENT: {len([x for x in all_pairs if x['label']==0])}")
    print(f"\n   Train: {len(train_data)} (Conflict: {train_conflict}, Consistent: {train_consistent})")
    print(f"   Validation: {len(val_data)}")
    print(f"   Test: {len(test_data)}")
    
    # Show samples
    print("\n📋 Sample data:")
    for sample in train_data[:5]:
        print(f"\n  [{sample['label_name']}]")
        print(f"    A: {sample['statement_a']}")
        print(f"    B: {sample['statement_b']}")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    generate_training_data()