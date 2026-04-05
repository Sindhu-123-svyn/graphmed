"""
Phase 7: Generate High-Quality Training Data for Conflict Classification.
Builds clinically coherent, diverse statement pairs for LoRA fine-tuning.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

GENDERS = ["male", "female"]
SETTINGS = ["outpatient follow-up", "emergency visit", "inpatient progress note", "telehealth check-in"]
TIMELINES = ["today", "this week", "at this visit", "in the last month"]
SEVERITY = ["mild", "moderate", "severe", "persistent", "intermittent"]

CONDITIONS = [
    "type 2 diabetes",
    "hypertension",
    "asthma",
    "chronic kidney disease",
    "heart failure",
    "atrial fibrillation",
    "copd",
    "hypothyroidism",
]

MEDICATIONS = [
    ("metformin", ["500 mg twice daily", "1000 mg twice daily", "850 mg twice daily"]),
    ("lisinopril", ["10 mg daily", "20 mg daily", "40 mg daily"]),
    ("amlodipine", ["5 mg daily", "10 mg daily"]),
    ("atorvastatin", ["20 mg nightly", "40 mg nightly", "80 mg nightly"]),
    ("warfarin", ["2.5 mg daily", "5 mg daily"]),
    ("levothyroxine", ["50 mcg daily", "75 mcg daily", "100 mcg daily"]),
    ("insulin glargine", ["12 units nightly", "18 units nightly", "24 units nightly"]),
    ("albuterol inhaler", ["2 puffs as needed", "2 puffs every 6 hours as needed"]),
]

ALLERGIES = ["penicillin", "sulfa drugs", "aspirin", "ibuprofen", "latex"]

SYMPTOMS = ["chest pain", "shortness of breath", "fatigue", "dizziness", "palpitations", "nausea"]

LABS = ["hba1c", "ldl", "creatinine", "egfr", "potassium"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _ctx() -> str:
    age = random.randint(22, 89)
    sex = random.choice(GENDERS)
    setting = random.choice(SETTINGS)
    return f"In {setting}, {age}-year-old {sex}"


def _decorate(text: str) -> str:
    return f"{_ctx()} {text}"


def _swap(a: str, b: str) -> Tuple[str, str]:
    return (b, a) if random.random() < 0.4 else (a, b)


def _dedupe(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for row in rows:
        key = (_norm(row["statement_a"]), _norm(row["statement_b"]), int(row["label"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _lab_values(name: str) -> Tuple[str, str]:
    if name == "hba1c":
        return (f"{random.uniform(8.0, 12.4):.1f}%", f"{random.uniform(4.8, 5.8):.1f}%")
    if name == "ldl":
        return (f"{random.randint(150, 220)} mg/dl", f"{random.randint(60, 100)} mg/dl")
    if name == "creatinine":
        return (f"{random.uniform(1.8, 3.4):.1f} mg/dl", f"{random.uniform(0.6, 1.1):.1f} mg/dl")
    if name == "egfr":
        return (f"{random.randint(18, 45)} ml/min", f"{random.randint(85, 120)} ml/min")
    return (f"{random.uniform(5.4, 6.3):.1f} mmol/l", f"{random.uniform(3.8, 4.8):.1f} mmol/l")


def _row(a: str, b: str, label: int, topic: str, difficulty: str) -> Dict:
    a, b = _swap(_decorate(a), _decorate(b))
    return {
        "statement_a": a,
        "statement_b": b,
        "label": label,
        "label_name": "CONFLICT" if label == 1 else "CONSISTENT",
        "topic": topic,
        "difficulty": difficulty,
    }


def _condition_consistent() -> Dict:
    c = random.choice(CONDITIONS)
    t = random.choice(TIMELINES)
    a = f"has known {c} and remains clinically stable {t}."
    b = f"assessment confirms ongoing {c} without decompensation {t}."
    return _row(a, b, 0, "condition", "easy")


def _condition_conflict() -> Dict:
    c = random.choice(CONDITIONS)
    t = random.choice(TIMELINES)
    a = f"denies any history of {c} {t}."
    b = f"problem list documents active {c} {t}."
    return _row(a, b, 1, "condition", "easy")


def _med_consistent() -> Dict:
    med, doses = random.choice(MEDICATIONS)
    dose = random.choice(doses)
    t = random.choice(TIMELINES)
    a = f"is taking {med} {dose} {t}."
    b = f"medication reconciliation confirms {med} {dose} {t}."
    return _row(a, b, 0, "medication", "medium")


def _med_conflict() -> Dict:
    med, doses = random.choice(MEDICATIONS)
    dose = random.choice(doses)
    t = random.choice(TIMELINES)
    a = f"states not taking {med} {t}."
    b = f"active plan says continue {med} {dose} {t}."
    return _row(a, b, 1, "medication", "medium")


def _allergy_consistent() -> Dict:
    al = random.choice(ALLERGIES)
    sev = random.choice(SEVERITY)
    a = f"reports {sev} allergy to {al}."
    b = f"allergy section lists {al} reaction with {sev} severity."
    return _row(a, b, 0, "allergy", "easy")


def _allergy_conflict() -> Dict:
    al = random.choice(ALLERGIES)
    a = "has no known drug allergies."
    b = f"chart documents allergy to {al}."
    return _row(a, b, 1, "allergy", "easy")


def _lab_consistent() -> Dict:
    lab = random.choice(LABS)
    high, low = _lab_values(lab)
    t = random.choice(TIMELINES)
    if random.random() < 0.5:
        a = f"{lab.upper()} is elevated at {high} {t}."
        b = f"latest {lab.upper()} result was {high}, above target {t}."
    else:
        a = f"{lab.upper()} is within normal range at {low} {t}."
        b = f"latest {lab.upper()} result was {low}, normal {t}."
    return _row(a, b, 0, "lab", "hard")


def _lab_conflict() -> Dict:
    lab = random.choice(LABS)
    high, low = _lab_values(lab)
    t = random.choice(TIMELINES)
    a = f"{lab.upper()} is normal at {low} {t}."
    b = f"latest {lab.upper()} measured {high}, clearly elevated {t}."
    return _row(a, b, 1, "lab", "hard")


def _symptom_consistent() -> Dict:
    s = random.choice(SYMPTOMS)
    sev = random.choice(SEVERITY)
    a = f"reports {sev} {s} {random.choice(TIMELINES)}."
    b = f"review of systems confirms {sev} {s}."
    return _row(a, b, 0, "symptom", "easy")


def _symptom_conflict() -> Dict:
    s = random.choice(SYMPTOMS)
    a = f"denies {s} {random.choice(TIMELINES)}."
    b = f"clinical note documents ongoing {s}."
    return _row(a, b, 1, "symptom", "medium")


def _hard_negative_consistent() -> Dict:
    c = random.choice(CONDITIONS)
    sev = random.choice(SEVERITY)
    a = f"has {c} but no recent exacerbation; disease appears {sev} and controlled."
    b = f"history of {c} with stable status and no acute worsening."
    return _row(a, b, 0, "hard_negative", "hard")


def _hard_conflict_timeline() -> Dict:
    med, doses = random.choice(MEDICATIONS)
    old_dose = random.choice(doses)
    new_dose = random.choice(doses)
    if new_dose == old_dose:
        new_dose = doses[0]
    a = f"{med} was discontinued last month due to side effects."
    b = f"plan at this visit is to continue {med} {new_dose}."
    return _row(a, b, 1, "timeline", "hard")


def _build_rows(target_consistent: int, target_conflict: int) -> List[Dict]:
    consistent_fns = [
        _condition_consistent,
        _med_consistent,
        _allergy_consistent,
        _lab_consistent,
        _symptom_consistent,
        _hard_negative_consistent,
    ]
    conflict_fns = [
        _condition_conflict,
        _med_conflict,
        _allergy_conflict,
        _lab_conflict,
        _symptom_conflict,
        _hard_conflict_timeline,
    ]

    rows = []
    for _ in range(target_consistent):
        rows.append(random.choice(consistent_fns)())
    for _ in range(target_conflict):
        rows.append(random.choice(conflict_fns)())

    return rows


def _assign_ids(rows: List[Dict]) -> List[Dict]:
    out = []
    for i, row in enumerate(rows, 1):
        r = dict(row)
        r["id"] = f"pair_{i:07d}"
        out.append(r)
    return out


def _counts(rows: List[Dict]) -> Dict[str, int]:
    return {
        "conflict": sum(1 for r in rows if int(r["label"]) == 1),
        "consistent": sum(1 for r in rows if int(r["label"]) == 0),
    }


def generate_training_data(
    output_dir: str = "data/conflict_data",
    total_consistent: int = 4000,
    total_conflict: int = 4000,
    min_after_dedupe: int = 5000,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Generate large, diverse, and stratified train/val/test datasets."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _dedupe(_build_rows(total_consistent, total_conflict))

    # If dedupe is too aggressive, auto-expand with another stochastic pass.
    attempts = 0
    while len(rows) < min_after_dedupe and attempts < 4:
        attempts += 1
        extra = _build_rows(total_consistent // 2, total_conflict // 2)
        rows = _dedupe(rows + extra)

    random.shuffle(rows)
    labels = [int(r["label"]) for r in rows]

    train_rows, temp_rows = train_test_split(
        rows,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )

    temp_labels = [int(r["label"]) for r in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels,
    )

    train_rows = _assign_ids(train_rows)
    val_rows = _assign_ids(val_rows)
    test_rows = _assign_ids(test_rows)

    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_rows, f, indent=2, ensure_ascii=False)
    with open(out_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_rows, f, indent=2, ensure_ascii=False)
    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_rows, f, indent=2, ensure_ascii=False)

    stats = {
        "seed": SEED,
        "requested_consistent": total_consistent,
        "requested_conflict": total_conflict,
        "min_after_dedupe": min_after_dedupe,
        "total_after_dedupe": len(rows),
        "attempts_for_min_size": attempts,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "all_labels": _counts(rows),
        "train_labels": _counts(train_rows),
        "val_labels": _counts(val_rows),
        "test_labels": _counts(test_rows),
    }

    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("GRAPHMED PHASE 7 - TRAINING DATA GENERATED")
    print("=" * 60)
    print(f"Total after dedupe: {stats['total_after_dedupe']}")
    print(f"Train/Val/Test: {stats['train_size']}/{stats['val_size']}/{stats['test_size']}")
    print(
        "Label totals: "
        f"CONFLICT={stats['all_labels']['conflict']} | "
        f"CONSISTENT={stats['all_labels']['consistent']}"
    )

    return train_rows, val_rows, test_rows


if __name__ == "__main__":
    generate_training_data()
