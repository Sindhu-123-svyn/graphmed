"""
Phase 9: Evaluation Module for GraphMed
Runs experiments comparing GraphMed against baseline RAG
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent_direct import DirectReActAgent
from src.baseline_rag import BaselineRAG
from src.conflict_runtime import get_conflict_detector


class Evaluator:
    """Evaluate GraphMed against baseline."""
    
    def __init__(self):
        self.graphmed = DirectReActAgent(max_steps=3)
        self.baseline = BaselineRAG()
        self.conflict_detector = get_conflict_detector()
        self.bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def generate_test_questions(self, patient_id: str) -> List[Dict]:
        """Generate test questions for a patient."""
        
        # Load patient data to create relevant questions
        patient_path = Path(f"data/patients_processed/{patient_id}.json")
        
        if not patient_path.exists():
            return []
        
        with open(patient_path, 'r') as f:
            patient = json.load(f)
        
        questions = []
        
        # Extract information for question generation
        all_conditions = set()
        all_medications = set()
        all_symptoms = set()
        
        for visit in patient.get('visits', []):
            for d in visit.get('diagnoses', []):
                all_conditions.add(d)
            for m in visit.get('medications', []):
                all_medications.add(m)
            for s in visit.get('symptoms', []):
                all_symptoms.add(s)
        
        # Create questions
        for condition in list(all_conditions)[:3]:
            questions.append({
                "question": f"What conditions does patient {patient_id} have?",
                "type": "conditions",
                "expected_keywords": [condition.lower()]
            })
            break
        
        for medication in list(all_medications)[:3]:
            questions.append({
                "question": f"What medications is patient {patient_id} taking?",
                "type": "medications", 
                "expected_keywords": [medication.lower()]
            })
            break
        
        for symptom in list(all_symptoms)[:3]:
            questions.append({
                "question": f"What symptoms has patient {patient_id} reported?",
                "type": "symptoms",
                "expected_keywords": [symptom.lower()]
            })
            break
        
        # Add drug interaction question if medications exist
        if len(all_medications) >= 2:
            meds = list(all_medications)[:2]
            questions.append({
                "question": f"Check if {meds[0]} and {meds[1]} interact",
                "type": "interaction",
                "expected_keywords": ["interaction", "safe"]
            })
        
        return questions
    
    def compute_factscore(self, answer: str, expected_keywords: List[str]) -> float:
        """Compute factual accuracy based on keyword presence."""
        answer_lower = answer.lower()
        matches = sum(1 for kw in expected_keywords if kw in answer_lower)
        return matches / len(expected_keywords) if expected_keywords else 0.5
    
    def compute_bertscore(self, answer: str, reference: str) -> float:
        """Compute BERTScore between answer and reference."""
        if not answer or not reference:
            return 0.0
        P, R, F1 = self.bertscorer.score([answer], [reference])
        return F1.item()
    
    def compute_rouge(self, answer: str, reference: str) -> float:
        """Compute ROUGE score."""
        if not answer or not reference:
            return 0.0
        scores = self.rouge_scorer.score(reference, answer)
        return scores['rougeL'].fmeasure
    
    def experiment_factual_accuracy(self, patient_ids: List[str]) -> Dict:
        """
        Experiment 1: Compare factual accuracy on clinical QA.
        """
        print("\n" + "="*60)
        print("📊 EXPERIMENT 1: FACTUAL ACCURACY")
        print("="*60)
        
        results = {
            "graphmed": {"scores": [], "answers": [], "times": []},
            "baseline": {"scores": [], "answers": [], "times": []}
        }
        
        for patient_id in patient_ids:
            questions = self.generate_test_questions(patient_id)
            
            for q in questions:
                # GraphMed
                start = time.time()
                graphmed_result = self.graphmed.invoke(patient_id, q["question"])
                graphmed_time = time.time() - start
                graphmed_score = self.compute_factscore(
                    graphmed_result["answer"], 
                    q["expected_keywords"]
                )
                
                results["graphmed"]["scores"].append(graphmed_score)
                results["graphmed"]["answers"].append(graphmed_result["answer"])
                results["graphmed"]["times"].append(graphmed_time)
                
                # Baseline
                start = time.time()
                baseline_answer = self.baseline.answer(patient_id, q["question"])
                baseline_time = time.time() - start
                baseline_score = self.compute_factscore(
                    baseline_answer,
                    q["expected_keywords"]
                )
                
                results["baseline"]["scores"].append(baseline_score)
                results["baseline"]["answers"].append(baseline_answer)
                results["baseline"]["times"].append(baseline_time)
                
                print(f"\n  Q: {q['question']}")
                print(f"    GraphMed Score: {graphmed_score:.3f} ({graphmed_time:.2f}s)")
                print(f"    Baseline Score: {baseline_score:.3f} ({baseline_time:.2f}s)")
        
        # Calculate statistics
        graphmed_avg = np.mean(results["graphmed"]["scores"])
        baseline_avg = np.mean(results["baseline"]["scores"])
        graphmed_avg_time = np.mean(results["graphmed"]["times"])
        baseline_avg_time = np.mean(results["baseline"]["times"])
        
        print(f"\n📈 Results:")
        print(f"   GraphMed Avg Accuracy: {graphmed_avg:.3f}")
        print(f"   Baseline Avg Accuracy: {baseline_avg:.3f}")
        print(f"   Improvement: {(graphmed_avg - baseline_avg) * 100:.1f}%")
        print(f"   GraphMed Avg Time: {graphmed_avg_time:.2f}s")
        print(f"   Baseline Avg Time: {baseline_avg_time:.2f}s")
        
        return results
    
    def experiment_longitudinal_memory(self, patient_id: str) -> Dict:
        """
        Experiment 2: Test if system remembers information across visits.
        """
        print("\n" + "="*60)
        print("📊 EXPERIMENT 2: LONGITUDINAL MEMORY")
        print("="*60)
        
        # Load patient data
        patient_path = Path(f"data/patients_processed/{patient_id}.json")
        with open(patient_path, 'r') as f:
            patient = json.load(f)
        
        visits = patient.get('visits', [])
        
        results = {
            "graphmed": {"correct": 0, "total": 0, "details": []},
            "baseline": {"correct": 0, "total": 0, "details": []}
        }
        
        # Test questions about early visits after seeing later ones
        for i, early_visit in enumerate(visits[:2]):  # Test first 2 visits
            early_date = early_visit.get('date', 'Unknown')
            early_diagnoses = early_visit.get('diagnoses', [])
            
            if early_diagnoses:
                question = f"What conditions were present in the visit on {early_date}?"
                
                # GraphMed answer
                graphmed_result = self.graphmed.invoke(patient_id, question)
                graphmed_answer = graphmed_result["answer"].lower()
                
                # Check if early diagnoses mentioned
                correct = any(d.lower() in graphmed_answer for d in early_diagnoses)
                results["graphmed"]["correct"] += 1 if correct else 0
                results["graphmed"]["total"] += 1
                results["graphmed"]["details"].append({
                    "visit_date": early_date,
                    "correct": correct,
                    "answer": graphmed_result["answer"][:100]
                })
                
                # Baseline answer
                baseline_answer = self.baseline.answer(patient_id, question).lower()
                baseline_correct = any(d.lower() in baseline_answer for d in early_diagnoses)
                results["baseline"]["correct"] += 1 if baseline_correct else 0
                results["baseline"]["total"] += 1
                results["baseline"]["details"].append({
                    "visit_date": early_date,
                    "correct": baseline_correct,
                    "answer": baseline_answer[:100]
                })
                
                print(f"\n  Visit {i+1} ({early_date}):")
                print(f"    GraphMed: {'✅' if correct else '❌'}")
                print(f"    Baseline: {'✅' if baseline_correct else '❌'}")
        
        graphmed_acc = results["graphmed"]["correct"] / results["graphmed"]["total"] if results["graphmed"]["total"] > 0 else 0
        baseline_acc = results["baseline"]["correct"] / results["baseline"]["total"] if results["baseline"]["total"] > 0 else 0
        
        print(f"\n📈 Results:")
        print(f"   GraphMed Accuracy: {graphmed_acc:.3f}")
        print(f"   Baseline Accuracy: {baseline_acc:.3f}")
        print(f"   Improvement: {(graphmed_acc - baseline_acc) * 100:.1f}%")
        
        return results
    
    def experiment_conflict_detection(self) -> Dict:
        """
        Experiment 3: Measure conflict detection performance.
        """
        print("\n" + "="*60)
        print("📊 EXPERIMENT 3: CONFLICT DETECTION")
        print("="*60)
        
        # Test cases with known labels
        test_cases = [
            ("No known drug allergies", "Allergy to penicillin", True),
            ("Taking metformin for diabetes", "Prescribed metformin 500mg", False),
            ("No history of hypertension", "Blood pressure 150/95", True),
            ("Patient reports chest pain", "Complains of chest pressure", False),
            ("Not on any blood thinners", "Taking warfarin daily", True),
            ("HbA1c within normal range", "Diabetes well-controlled", False),
            ("Denies shortness of breath", "SOB on minimal exertion", True),
            ("No kidney problems", "eGFR 45, stage 3 CKD", True),
            ("Patient is healthy", "Multiple chronic conditions", True),
            ("Lisinopril discontinued", "Continue lisinopril 10mg", True),
        ]
        
        predictions = []
        true_labels = []
        
        print("\n📋 Test Results:")
        print("-" * 70)
        
        for stmt_a, stmt_b, expected in test_cases:
            result = self.conflict_detector.predict(stmt_a, stmt_b)
            predicted = result["is_conflict"]
            confidence = result["confidence"]
            
            predictions.append(1 if predicted else 0)
            true_labels.append(1 if expected else 0)
            
            status = "✅" if predicted == expected else "❌"
            print(f"{status} Expected: {expected} | Predicted: {predicted} | Conf: {confidence:.3f}")
            print(f"   A: {stmt_a[:40]}...")
            print(f"   B: {stmt_b[:40]}...")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        print(f"\n📈 Results:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": predictions,
            "true_labels": true_labels
        }
    
    def run_all_experiments(self):
        """Run all three experiments."""
        
        print("\n" + "="*60)
        print("🏥 GRAPHMED - PHASE 9: EVALUATION")
        print("="*60)
        
        # Experiment 1: Factual Accuracy
        patient_ids = [f"P{i:03d}" for i in range(1, 6)]  # First 5 patients
        exp1_results = self.experiment_factual_accuracy(patient_ids)
        
        # Experiment 2: Longitudinal Memory
        exp2_results = self.experiment_longitudinal_memory("P001")
        
        # Experiment 3: Conflict Detection
        exp3_results = self.experiment_conflict_detection()
        
        # Summary
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        graphmed_avg = np.mean(exp1_results["graphmed"]["scores"])
        baseline_avg = np.mean(exp1_results["baseline"]["scores"])
        
        print(f"\nExperiment 1 - Factual Accuracy:")
        print(f"   GraphMed: {graphmed_avg:.3f}")
        print(f"   Baseline: {baseline_avg:.3f}")
        print(f"   Δ: +{(graphmed_avg - baseline_avg) * 100:.1f}%")
        
        print(f"\nExperiment 2 - Longitudinal Memory:")
        graphmed_acc = exp2_results["graphmed"]["correct"] / exp2_results["graphmed"]["total"] if exp2_results["graphmed"]["total"] > 0 else 0
        baseline_acc = exp2_results["baseline"]["correct"] / exp2_results["baseline"]["total"] if exp2_results["baseline"]["total"] > 0 else 0
        print(f"   GraphMed: {graphmed_acc:.3f}")
        print(f"   Baseline: {baseline_acc:.3f}")
        print(f"   Δ: +{(graphmed_acc - baseline_acc) * 100:.1f}%")
        
        print(f"\nExperiment 3 - Conflict Detection:")
        print(f"   Accuracy: {exp3_results['accuracy']:.3f}")
        print(f"   F1 Score: {exp3_results['f1']:.3f}")
        
        return {
            "experiment1": exp1_results,
            "experiment2": exp2_results,
            "experiment3": exp3_results
        }


def main():
    """Main execution for Phase 9."""
    
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 9: EVALUATION")
    print("="*60)
    
    evaluator = Evaluator()
    
    print("\nOptions:")
    print("  1. Run all experiments")
    print("  2. Run Experiment 1 only (Factual Accuracy)")
    print("  3. Run Experiment 2 only (Longitudinal Memory)")
    print("  4. Run Experiment 3 only (Conflict Detection)")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        evaluator.run_all_experiments()
    elif choice == "2":
        patient_ids = [f"P{i:03d}" for i in range(1, 6)]
        evaluator.experiment_factual_accuracy(patient_ids)
    elif choice == "3":
        evaluator.experiment_longitudinal_memory("P001")
    elif choice == "4":
        evaluator.experiment_conflict_detection()
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice. Running all experiments...")
        evaluator.run_all_experiments()


if __name__ == "__main__":
    main()