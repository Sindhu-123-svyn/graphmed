"""
Conflict Detector Module for GraphMed
Uses trained LoRA model to detect clinical contradictions
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")


class ConflictDetector:
    """Detect conflicts between clinical statements."""
    
    def __init__(self, model_path: str = "models/conflict_classifier"):
        """
        Initialize the conflict detector.
        
        Args:
            model_path: Path to trained LoRA model
        """
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased"
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=2
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(self.base_model, str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print("✅ Conflict Detector loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Could not load trained model: {e}")
            print("   Using fallback rule-based detection")
            self.is_loaded = False
    
    def predict(self, statement_a: str, statement_b: str) -> dict:
        """
        Predict if two statements conflict.
        
        Args:
            statement_a: First clinical statement
            statement_b: Second clinical statement
        
        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_loaded:
            return self._rule_based_predict(statement_a, statement_b)
        
        # Combine statements
        text = f"{statement_a} [SEP] {statement_b}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "is_conflict": predicted_class == 1,
            "confidence": confidence,
            "prediction": "CONFLICT" if predicted_class == 1 else "CONSISTENT",
            "statement_a": statement_a,
            "statement_b": statement_b
        }
    
    def _rule_based_predict(self, statement_a: str, statement_b: str) -> dict:
        """Fallback rule-based detection when model not available."""
        
        # Convert to lowercase for comparison
        a_lower = statement_a.lower()
        b_lower = statement_b.lower()
        
        # Check for negation patterns
        negation_patterns = ["no ", "not ", "denies", "without", "none"]
        has_negation_a = any(pattern in a_lower for pattern in negation_patterns)
        has_negation_b = any(pattern in b_lower for pattern in negation_patterns)
        
        # Check for allergy patterns
        allergy_patterns = ["allergy", "allergic", "reaction"]
        has_allergy_a = any(pattern in a_lower for pattern in allergy_patterns)
        has_allergy_b = any(pattern in b_lower for pattern in allergy_patterns)
        
        # Simple rule: if one says "no X" and other mentions X, likely conflict
        if has_negation_a != has_negation_b:
            # Extract key terms
            import re
            words_a = set(re.findall(r'\b\w+\b', a_lower))
            words_b = set(re.findall(r'\b\w+\b', b_lower))
            
            # If they share significant terms but one has negation
            common_words = words_a.intersection(words_b)
            if len(common_words) > 2:
                return {
                    "is_conflict": True,
                    "confidence": 0.7,
                    "prediction": "CONFLICT",
                    "statement_a": statement_a,
                    "statement_b": statement_b
                }
        
        return {
            "is_conflict": False,
            "confidence": 0.5,
            "prediction": "CONSISTENT",
            "statement_a": statement_a,
            "statement_b": statement_b
        }
    
    def detect_in_visit_sequence(self, statements: list) -> list:
        """
        Detect conflicts across a sequence of statements.
        
        Args:
            statements: List of (statement, visit_id) tuples
        
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                stmt_a, visit_a = statements[i]
                stmt_b, visit_b = statements[j]
                
                result = self.predict(stmt_a, stmt_b)
                
                if result["is_conflict"] and result["confidence"] > 0.6:
                    conflicts.append({
                        "statement_a": stmt_a,
                        "visit_a": visit_a,
                        "statement_b": stmt_b,
                        "visit_b": visit_b,
                        "confidence": result["confidence"]
                    })
        
        return conflicts


# Singleton instance
_conflict_detector = None


def get_conflict_detector() -> ConflictDetector:
    """Get or create conflict detector singleton."""
    global _conflict_detector
    if _conflict_detector is None:
        _conflict_detector = ConflictDetector()
    return _conflict_detector


def detect_conflict(statement_a: str, statement_b: str) -> dict:
    """Convenience function to detect conflict between two statements."""
    detector = get_conflict_detector()
    return detector.predict(statement_a, statement_b)


if __name__ == "__main__":
    # Test the conflict detector
    detector = ConflictDetector()
    
    test_pairs = [
        ("Patient has no known drug allergies.", "Patient has penicillin allergy."),
        ("Patient is on Metformin.", "Taking Metformin 500mg daily."),
        ("No history of diabetes.", "Diagnosed with Type 2 Diabetes."),
    ]
    
    print("\n🧪 Testing Conflict Detector:")
    for stmt_a, stmt_b in test_pairs:
        result = detector.predict(stmt_a, stmt_b)
        print(f"\n{result['prediction']} (conf: {result['confidence']:.3f})")
        print(f"  A: {stmt_a}")
        print(f"  B: {stmt_b}")