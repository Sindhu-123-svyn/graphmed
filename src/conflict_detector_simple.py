"""
Simple Conflict Detector for GraphMed
Uses TF-IDF + Logistic Regression (no GPU required)
"""

import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class SimpleConflictDetector:
    """Detect conflicts between clinical statements using simple ML model."""
    
    def __init__(self, model_path: str = "models/simple_conflict_classifier"):
        """
        Initialize the conflict detector.
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        
        try:
            with open(self.model_path / "vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            
            with open(self.model_path / "classifier.pkl", "rb") as f:
                self.classifier = pickle.load(f)
            
            self.is_loaded = True
            print("✅ Simple Conflict Detector loaded successfully")
            
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("   Using rule-based detection")
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
        
        # Transform
        X = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)
        
        return {
            "is_conflict": prediction == 1,
            "confidence": float(confidence),
            "prediction": "CONFLICT" if prediction == 1 else "CONSISTENT",
            "statement_a": statement_a,
            "statement_b": statement_b
        }
    
    def _rule_based_predict(self, statement_a: str, statement_b: str) -> dict:
        """Fallback rule-based detection."""
        a_lower = statement_a.lower()
        b_lower = statement_b.lower()
        
        negation_patterns = ["no ", "not ", "denies", "without", "none", "no known"]
        has_negation_a = any(pattern in a_lower for pattern in negation_patterns)
        has_negation_b = any(pattern in b_lower for pattern in negation_patterns)
        
        if has_negation_a != has_negation_b:
            return {
                "is_conflict": True,
                "confidence": 0.65,
                "prediction": "CONFLICT",
                "statement_a": statement_a,
                "statement_b": statement_b
            }
        
        return {
            "is_conflict": False,
            "confidence": 0.55,
            "prediction": "CONSISTENT",
            "statement_a": statement_a,
            "statement_b": statement_b
        }
    
    def batch_predict(self, pairs: list) -> list:
        """Predict for multiple statement pairs."""
        results = []
        for stmt_a, stmt_b in pairs:
            results.append(self.predict(stmt_a, stmt_b))
        return results


# Singleton instance
_detector = None


def get_conflict_detector() -> SimpleConflictDetector:
    """Get or create conflict detector singleton."""
    global _detector
    if _detector is None:
        _detector = SimpleConflictDetector()
    return _detector


def detect_conflict(statement_a: str, statement_b: str) -> dict:
    """Convenience function to detect conflict."""
    detector = get_conflict_detector()
    return detector.predict(statement_a, statement_b)


if __name__ == "__main__":
    # Test
    detector = SimpleConflictDetector()
    
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