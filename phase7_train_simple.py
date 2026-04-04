"""
Phase 7: Improved Conflict Classifier with Better Features
"""

import json
import torch
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import pickle
import re

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def load_data(data_dir: str = "data/conflict_data"):
    """Load training data."""
    data_dir = Path(data_dir)
    
    with open(data_dir / "train.json", "r") as f:
        train_data = json.load(f)
    
    with open(data_dir / "val.json", "r") as f:
        val_data = json.load(f)
    
    with open(data_dir / "test.json", "r") as f:
        test_data = json.load(f)
    
    print(f"✅ Loaded {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test samples")
    return train_data, val_data, test_data


def extract_features(text):
    """Extract additional features from text."""
    features = {}
    
    # Negation detection
    negation_words = ['no', 'not', 'denies', 'without', 'none', 'never', 'denied']
    features['has_negation'] = any(word in text.lower() for word in negation_words)
    
    # Length features
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    # Medical term indicators
    medical_terms = ['allergy', 'diabetes', 'hypertension', 'medication', 'prescribed', 
                     'diagnosed', 'symptoms', 'blood', 'pressure', 'glucose']
    features['medical_term_count'] = sum(1 for term in medical_terms if term in text.lower())
    
    return features


def prepare_features(data, vectorizer=None, fit_vectorizer=False):
    """Prepare TF-IDF + custom features."""
    
    texts = []
    labels = []
    custom_features = []
    
    for item in data:
        text = f"{item['statement_a']} [SEP] {item['statement_b']}"
        texts.append(text)
        labels.append(item['label'])
        custom_features.append(extract_features(text))
    
    # TF-IDF features
    if fit_vectorizer:
        X_tfidf = vectorizer.fit_transform(texts)
    else:
        X_tfidf = vectorizer.transform(texts)
    
    # Combine with custom features
    custom_feat_matrix = np.array([[f['has_negation'], f['text_length'], 
                                      f['word_count'], f['medical_term_count']] 
                                    for f in custom_features])
    
    # Combine all features
    from scipy.sparse import hstack
    X_combined = hstack([X_tfidf, custom_feat_matrix])
    
    return X_combined, labels


def train_model(train_data, val_data, test_data):
    """Train improved classifier."""
    
    print("\n" + "="*60)
    print("🏥 PHASE 7: TRAINING IMPROVED CONFLICT CLASSIFIER")
    print("="*60)
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    
    # Prepare features
    print("\n📊 Creating features...")
    X_train, y_train = prepare_features(train_data, vectorizer, fit_vectorizer=True)
    X_val, y_val = prepare_features(val_data, vectorizer, fit_vectorizer=False)
    X_test, y_test = prepare_features(test_data, vectorizer, fit_vectorizer=False)
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Feature dimension: {X_train.shape[1]}")
    
    # Train Random Forest (better for this task)
    print("\n🚀 Training Random Forest classifier...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    classifier.fit(X_train, y_train)
    
    # Evaluate
    train_pred = classifier.predict(X_train)
    val_pred = classifier.predict(X_val)
    test_pred = classifier.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred)
    
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred)
    
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)
    
    print("\n📊 Results:")
    print(f"   Training - Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"   Validation - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    print(f"   Test - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    print("\n📋 Classification Report (Test):")
    print(classification_report(y_test, test_pred, target_names=['CONSISTENT', 'CONFLICT']))
    
    # Save model
    output_dir = Path("models/improved_conflict_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(output_dir / "classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)
    
    print(f"\n✅ Model saved to: {output_dir}")
    
    return classifier, vectorizer, test_acc, test_f1


def test_model():
    """Test the trained model on realistic examples."""
    
    print("\n" + "="*60)
    print("🧪 TESTING IMPROVED CONFLICT CLASSIFIER")
    print("="*60)
    
    model_dir = Path("models/improved_conflict_classifier")
    
    if not model_dir.exists():
        print("❌ Model not found. Please train first.")
        return
    
    with open(model_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    with open(model_dir / "classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    
    # Better test cases
    test_cases = [
        ("No known drug allergies", "Allergy to penicillin", "CONFLICT"),
        ("Taking metformin for diabetes", "Prescribed metformin 500mg", "CONSISTENT"),
        ("No history of hypertension", "Blood pressure 150/95", "CONFLICT"),
        ("Patient reports chest pain", "Complains of chest pressure", "CONSISTENT"),
        ("Not on any blood thinners", "Taking warfarin daily", "CONFLICT"),
        ("HbA1c 6.5%", "Diabetes well-controlled", "CONSISTENT"),
        ("Denies shortness of breath", "SOB on minimal exertion", "CONFLICT"),
        ("No kidney problems", "eGFR 45, stage 3 CKD", "CONFLICT"),
        ("Lisinopril discontinued", "Continue lisinopril 10mg", "CONFLICT"),
        ("Patient is healthy", "Multiple chronic conditions", "CONFLICT"),
    ]
    
    print("\n📋 Test Results:")
    print("-" * 70)
    
    correct = 0
    for stmt_a, stmt_b, expected in test_cases:
        text = f"{stmt_a} [SEP] {stmt_b}"
        
        # Extract features
        X_tfidf = vectorizer.transform([text])
        custom = np.array([[extract_features(text)['has_negation'],
                            extract_features(text)['text_length'],
                            extract_features(text)['word_count'],
                            extract_features(text)['medical_term_count']]])
        
        from scipy.sparse import hstack
        X = hstack([X_tfidf, custom])
        
        prediction = classifier.predict(X)[0]
        proba = classifier.predict_proba(X)[0]
        confidence = max(proba)
        
        predicted = "CONFLICT" if prediction == 1 else "CONSISTENT"
        status = "✅" if predicted == expected else "❌"
        
        if predicted == expected:
            correct += 1
        
        print(f"\n{status} Expected: {expected} | Predicted: {predicted} | Conf: {confidence:.3f}")
        print(f"   A: {stmt_a}")
        print(f"   B: {stmt_b}")
    
    print(f"\n📊 Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")


def main():
    """Main execution."""
    
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 7: IMPROVED CONFLICT CLASSIFIER")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Generate training data")
    print("  2. Train improved model")
    print("  3. Test model")
    print("  4. Full pipeline (Generate + Train + Test)")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        from phase7_generate_training_data import generate_training_data
        generate_training_data()
        
    elif choice == "2":
        train_data, val_data, test_data = load_data()
        train_model(train_data, val_data, test_data)
        
    elif choice == "3":
        test_model()
        
    elif choice == "4":
        print("\n🚀 Running full pipeline...")
        from phase7_generate_training_data import generate_training_data
        generate_training_data()
        train_data, val_data, test_data = load_data()
        train_model(train_data, val_data, test_data)
        test_model()
        
    elif choice == "5":
        print("Exiting...")


if __name__ == "__main__":
    main()