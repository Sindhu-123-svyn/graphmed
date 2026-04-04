"""
Phase 7: Train LoRA Conflict Classifier for Clinical Contradictions
Uses BioMedBERT with LoRA for efficient fine-tuning
"""

import json
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


class ClinicalConflictDataset(Dataset):
    """Dataset for clinical conflict classification."""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine statements with [SEP] token
        text = f"{item['statement_a']} [SEP] {item['statement_b']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


def load_data(data_dir: str = "data/conflict_data"):
    """Load training, validation, and test data."""
    
    data_dir = Path(data_dir)
    
    with open(data_dir / "train.json", "r") as f:
        train_data = json.load(f)
    
    with open(data_dir / "val.json", "r") as f:
        val_data = json.load(f)
    
    with open(data_dir / "test.json", "r") as f:
        test_data = json.load(f)
    
    print(f"✅ Loaded {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test samples")
    
    return train_data, val_data, test_data


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_model(train_data, val_data, test_data, output_dir: str = "models/conflict_classifier"):
    """Train the LoRA fine-tuned model."""
    
    print("\n" + "="*60)
    print("🏥 PHASE 7: TRAINING CONFLICT CLASSIFIER")
    print("="*60)
    
    # Model configuration
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased"
    
    print(f"\n📚 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # CONFLICT (1) vs CONSISTENT (0)
        torch_dtype=torch.float32
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # Rank
        lora_alpha=16,
        target_modules=["query", "value", "key", "dense"],
        lora_dropout=0.1,
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Parameters:")
    print(f"   Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   Total: {total_params:,}")
    
    # Create datasets
    train_dataset = ClinicalConflictDataset(train_data, tokenizer)
    val_dataset = ClinicalConflictDataset(val_data, tokenizer)
    test_dataset = ClinicalConflictDataset(test_data, tokenizer)
    
    # Training arguments
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=str(output_path / "logs"),
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    print(f"\n✅ Model saved to: {output_path}")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Results:")
    print(f"   Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"   F1 Score: {test_results['eval_f1']:.4f}")
    print(f"   Precision: {test_results['eval_precision']:.4f}")
    print(f"   Recall: {test_results['eval_recall']:.4f}")
    
    return model, tokenizer, test_results


def test_model(model_path: str = "models/conflict_classifier"):
    """Test the trained model on sample pairs."""
    
    print("\n" + "="*60)
    print("🧪 TESTING CONFLICT CLASSIFIER")
    print("="*60)
    
    # Load model and tokenizer
    from peft import PeftModel
    
    base_model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Test cases
    test_cases = [
        ("Patient has no known drug allergies.", "Patient has penicillin allergy.", "CONFLICT"),
        ("Patient is on Metformin for diabetes.", "Taking Metformin 500mg daily.", "CONSISTENT"),
        ("No history of hypertension.", "Blood pressure 150/95, diagnosed with hypertension.", "CONFLICT"),
        ("Patient reports chest pain.", "Complains of chest pain on exertion.", "CONSISTENT"),
        ("Discontinued Warfarin.", "Still taking Warfarin 5mg daily.", "CONFLICT"),
        ("HbA1c within normal range.", "Diabetes well-controlled.", "CONSISTENT"),
    ]
    
    print("\n📋 Test Results:")
    print("-" * 70)
    
    for statement_a, statement_b, expected in test_cases:
        text = f"{statement_a} [SEP] {statement_b}"
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        predicted = "CONFLICT" if predicted_class == 1 else "CONSISTENT"
        status = "✅" if predicted == expected else "❌"
        
        print(f"\n{status} Expected: {expected} | Predicted: {predicted} | Confidence: {confidence:.3f}")
        print(f"   A: {statement_a}")
        print(f"   B: {statement_b}")
    
    return model, tokenizer


def main():
    """Main execution for Phase 7."""
    
    print("\n" + "="*60)
    print("🏥 GRAPHMED - PHASE 7: LORA CONFLICT CLASSIFIER")
    print("="*60)
    
    print("\nOptions:")
    print("  1. Generate training data")
    print("  2. Train model (requires GPU recommended)")
    print("  3. Test trained model")
    print("  4. Generate data + Train + Test (full pipeline)")
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
        
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()