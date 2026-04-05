"""
Phase 7: Train LoRA Conflict Classifier
Fine-tunes a compact BERT model with LoRA for CONFLICT vs CONSISTENT.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


SEED = 42
MODEL_NAME = os.getenv("CONFLICT_BASE_MODEL", "google-bert/bert-base-uncased")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ClinicalConflictDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int = 128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        enc = self.tokenizer(
            item["statement_a"],
            item["statement_b"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(item["label"]), dtype=torch.long),
        }


def load_data(data_dir: str = "data/conflict_data") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    p = Path(data_dir)
    with open(p / "train.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open(p / "val.json", "r", encoding="utf-8") as f:
        val = json.load(f)
    with open(p / "test.json", "r", encoding="utf-8") as f:
        test = json.load(f)
    print(f"Loaded train/val/test: {len(train)}/{len(val)}/{len(test)}")
    return train, val, test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model(
    train_rows: List[Dict],
    val_rows: List[Dict],
    test_rows: List[Dict],
    output_dir: str = "models/conflict_classifier",
    num_train_epochs: int = 3,
    learning_rate: float = 3e-5,
    train_batch_size: int = 16,
):
    set_seed(SEED)

    print("\n" + "=" * 60)
    print("GRAPHMED PHASE 7 - LORA TRAINING")
    print("=" * 60)
    print(f"Base model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        torch_dtype=torch.float32,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        modules_to_save=["classifier"],
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")

    train_ds = ClinicalConflictDataset(train_rows, tokenizer)
    val_ds = ClinicalConflictDataset(val_rows, tokenizer)
    test_ds = ClinicalConflictDataset(test_rows, tokenizer)

    labels = np.array([int(r["label"]) for r in train_rows])
    neg = max(1, int((labels == 0).sum()))
    pos = max(1, int((labels == 1).sum()))
    class_weights = torch.tensor([1.0, float(neg) / float(pos)], dtype=torch.float32)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    eval_test = trainer.evaluate(test_ds)

    meta = {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "class_weights": class_weights.tolist(),
        "test_metrics": {
            "accuracy": float(eval_test.get("eval_accuracy", 0.0)),
            "f1": float(eval_test.get("eval_f1", 0.0)),
            "precision": float(eval_test.get("eval_precision", 0.0)),
            "recall": float(eval_test.get("eval_recall", 0.0)),
        },
    }

    with open(out / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining complete")
    print(json.dumps(meta["test_metrics"], indent=2))

    return model, tokenizer, meta


def test_model(model_dir: str = "models/conflict_classifier"):
    from peft import PeftModel

    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = PeftModel.from_pretrained(base, str(model_path))
    model.eval()

    examples = [
        ("No known drug allergies", "Allergy to penicillin", "CONFLICT"),
        ("Taking metformin", "On metformin 500mg BID", "CONSISTENT"),
        ("No hypertension", "BP 160/100 uncontrolled", "CONFLICT"),
        ("Chest pain on exertion", "Exertional chest discomfort", "CONSISTENT"),
    ]

    print("\nQuick test")
    for a, b, expected in examples:
        inp = tokenizer(a, b, return_tensors="pt", truncation=True, max_length=192)
        with torch.no_grad():
            logits = model(**inp).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred = int(torch.argmax(probs).item())
        name = "CONFLICT" if pred == 1 else "CONSISTENT"
        conf = float(probs[pred].item())
        ok = "OK" if name == expected else "ERR"
        print(f"{ok} expected={expected} predicted={name} confidence={conf:.3f}")
        print(f"  A: {a}")
        print(f"  B: {b}")


def main():
    print("\n" + "=" * 60)
    print("GRAPHMED - PHASE 7 LORA CONFLICT CLASSIFIER")
    print("=" * 60)
    print("1. Generate training data")
    print("2. Train LoRA classifier")
    print("3. Test LoRA classifier")
    print("4. Full pipeline")
    print("5. Exit")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == "1":
        from phase7_generate_training_data import generate_training_data

        generate_training_data()
    elif choice == "2":
        train, val, test = load_data()
        train_model(train, val, test)
    elif choice == "3":
        test_model()
    elif choice == "4":
        from phase7_generate_training_data import generate_training_data

        generate_training_data()
        train, val, test = load_data()
        train_model(train, val, test)
        test_model()
    else:
        print("Exiting")


if __name__ == "__main__":
    main()
