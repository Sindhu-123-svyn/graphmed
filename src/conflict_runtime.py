"""
Runtime conflict detector router.
Prefers LoRA detector, falls back to simple ML detector, then rule-based logic.
"""

import os
from typing import Any, Dict

from src.conflict_detector import ConflictDetector
from src.conflict_detector_simple import SimpleConflictDetector


class RuntimeConflictDetector:
    def __init__(
        self,
        lora_model_path: str = None,
        simple_model_path: str = None,
    ):
        self.lora_model_path = lora_model_path or os.getenv("CONFLICT_MODEL_PATH", "models/conflict_classifier")
        self.simple_model_path = simple_model_path or os.getenv("SIMPLE_CONFLICT_MODEL_PATH", "models/simple_conflict_classifier")

        self.primary = ConflictDetector(model_path=self.lora_model_path)
        self.mode = "lora"

        if not getattr(self.primary, "is_loaded", False):
            self.primary = SimpleConflictDetector(model_path=self.simple_model_path)
            self.mode = "simple"

    def predict(self, statement_a: str, statement_b: str) -> Dict[str, Any]:
        result = self.primary.predict(statement_a, statement_b)
        if "runtime_mode" not in result:
            result["runtime_mode"] = self.mode
        return result


_detector = None


def get_conflict_detector() -> RuntimeConflictDetector:
    global _detector
    if _detector is None:
        _detector = RuntimeConflictDetector()
    return _detector


def detect_conflict(statement_a: str, statement_b: str) -> Dict[str, Any]:
    detector = get_conflict_detector()
    return detector.predict(statement_a, statement_b)
