"""
Phase 10 driver script.
Runs the evaluation-driven knowledge graph updater from Phase 9 artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.knowledge_updater import EvaluationDrivenGraphUpdater


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_phase10(
    persist_dir: str = "data",
    phase9_output_dir: str = "evaluation/results/phase9",
) -> int:
    output_dir = Path(phase9_output_dir)
    e1_path = output_dir / "experiment1_factual_accuracy.json"
    e2_path = output_dir / "experiment2_longitudinal_consistency.json"

    e1_result = _load_json(e1_path)
    e2_result = _load_json(e2_path)

    if e1_result is None or e2_result is None:
        print("[Phase10] Missing Phase 9 artifacts required for updater.")
        print(f"[Phase10] Required: {e1_path}")
        print(f"[Phase10] Required: {e2_path}")
        print("[Phase10] Run Phase 9 first: python phase9_run_evaluation.py")
        return 1

    updater = EvaluationDrivenGraphUpdater(
        persist_dir=persist_dir,
        audit_dir=phase9_output_dir,
    )
    summary = updater.apply_updates(e1_result=e1_result, e2_result=e2_result)

    print("\n" + "=" * 72)
    print("GRAPHMED PHASE 10 KNOWLEDGE UPDATER")
    print("=" * 72)
    print(f"Run ID: {summary.get('run_id')}")
    print(f"Patients evaluated: {summary.get('patients_evaluated_for_circumstances')}")
    print(f"Patients updated: {summary.get('patients_updated')}")
    print(f"Total update events: {summary.get('total_update_events')}")
    print(f"Audit log: {summary.get('audit_log_path')}")
    print("=" * 72)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 10 evaluation-driven graph updater from Phase 9 artifacts."
    )
    parser.add_argument(
        "--persist-dir",
        default="data",
        help="Base persisted data directory (default: data)",
    )
    parser.add_argument(
        "--phase9-output-dir",
        default="evaluation/results/phase9",
        help="Phase 9 output directory containing experiment JSON files",
    )
    args = parser.parse_args()

    raise SystemExit(
        run_phase10(
            persist_dir=args.persist_dir,
            phase9_output_dir=args.phase9_output_dir,
        )
    )


if __name__ == "__main__":
    main()
