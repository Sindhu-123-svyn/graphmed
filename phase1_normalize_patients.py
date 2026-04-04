"""
Phase 1: Normalize patient data and generate validation report.

What this script does:
1) Auto-fixes labs.BP into labs.BP_systolic and labs.BP_diastolic.
2) Validates patient/visit schema across all files.
3) Writes a reproducible JSON report with counts and details.

Usage examples:
  python phase1_normalize_patients.py
  python phase1_normalize_patients.py --input-dir data/patients --report-path data/reports/patient_normalization_report.json
  python phase1_normalize_patients.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_TOP_KEYS = ["patient_id", "visits"]
REQUIRED_VISIT_KEYS = [
    "visit_id",
    "date",
    "note",
    "labs",
    "medications",
    "diagnoses",
    "symptoms",
]


@dataclass
class BPFixResult:
    changed: bool
    parsed: bool
    original_bp: Any
    systolic: Optional[int]
    diastolic: Optional[int]


def parse_bp_value(bp_value: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse BP from various forms into systolic/diastolic integers.

    Supported examples:
    - "140/90"
    - "BP 140/90 mmHg"
    - [140, 90]
    - {"systolic": 140, "diastolic": 90}
    """
    if isinstance(bp_value, str):
        match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", bp_value)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    if isinstance(bp_value, (list, tuple)) and len(bp_value) >= 2:
        try:
            return int(bp_value[0]), int(bp_value[1])
        except (TypeError, ValueError):
            return None, None

    if isinstance(bp_value, dict):
        sys_val = bp_value.get("systolic")
        dia_val = bp_value.get("diastolic")
        try:
            if sys_val is not None and dia_val is not None:
                return int(sys_val), int(dia_val)
        except (TypeError, ValueError):
            return None, None

    return None, None


def normalize_visit_bp_keys(visit: Dict[str, Any]) -> BPFixResult:
    """Normalize BP key in a single visit's labs dictionary."""
    labs = visit.get("labs")
    if not isinstance(labs, dict):
        return BPFixResult(False, False, None, None, None)

    if "BP" not in labs:
        return BPFixResult(False, False, None, None, None)

    original_bp = labs.get("BP")
    systolic, diastolic = parse_bp_value(original_bp)

    changed = False
    parsed = systolic is not None and diastolic is not None

    if parsed:
        # Only fill missing values; keep existing explicit values untouched.
        if "BP_systolic" not in labs:
            labs["BP_systolic"] = systolic
            changed = True
        if "BP_diastolic" not in labs:
            labs["BP_diastolic"] = diastolic
            changed = True

    # Remove the non-standard BP key once processed.
    labs.pop("BP", None)
    changed = True

    return BPFixResult(changed, parsed, original_bp, systolic, diastolic)


def validate_file_schema(data: Dict[str, Any], filename: str) -> List[str]:
    """Return list of schema issues for a single patient file."""
    issues: List[str] = []

    for key in REQUIRED_TOP_KEYS:
        if key not in data:
            issues.append(f"{filename}: missing top-level key '{key}'")

    visits = data.get("visits")
    if not isinstance(visits, list):
        issues.append(f"{filename}: 'visits' is not a list")
        return issues

    for idx, visit in enumerate(visits, start=1):
        if not isinstance(visit, dict):
            issues.append(f"{filename}: visit #{idx} is not an object")
            continue

        for key in REQUIRED_VISIT_KEYS:
            if key not in visit:
                issues.append(f"{filename}: visit #{idx} missing key '{key}'")

        labs = visit.get("labs")
        if labs is not None and not isinstance(labs, dict):
            issues.append(f"{filename}: visit #{idx} 'labs' is not an object")

    return issues


def normalize_dataset(input_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Normalize all patient files and return report payload."""
    files = sorted(input_dir.glob("*.json"))

    top_key_freq: Counter = Counter()
    visit_key_freq: Counter = Counter()
    lab_key_freq_before: Counter = Counter()
    lab_key_freq_after: Counter = Counter()

    schema_issues: List[str] = []
    parse_failures: List[Dict[str, Any]] = []
    file_summaries: List[Dict[str, Any]] = []

    total_visits = 0
    bp_found = 0
    bp_fixed = 0
    files_modified = 0

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        filename = file_path.name
        changed_this_file = False
        file_bp_found = 0
        file_bp_fixed = 0

        for key in data.keys():
            top_key_freq[key] += 1

        schema_issues.extend(validate_file_schema(data, filename))

        visits = data.get("visits", []) if isinstance(data.get("visits", []), list) else []
        total_visits += len(visits)

        for v_idx, visit in enumerate(visits, start=1):
            if isinstance(visit, dict):
                for key in visit.keys():
                    visit_key_freq[key] += 1

            labs = visit.get("labs") if isinstance(visit, dict) else None
            if isinstance(labs, dict):
                for lk in labs.keys():
                    lab_key_freq_before[lk] += 1

                if "BP" in labs:
                    bp_found += 1
                    file_bp_found += 1
                    result = normalize_visit_bp_keys(visit)
                    if result.changed:
                        changed_this_file = True
                    if result.parsed:
                        bp_fixed += 1
                        file_bp_fixed += 1
                    else:
                        parse_failures.append(
                            {
                                "file": filename,
                                "visit_index": v_idx,
                                "visit_id": visit.get("visit_id"),
                                "bp_value": result.original_bp,
                            }
                        )

            labs_after = visit.get("labs") if isinstance(visit, dict) else None
            if isinstance(labs_after, dict):
                for lk in labs_after.keys():
                    lab_key_freq_after[lk] += 1

        if changed_this_file:
            files_modified += 1
            if not dry_run:
                with file_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.write("\n")

        file_summaries.append(
            {
                "file": filename,
                "visits": len(visits),
                "bp_found": file_bp_found,
                "bp_fixed": file_bp_fixed,
                "modified": changed_this_file,
            }
        )

    visit_counts = [item["visits"] for item in file_summaries]
    visit_count_summary = {
        "min": min(visit_counts) if visit_counts else 0,
        "max": max(visit_counts) if visit_counts else 0,
        "avg": round(sum(visit_counts) / len(visit_counts), 2) if visit_counts else 0,
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "phase1_normalize_patients.py",
        "dry_run": dry_run,
        "input_dir": str(input_dir).replace("\\", "/"),
        "total_files": len(files),
        "total_visits": total_visits,
        "files_modified": files_modified,
        "bp_key_found_count": bp_found,
        "bp_key_fixed_count": bp_fixed,
        "bp_key_parse_failures": parse_failures,
        "schema_issue_count": len(schema_issues),
        "schema_issues": schema_issues,
        "visit_count_summary": visit_count_summary,
        "top_level_key_frequency": dict(sorted(top_key_freq.items())),
        "visit_key_frequency": dict(sorted(visit_key_freq.items())),
        "lab_key_frequency_before": dict(sorted(lab_key_freq_before.items())),
        "lab_key_frequency_after": dict(sorted(lab_key_freq_after.items())),
        "file_summaries": file_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize patient data and generate validation report")
    parser.add_argument("--input-dir", default="data/patients", help="Directory containing patient JSON files")
    parser.add_argument(
        "--report-path",
        default="data/reports/patient_normalization_report.json",
        help="Path to write JSON report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and report only, do not write patient file changes",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    report_path = Path(args.report_path)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    report = normalize_dataset(input_dir=input_dir, dry_run=args.dry_run)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("=" * 60)
    print("Phase 1 normalization complete")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Total files: {report['total_files']}")
    print(f"Total visits: {report['total_visits']}")
    print(f"Files modified: {report['files_modified']}")
    print(f"BP key found: {report['bp_key_found_count']}")
    print(f"BP key fixed: {report['bp_key_fixed_count']}")
    print(f"Schema issues: {report['schema_issue_count']}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
