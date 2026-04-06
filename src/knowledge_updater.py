"""
Evaluation-driven graph updater for GraphMed.

Applies automatic graph evolution updates based on Phase 9 outcomes and
newly detected visit data, then persists an audit trail for each applied change.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.graph import PatientKnowledgeGraph
from src.graph_evolution import evolve_patient_graph


class EvaluationDrivenGraphUpdater:
    """
    Automatically evolves patient graphs based on evaluation circumstances.

    Triggers include:
    - Low E1 factual score for a patient
    - Low E2 longitudinal consistency for a patient
    - GraphMed underperforming baseline by margin
    - Newly detected visit date not reflected in graph timeline
    """

    def __init__(
        self,
        persist_dir: str = "data",
        audit_dir: str = "evaluation/results/phase9",
        factscore_threshold: float = 0.55,
        consistency_threshold: float = 0.60,
        underperform_margin: float = 0.05,
    ):
        self.persist_dir = Path(persist_dir)
        self.graphs_dir = self.persist_dir / "graphs"
        self.patients_dir = self.persist_dir / "patients_processed"
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        self.factscore_threshold = factscore_threshold
        self.consistency_threshold = consistency_threshold
        self.underperform_margin = underperform_margin

        self.audit_jsonl = self.audit_dir / "evaluation_graph_update_audit.jsonl"
        self.last_run_summary_json = self.audit_dir / "evaluation_graph_update_summary.json"

    def _utc_now(self) -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _load_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        fp = self.patients_dir / f"{patient_id}.json"
        if not fp.exists():
            return None
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)

    def _all_patient_ids(self) -> List[str]:
        if not self.patients_dir.exists():
            return []
        return sorted([p.stem for p in self.patients_dir.glob("*.json")])

    def _graph_path(self, patient_id: str) -> Path:
        return self.graphs_dir / f"{patient_id}_graph.json"

    def _graph_summary(self, patient_id: str) -> Dict[str, Any]:
        graph_path = self._graph_path(patient_id)
        if not graph_path.exists():
            return {
                "exists": False,
                "patient_id": patient_id,
                "total_nodes": 0,
                "total_edges": 0,
                "conflicts": 0,
                "latest_confirmed_date": None,
            }

        graph = PatientKnowledgeGraph.load(str(graph_path))
        summary = graph.summary()

        latest_dates: List[str] = []
        for _, data in graph.G.nodes(data=True):
            d = data.get("last_confirmed") or data.get("added")
            if d:
                latest_dates.append(str(d)[:10])

        return {
            "exists": True,
            "patient_id": patient_id,
            "total_nodes": int(summary.get("total_nodes", 0)),
            "total_edges": int(summary.get("total_edges", 0)),
            "conflicts": int(summary.get("conflicts", 0)),
            "latest_confirmed_date": max(latest_dates) if latest_dates else None,
        }

    def _normalize_visit_for_evolution(self, visit: Dict[str, Any]) -> Dict[str, Any]:
        extracted = dict(visit.get("extracted", {}) or {})

        if "conditions" not in extracted:
            extracted["conditions"] = list(visit.get("diagnoses", []) or [])
        if "medications" not in extracted:
            extracted["medications"] = list(visit.get("medications", []) or [])
        if "symptoms" not in extracted:
            extracted["symptoms"] = list(visit.get("symptoms", []) or [])
        if "lab_values" not in extracted:
            extracted["lab_values"] = dict(visit.get("labs", {}) or {})

        return {
            "date": str(visit.get("date", ""))[:10],
            "extracted": {
                "conditions": list(extracted.get("conditions", []) or []),
                "medications": list(extracted.get("medications", []) or []),
                "symptoms": list(extracted.get("symptoms", []) or []),
                "lab_values": dict(extracted.get("lab_values", {}) or {}),
            },
        }

    def _latest_visit(self, patient: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        visits = list(patient.get("visits", []) or [])
        if not visits:
            return None
        visits.sort(key=lambda v: str(v.get("date", "")))
        return visits[-1]

    def _first_visit(self, patient: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        visits = list(patient.get("visits", []) or [])
        if not visits:
            return None
        visits.sort(key=lambda v: str(v.get("date", "")))
        return visits[0]

    def _needs_new_data_sync(self, patient_id: str, latest_visit_date: str) -> bool:
        summary = self._graph_summary(patient_id)
        if not summary.get("exists"):
            return True
        graph_date = summary.get("latest_confirmed_date")
        if not graph_date:
            return True
        return str(latest_visit_date)[:10] > str(graph_date)[:10]

    def _aggregate_e1_scores(self, e1_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        bucket: Dict[str, Dict[str, List[float]]] = {}
        for r in e1_rows:
            pid = str(r.get("patient_id", "")).strip()
            system = str(r.get("system", "")).strip().lower()
            if not pid or system not in {"graphmed", "baseline"}:
                continue
            score = self._safe_float(r.get("factscore", 0.0))
            bucket.setdefault(pid, {"graphmed": [], "baseline": []})
            bucket[pid][system].append(score)

        out: Dict[str, Dict[str, float]] = {}
        for pid, vals in bucket.items():
            gm = vals["graphmed"]
            bl = vals["baseline"]
            out[pid] = {
                "graphmed_factscore_mean": (sum(gm) / len(gm)) if gm else 0.0,
                "baseline_factscore_mean": (sum(bl) / len(bl)) if bl else 0.0,
            }
        return out

    def _aggregate_e2_scores(self, e2_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        bucket: Dict[str, Dict[str, List[float]]] = {}
        for r in e2_rows:
            pid = str(r.get("patient_id", "")).strip()
            system = str(r.get("system", "")).strip().lower()
            if not pid or system not in {"graphmed", "baseline"}:
                continue
            score = self._safe_float(r.get("consistency_score", 0.0))
            bucket.setdefault(pid, {"graphmed": [], "baseline": []})
            bucket[pid][system].append(score)

        out: Dict[str, Dict[str, float]] = {}
        for pid, vals in bucket.items():
            gm = vals["graphmed"]
            bl = vals["baseline"]
            out[pid] = {
                "graphmed_consistency_mean": (sum(gm) / len(gm)) if gm else 0.0,
                "baseline_consistency_mean": (sum(bl) / len(bl)) if bl else 0.0,
            }
        return out

    def _build_circumstances(
        self,
        e1_scores: Dict[str, Dict[str, float]],
        e2_scores: Dict[str, Dict[str, float]],
    ) -> Dict[str, List[str]]:
        reasons: Dict[str, List[str]] = {}

        all_patient_ids = set(e1_scores.keys()) | set(e2_scores.keys()) | set(self._all_patient_ids())
        for pid in sorted(all_patient_ids):
            pid_reasons: List[str] = []

            if pid in e1_scores:
                gm = e1_scores[pid].get("graphmed_factscore_mean", 0.0)
                bl = e1_scores[pid].get("baseline_factscore_mean", 0.0)
                if gm < self.factscore_threshold:
                    pid_reasons.append("low_factscore")
                if (bl - gm) > self.underperform_margin:
                    pid_reasons.append("factscore_underperform_vs_baseline")

            if pid in e2_scores:
                gm = e2_scores[pid].get("graphmed_consistency_mean", 0.0)
                bl = e2_scores[pid].get("baseline_consistency_mean", 0.0)
                if gm < self.consistency_threshold:
                    pid_reasons.append("low_longitudinal_consistency")
                if (bl - gm) > self.underperform_margin:
                    pid_reasons.append("consistency_underperform_vs_baseline")

            patient = self._load_patient(pid)
            latest = self._latest_visit(patient) if patient else None
            if latest:
                latest_date = str(latest.get("date", ""))[:10]
                if latest_date and self._needs_new_data_sync(pid, latest_date):
                    pid_reasons.append("new_visit_data_detected")

            if pid_reasons:
                # Preserve insertion order while deduping.
                seen = set()
                reasons[pid] = [r for r in pid_reasons if not (r in seen or seen.add(r))]

        return reasons

    def _operation_counts(self, evolution_results: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"ADD": 0, "UPDATE": 0, "CONFLICT": 0}
        for r in evolution_results:
            op = str(r.get("operation", "")).strip().upper()
            if op in counts:
                counts[op] += 1
        return counts

    def _append_audit_entries(self, entries: List[Dict[str, Any]]):
        if not entries:
            return
        self.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_jsonl, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def _apply_single_update(
        self,
        patient_id: str,
        visit_payload: Dict[str, Any],
        reason: str,
        run_id: str,
        score_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        before = self._graph_summary(patient_id)
        result = evolve_patient_graph(patient_id, visit_payload, str(self.graphs_dir))
        after = self._graph_summary(patient_id)
        evolution_results = list(result.get("evolution_results", []) or [])

        return {
            "timestamp": self._utc_now(),
            "run_id": run_id,
            "event": "evaluation_driven_graph_update",
            "reason": reason,
            "patient_id": patient_id,
            "visit_date": str(visit_payload.get("date", ""))[:10],
            "scores": score_snapshot,
            "before": before,
            "after": after,
            "operation_counts": self._operation_counts(evolution_results),
            "total_entity_operations": len(evolution_results),
            "evolution_summary": result.get("summary", {}),
        }

    def apply_updates(
        self,
        e1_result: Dict[str, Any],
        e2_result: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_id = run_id or self._utc_now().replace(":", "-")

        e1_rows = list(e1_result.get("rows", []) or [])
        e2_rows = list(e2_result.get("rows", []) or [])
        e1_scores = self._aggregate_e1_scores(e1_rows)
        e2_scores = self._aggregate_e2_scores(e2_rows)

        circumstances = self._build_circumstances(e1_scores, e2_scores)

        entries: List[Dict[str, Any]] = []
        updated_patients = 0

        for patient_id, reasons in circumstances.items():
            patient = self._load_patient(patient_id)
            if not patient:
                continue

            latest_visit = self._latest_visit(patient)
            first_visit = self._first_visit(patient)
            if not latest_visit:
                continue

            score_snapshot = {
                "e1": e1_scores.get(patient_id, {}),
                "e2": e2_scores.get(patient_id, {}),
            }

            # Always sync latest visit first for automatic new-data propagation.
            latest_payload = self._normalize_visit_for_evolution(latest_visit)
            entries.append(
                self._apply_single_update(
                    patient_id=patient_id,
                    visit_payload=latest_payload,
                    reason="+".join(reasons),
                    run_id=run_id,
                    score_snapshot=score_snapshot,
                )
            )
            updated_patients += 1

            # If longitudinal consistency is poor, re-anchor with first visit context too.
            if "low_longitudinal_consistency" in reasons and first_visit:
                first_payload = self._normalize_visit_for_evolution(first_visit)
                first_date = str(first_payload.get("date", ""))[:10]
                latest_date = str(latest_payload.get("date", ""))[:10]
                if first_date and first_date != latest_date:
                    entries.append(
                        self._apply_single_update(
                            patient_id=patient_id,
                            visit_payload=first_payload,
                            reason="longitudinal_reanchor",
                            run_id=run_id,
                            score_snapshot=score_snapshot,
                        )
                    )

        self._append_audit_entries(entries)

        summary = {
            "run_id": run_id,
            "timestamp": self._utc_now(),
            "auto_update_enabled": True,
            "patients_evaluated_for_circumstances": len(circumstances),
            "patients_updated": updated_patients,
            "total_update_events": len(entries),
            "audit_log_path": str(self.audit_jsonl),
            "circumstances": circumstances,
        }

        with open(self.last_run_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
