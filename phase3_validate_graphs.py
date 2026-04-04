"""
Phase 3: Validate patient knowledge graphs and generate a reproducibility report.

What this script validates:
1) Graph file integrity (JSON parse, required top-level keys).
2) Node/edge structural quality and distribution.
3) Temporal metadata coverage (added/last_confirmed/observed_dates).
4) Relationship quality (relation frequency, confidence ranges).

Usage:
  python phase3_validate_graphs.py
  python phase3_validate_graphs.py --graphs-dir data/graphs --report-path data/reports/phase3_graph_validation_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_GRAPH_KEYS = ["patient_id", "created_at", "nodes", "edges"]
REQUIRED_NODE_KEYS = ["type", "added", "last_confirmed", "confidence"]
REQUIRED_EDGE_KEYS = ["relation", "established", "confidence"]


def safe_mean(values: List[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def safe_median(values: List[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def validate_graph_file(file_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a single graph file and return stats + issues."""
    issues: List[str] = []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception as e:
        return {}, [f"{file_path.name}: JSON parse failed: {e}"]

    for key in REQUIRED_GRAPH_KEYS:
        if key not in graph:
            issues.append(f"{file_path.name}: missing top-level key '{key}'")

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    if not isinstance(nodes, list):
        issues.append(f"{file_path.name}: 'nodes' is not a list")
        nodes = []

    if not isinstance(edges, list):
        issues.append(f"{file_path.name}: 'edges' is not a list")
        edges = []

    node_types = Counter()
    relation_types = Counter()

    node_confidences: List[float] = []
    edge_confidences: List[float] = []

    nodes_missing_temporal = 0
    nodes_missing_observed_dates = 0
    isolated_non_patient_nodes = 0

    node_ids = set()
    for node in nodes:
        node_id = node.get("id")
        if node_id:
            node_ids.add(node_id)

        node_type = node.get("type", "UNKNOWN")
        node_types[node_type] += 1

        for key in REQUIRED_NODE_KEYS:
            if key not in node:
                issues.append(f"{file_path.name}: node missing '{key}'")

        if "added" not in node or "last_confirmed" not in node:
            nodes_missing_temporal += 1

        if "observed_dates" not in node:
            nodes_missing_observed_dates += 1

        conf = node.get("confidence")
        if isinstance(conf, (int, float)):
            node_confidences.append(float(conf))
            if conf < 0 or conf > 1:
                issues.append(f"{file_path.name}: node confidence out of [0,1]")

    # Build degree map from edges.
    degree_count = Counter()

    dangling_edges = 0
    edges_missing_temporal = 0

    for edge in edges:
        src = edge.get("source")
        dst = edge.get("target")

        if src:
            degree_count[src] += 1
        if dst:
            degree_count[dst] += 1

        if src not in node_ids or dst not in node_ids:
            dangling_edges += 1

        relation = edge.get("relation", "UNKNOWN")
        relation_types[relation] += 1

        for key in REQUIRED_EDGE_KEYS:
            if key not in edge:
                issues.append(f"{file_path.name}: edge missing '{key}'")

        if "established" not in edge:
            edges_missing_temporal += 1

        conf = edge.get("confidence")
        if isinstance(conf, (int, float)):
            edge_confidences.append(float(conf))
            if conf < 0 or conf > 1:
                issues.append(f"{file_path.name}: edge confidence out of [0,1]")

    # Count isolated non-patient nodes.
    for node in nodes:
        node_id = node.get("id")
        node_type = node.get("type")
        if node_type != "PATIENT" and degree_count.get(node_id, 0) == 0:
            isolated_non_patient_nodes += 1

    num_nodes = len(nodes)
    num_edges = len(edges)
    edges_per_node = round(num_edges / max(num_nodes, 1), 4)

    graph_stats = {
        "file": file_path.name,
        "patient_id": graph.get("patient_id", file_path.stem.replace("_graph", "")),
        "nodes": num_nodes,
        "edges": num_edges,
        "edges_per_node": edges_per_node,
        "node_types": dict(sorted(node_types.items())),
        "relation_types": dict(sorted(relation_types.items())),
        "node_confidence_avg": safe_mean(node_confidences),
        "node_confidence_median": safe_median(node_confidences),
        "edge_confidence_avg": safe_mean(edge_confidences),
        "edge_confidence_median": safe_median(edge_confidences),
        "nodes_missing_temporal": nodes_missing_temporal,
        "nodes_missing_observed_dates": nodes_missing_observed_dates,
        "edges_missing_temporal": edges_missing_temporal,
        "dangling_edges": dangling_edges,
        "isolated_non_patient_nodes": isolated_non_patient_nodes,
    }

    return graph_stats, issues


def build_phase3_validation_report(graphs_dir: Path) -> Dict[str, Any]:
    """Aggregate graph validation across all graph JSON files."""
    files = sorted(graphs_dir.glob("*_graph.json"))

    per_graph: List[Dict[str, Any]] = []
    all_issues: List[str] = []

    total_node_types = Counter()
    total_relation_types = Counter()

    nodes_counts: List[int] = []
    edges_counts: List[int] = []
    edges_per_node_values: List[float] = []

    for file_path in files:
        stats, issues = validate_graph_file(file_path)
        if stats:
            per_graph.append(stats)
            nodes_counts.append(stats["nodes"])
            edges_counts.append(stats["edges"])
            edges_per_node_values.append(stats["edges_per_node"])
            total_node_types.update(stats.get("node_types", {}))
            total_relation_types.update(stats.get("relation_types", {}))
        all_issues.extend(issues)

    summary = {
        "total_graph_files": len(files),
        "validated_graph_files": len(per_graph),
        "issue_count": len(all_issues),
        "nodes_min": min(nodes_counts) if nodes_counts else 0,
        "nodes_max": max(nodes_counts) if nodes_counts else 0,
        "nodes_avg": safe_mean(nodes_counts),
        "edges_min": min(edges_counts) if edges_counts else 0,
        "edges_max": max(edges_counts) if edges_counts else 0,
        "edges_avg": safe_mean(edges_counts),
        "edges_per_node_avg": safe_mean(edges_per_node_values),
        "edges_per_node_median": safe_median(edges_per_node_values),
        "node_types_total": dict(sorted(total_node_types.items())),
        "relation_types_total": dict(sorted(total_relation_types.items())),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "phase3_validate_graphs.py",
        "graphs_dir": str(graphs_dir).replace("\\", "/"),
        "summary": summary,
        "issues": all_issues,
        "per_graph": per_graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 3 graph outputs")
    parser.add_argument("--graphs-dir", default="data/graphs", help="Directory containing *_graph.json files")
    parser.add_argument(
        "--report-path",
        default="data/reports/phase3_graph_validation_report.json",
        help="Path to output JSON validation report",
    )
    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)
    report_path = Path(args.report_path)

    if not graphs_dir.exists() or not graphs_dir.is_dir():
        raise FileNotFoundError(f"Graphs directory not found: {graphs_dir}")

    report = build_phase3_validation_report(graphs_dir)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = report["summary"]

    print("=" * 60)
    print("Phase 3 validation complete")
    print("=" * 60)
    print(f"Graphs directory: {graphs_dir}")
    print(f"Validated files: {summary['validated_graph_files']}/{summary['total_graph_files']}")
    print(f"Issues: {summary['issue_count']}")
    print(f"Nodes avg (min-max): {summary['nodes_avg']} ({summary['nodes_min']} - {summary['nodes_max']})")
    print(f"Edges avg (min-max): {summary['edges_avg']} ({summary['edges_min']} - {summary['edges_max']})")
    print(f"Edges/node avg: {summary['edges_per_node_avg']}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
