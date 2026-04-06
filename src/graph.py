"""
Phase 3: Knowledge Graph Construction for GraphMed
Builds temporally-aware patient knowledge graphs with NetworkX
"""

import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import re


class PatientKnowledgeGraph:
    """
    Temporally-aware knowledge graph for a single patient.
    Stores medical entities as nodes and relationships as edges.
    Each node has timestamp and confidence that decays over time.
    """
    
    def __init__(self, patient_id: str):
        """
        Initialize an empty knowledge graph for a patient.
        
        Args:
            patient_id: Unique identifier for the patient
        """
        self.patient_id = patient_id
        self.G = nx.MultiDiGraph()  # MultiDiGraph allows multiple edges between nodes
        self.created_at = datetime.now().isoformat()
        
        # Add patient as root node
        self.G.add_node(patient_id, 
                        type="PATIENT",
                        name=patient_id,
                        added=self.created_at,
                        last_confirmed=self.created_at,
                        confidence=1.0)

        # Confidence policy: confidence is a freshness/trust signal, not truth probability.
        self.decay_rate_by_type = {
            "CONDITION": 0.015,
            "MEDICATION": 0.04,
            "LAB_VALUE": 0.05,
            "SYMPTOM": 0.08,
            "PROCEDURE": 0.03,
        }
        self.confidence_floor_by_type = {
            "CONDITION": 0.35,
            "MEDICATION": 0.2,
            "LAB_VALUE": 0.2,
            "SYMPTOM": 0.15,
            "PROCEDURE": 0.2,
        }
        self.global_floor = 0.2

    def _normalize_name(self, name: str) -> str:
        """Normalize entity names for stable node identity."""
        value = str(name).strip().lower()
        value = re.sub(r"\s+", " ", value)
        return value

    def _stable_node_id(self, entity_type: str, name: str) -> str:
        """Stable node IDs keep one node per entity per patient over time."""
        normalized = self._normalize_name(name)
        safe = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        return f"{entity_type}_{safe}"

    def _as_date_str(self, date: str) -> str:
        """Normalize dates to YYYY-MM-DD where possible."""
        value = str(date).strip()
        if not value:
            return datetime.now().strftime("%Y-%m-%d")
        # Fast path for common yyyy-mm-dd prefix.
        if re.match(r"^\d{4}-\d{2}-\d{2}$", value[:10]):
            return value[:10]

        # ISO timestamps and close variants.
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            pass

        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
            except Exception:
                pass
        # If parsing fails, keep first 10 chars (common ISO date prefix).
        return value[:10]

    def _parse_date(self, date: str) -> datetime:
        value = self._as_date_str(date)
        return datetime.strptime(value, "%Y-%m-%d")

    def _is_allergy_or_adverse_event(self, name: str, entity_type: str) -> bool:
        text = self._normalize_name(name)
        if entity_type != "CONDITION":
            return False
        allergy_markers = [
            "allergy",
            "allergic",
            "anaphylaxis",
            "adverse reaction",
            "drug reaction",
            "penicillin",
            "sulfa",
        ]
        return any(marker in text for marker in allergy_markers)

    def _is_chronic_condition(self, name: str, entity_type: str) -> bool:
        if entity_type != "CONDITION":
            return False
        text = self._normalize_name(name)
        chronic_markers = [
            "chronic",
            "diabetes",
            "hypertension",
            "ckd",
            "kidney disease",
            "coronary artery disease",
            "cad",
            "heart failure",
            "copd",
            "asthma",
            "hyperlipidemia",
            "hypothyroidism",
        ]
        return any(marker in text for marker in chronic_markers)

    def _type_decay_rate(self, node_data: Dict, default_decay_rate: float) -> float:
        entity_type = str(node_data.get("type", "")).upper()
        base_rate = self.decay_rate_by_type.get(entity_type, default_decay_rate)

        if self._is_allergy_or_adverse_event(node_data.get("name", ""), entity_type):
            return min(base_rate, 0.005)
        if self._is_chronic_condition(node_data.get("name", ""), entity_type):
            return min(base_rate, 0.01)
        return base_rate

    def _type_confidence_floor(self, node_data: Dict) -> float:
        entity_type = str(node_data.get("type", "")).upper()
        floor = self.confidence_floor_by_type.get(entity_type, self.global_floor)

        if self._is_allergy_or_adverse_event(node_data.get("name", ""), entity_type):
            return max(floor, 0.55)
        if self._is_chronic_condition(node_data.get("name", ""), entity_type):
            return max(floor, 0.45)
        return floor

    def _is_conflicted(self, node_data: Dict) -> bool:
        return str(node_data.get("status", "")).upper() == "CONFLICTED"

    def _recency_score(self, node_data: Dict, as_of_date: str) -> float:
        try:
            as_of = self._parse_date(as_of_date)
            last = self._parse_date(node_data.get("last_confirmed", as_of_date))
            months_old = max(0.0, (as_of - last).days / 30.0)
            return 1.0 / (1.0 + months_old)
        except Exception:
            return 0.5

    def _semantic_overlap_score(self, query: str, name: str) -> float:
        q_terms = {t for t in re.findall(r"[a-z0-9]+", self._normalize_name(query)) if t}
        n_terms = {t for t in re.findall(r"[a-z0-9]+", self._normalize_name(name)) if t}
        if not q_terms or not n_terms:
            return 0.0
        overlap = len(q_terms.intersection(n_terms))
        return overlap / float(max(len(q_terms), 1))

    def _source_reliability(self, node_data: Dict) -> float:
        if "source_reliability" in node_data:
            try:
                return max(0.0, min(1.0, float(node_data.get("source_reliability", 0.7))))
            except Exception:
                pass

        source = str(node_data.get("source", "")).lower()
        if "structured" in source or "ehr" in source:
            return 0.9
        if "manual" in source:
            return 0.75
        if "llm" in source or "inferred" in source:
            return 0.6
        return 0.7

    def _hybrid_score(
        self,
        query: str,
        node_data: Dict,
        as_of_date: str,
        semantic_weight: float = 0.45,
        confidence_weight: float = 0.25,
        recency_weight: float = 0.2,
        source_weight: float = 0.1,
    ) -> Dict[str, float]:
        semantic = self._semantic_overlap_score(query, node_data.get("name", ""))
        confidence = max(0.0, min(1.0, float(node_data.get("confidence", 1.0))))
        recency = self._recency_score(node_data, as_of_date)
        source_reliability = self._source_reliability(node_data)

        total = (
            semantic_weight * semantic
            + confidence_weight * confidence
            + recency_weight * recency
            + source_weight * source_reliability
        )
        return {
            "hybrid_score": total,
            "semantic": semantic,
            "confidence": confidence,
            "recency": recency,
            "source_reliability": source_reliability,
        }

    def retrieve_hybrid_facts(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 8,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid retrieval combining semantic relevance, confidence, recency,
        and source reliability.
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime("%Y-%m-%d")

        scored: List[Dict[str, Any]] = []
        for node_id, data in self.G.nodes(data=True):
            node_type = str(data.get("type", "")).upper()
            if node_type == "PATIENT":
                continue
            if entity_types and node_type not in {t.upper() for t in entity_types}:
                continue

            comp = self._hybrid_score(query, data, as_of_date)
            scored.append(
                {
                    "node_id": node_id,
                    "name": data.get("name", ""),
                    "type": node_type,
                    "last_confirmed": data.get("last_confirmed"),
                    "status": data.get("status", ""),
                    **comp,
                }
            )

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top = scored[: max(1, top_k)]

        conflicts = [
            x
            for x in scored
            if str(x.get("status", "")).upper() == "CONFLICTED"
        ]

        return {
            "query": query,
            "scoring_formula": "semantic_relevance + confidence + recency + source_reliability",
            "confidence_semantics": "Confidence is a freshness/trust signal, not truth probability.",
            "top_facts": top,
            "conflict_candidates": conflicts,
        }

    def _is_critical_historical(self, node_data: Dict) -> bool:
        entity_type = str(node_data.get("type", "")).upper()
        name = str(node_data.get("name", ""))
        return (
            self._is_allergy_or_adverse_event(name, entity_type)
            or self._is_chronic_condition(name, entity_type)
            or self._is_conflicted(node_data)
        )

    def retrieve_dual_channel_facts(
        self,
        query: str,
        current_top_k: int = 6,
        historical_top_k: int = 6,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Dual channel retrieval:
        - current_likely_state: what is most relevant and fresh now
        - critical_historical_facts: allergies/chronic/conflicts even if older
        """
        hybrid = self.retrieve_hybrid_facts(query, top_k=max(current_top_k, historical_top_k) * 3, as_of_date=as_of_date)
        all_scored = list(hybrid.get("top_facts", []))

        # retrieve_hybrid_facts returns only top_k; for dual-channel we need all nodes.
        # Recompute with very high top_k bounded by node count.
        total_nodes = max(1, self.G.number_of_nodes() - 1)
        full = self.retrieve_hybrid_facts(query, top_k=total_nodes, as_of_date=as_of_date)
        all_scored = list(full.get("top_facts", []))

        current_likely_state = all_scored[: max(1, current_top_k)]
        critical_historical = [x for x in all_scored if self._is_critical_historical(self.G.nodes[x["node_id"]])]
        critical_historical = critical_historical[: max(1, historical_top_k)]

        conflict_candidates = full.get("conflict_candidates", [])

        return {
            "query": query,
            "confidence_semantics": "Confidence is a freshness/trust signal, not truth probability.",
            "current_likely_state": current_likely_state,
            "critical_historical_facts": critical_historical,
            "conflict_candidates": conflict_candidates,
        }
    
    def add_entity(self, name: str, entity_type: str, date: str, 
                   confidence: float = 1.0, **kwargs) -> str:
        """
        Add a medical entity node to the graph.
        
        Args:
            name: Entity name (e.g., "Type 2 Diabetes", "Metformin")
            entity_type: Type of entity (CONDITION, MEDICATION, LAB_VALUE, SYMPTOM, PROCEDURE)
            date: Date when this entity was observed/added
            confidence: Initial confidence score (0.0 to 1.0)
            **kwargs: Additional attributes (value, unit, etc.)
        
        Returns:
            node_id: Unique identifier for the node
        """
        norm_date = self._as_date_str(date)
        clean_name = self._normalize_name(name)
        node_id = self._stable_node_id(entity_type, clean_name)

        if node_id in self.G.nodes:
            self.update_entity(node_id, norm_date, confirmed=True, **kwargs)
            existing_conf = self.G.nodes[node_id].get("confidence", 1.0)
            self.G.nodes[node_id]["confidence"] = max(existing_conf, confidence)
            return node_id

        self.G.add_node(
            node_id,
            name=clean_name,
            type=entity_type,
            added=norm_date,
            last_confirmed=norm_date,
            confidence=confidence,
            observed_dates=[norm_date],
            **kwargs,
        )

        self.G.add_edge(
            self.patient_id,
            node_id,
            relation="has",
            established=norm_date,
            confidence=1.0,
        )

        return node_id
    
    def add_relation(self, src: str, dst: str, relation: str, date: str,
                     confidence: float = 1.0, **kwargs):
        """
        Add a relationship between two entities.
        
        Args:
            src: Source node ID or name
            dst: Destination node ID or name
            relation: Type of relationship (managed_by, causes, treated_with, etc.)
            date: Date when relationship was established
            confidence: Confidence score for this relationship
            **kwargs: Additional attributes
        """
        norm_date = self._as_date_str(date)
        if src not in self.G.nodes or dst not in self.G.nodes:
            return

        # Keep one edge per (src, dst, relation) and update temporal metadata.
        for _, existing_dst, key, edge_data in self.G.out_edges(src, keys=True, data=True):
            if existing_dst == dst and edge_data.get("relation") == relation:
                old_conf = float(edge_data.get("confidence", confidence))
                edge_data["confidence"] = max(old_conf, confidence)
                edge_data["last_observed"] = norm_date
                edge_data["observations"] = int(edge_data.get("observations", 1)) + 1
                dates = edge_data.get("observed_dates", [])
                if norm_date not in dates:
                    dates.append(norm_date)
                edge_data["observed_dates"] = sorted(dates)
                for k, v in kwargs.items():
                    edge_data[k] = v
                return

        self.G.add_edge(
            src,
            dst,
            relation=relation,
            established=norm_date,
            last_observed=norm_date,
            observed_dates=[norm_date],
            observations=1,
            confidence=confidence,
            **kwargs,
        )
    
    def update_entity(self, node_id: str, date: str, 
                      new_value: Any = None, **kwargs):
        """
        Update an existing entity with new information.
        
        Args:
            node_id: ID of node to update
            date: Date of update
            new_value: New value (for lab values, etc.)
            **kwargs: Additional attributes to update
        """
        if node_id not in self.G.nodes:
            print(f"Warning: Node {node_id} not found")
            return
        
        norm_date = self._as_date_str(date)

        # Store history
        if 'history' not in self.G.nodes[node_id]:
            self.G.nodes[node_id]['history'] = []
        if 'observed_dates' not in self.G.nodes[node_id]:
            self.G.nodes[node_id]['observed_dates'] = []
        
        # Record old values before update
        old_values = {}
        for key in ['value', 'confidence']:
            if key in self.G.nodes[node_id]:
                old_values[key] = self.G.nodes[node_id][key]
        
        # Update node
        self.G.nodes[node_id]['last_confirmed'] = norm_date
        if norm_date not in self.G.nodes[node_id]['observed_dates']:
            self.G.nodes[node_id]['observed_dates'].append(norm_date)
        if new_value is not None:
            self.G.nodes[node_id]['value'] = new_value
        for key, val in kwargs.items():
            self.G.nodes[node_id][key] = val
        
        # Record in history
        self.G.nodes[node_id]['history'].append({
            'date': norm_date,
            'old': old_values,
            'new': {k: v for k, v in kwargs.items()}
        })
    
    def decay_confidence(self, current_date: str, decay_rate: float = 0.05):
        """
        Decay confidence of nodes not recently confirmed.
        
        Args:
            current_date: Current date for decay calculation
            decay_rate: Confidence decay per month (default 5%)
        """
        cur_date = datetime.strptime(self._as_date_str(current_date), '%Y-%m-%d')
        
        for node in self.G.nodes:
            node_data = self.G.nodes[node]
            
            # Don't decay patient node
            if node_data.get('type') == 'PATIENT':
                continue
            
            last_confirmed = datetime.strptime(self._as_date_str(node_data.get('last_confirmed', current_date)), '%Y-%m-%d')
            months_old = (cur_date - last_confirmed).days / 30.0
            
            # Type-specific decay with category safety floors.
            old_conf = node_data.get('confidence', 1.0)
            type_decay_rate = self._type_decay_rate(node_data, decay_rate)
            floor = self._type_confidence_floor(node_data)
            new_conf = max(floor, old_conf - months_old * type_decay_rate)
            
            node_data['confidence'] = new_conf
            node_data['decayed'] = True
            node_data['decay_rate_applied'] = type_decay_rate
            node_data['confidence_floor_applied'] = floor
    
    def get_entity(self, name: str, entity_type: str = None) -> Optional[str]:
        """
        Find node ID for an entity by name.
        
        Args:
            name: Entity name to search for
            entity_type: Optional type filter
        
        Returns:
            node_id if found, None otherwise
        """
        clean_name = self._normalize_name(name)
        
        for node, data in self.G.nodes(data=True):
            if self._normalize_name(data.get('name', '')) == clean_name:
                if entity_type is None or data.get('type') == entity_type:
                    return node
        return None

    def visualize(self, output_file: str = "graph.html", notebook: bool = False) -> str:
        """
        Visualize graph using PyVis and save to an HTML file.

        Returns:
            Path to generated HTML file
        """
        try:
            from pyvis.network import Network
        except Exception as e:
            raise RuntimeError("PyVis is not installed. Install with: pip install pyvis") from e

        net = Network(height="700px", width="100%", directed=True, notebook=notebook)

        color_map = {
            "PATIENT": "#1f2937",
            "CONDITION": "#b91c1c",
            "MEDICATION": "#0369a1",
            "LAB_VALUE": "#047857",
            "SYMPTOM": "#a21caf",
            "PROCEDURE": "#b45309",
        }

        for node_id, data in self.G.nodes(data=True):
            node_type = data.get("type", "UNKNOWN")
            label = data.get("name", node_id)
            confidence = data.get("confidence", 1.0)
            title = (
                f"Type: {node_type}<br>"
                f"Name: {label}<br>"
                f"Added: {data.get('added')}<br>"
                f"Last confirmed: {data.get('last_confirmed')}<br>"
                f"Confidence: {confidence:.2f}"
            )
            net.add_node(
                node_id,
                label=label if node_type != "PATIENT" else self.patient_id,
                color=color_map.get(node_type, "#6b7280"),
                title=title,
                size=30 if node_type == "PATIENT" else 18,
            )

        for src, dst, data in self.G.edges(data=True):
            relation = data.get("relation", "related_to")
            edge_title = f"Relation: {relation}<br>Date: {data.get('established')}<br>Confidence: {data.get('confidence', 1.0):.2f}"
            net.add_edge(src, dst, label=relation, title=edge_title)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))
        return str(output_path)
    
    def get_entities_by_type(self, entity_type: str) -> List[Tuple[str, Dict]]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type to filter by (CONDITION, MEDICATION, etc.)
        
        Returns:
            List of (node_id, node_data) tuples
        """
        results = []
        for node, data in self.G.nodes(data=True):
            if data.get('type') == entity_type:
                results.append((node, data))
        return results
    
    def get_lab_trend(self, lab_name: str) -> List[Dict]:
        """
        Get trend data for a specific lab value.
        
        Args:
            lab_name: Name of the lab value
        
        Returns:
            List of lab readings with dates
        """
        trend = []
        clean_name = lab_name.lower().strip()
        
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'LAB_VALUE' and clean_name in data.get('name', '').lower():
                trend.append({
                    'date': data.get('added'),
                    'value': data.get('value'),
                    'confidence': data.get('confidence')
                })
        
        # Sort by date
        trend.sort(key=lambda x: x['date'])
        return trend
    
    def get_conflicts(self) -> List[Dict]:
        """
        Get all flagged conflicts in the graph.
        
        Returns:
            List of conflict records
        """
        conflicts = []
        for node, data in self.G.nodes(data=True):
            if data.get('status') == 'CONFLICTED':
                conflicts.append({
                    'node': node,
                    'name': data.get('name'),
                    'type': data.get('type'),
                    'conflict_note': data.get('conflict_note'),
                    'date': data.get('last_confirmed')
                })
        return conflicts
    
    def summary(self) -> Dict:
        """
        Get a summary of the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        summary = {
            'patient_id': self.patient_id,
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'node_types': {},
            'conflicts': len(self.get_conflicts())
        }
        
        for node, data in self.G.nodes(data=True):
            node_type = data.get('type', 'UNKNOWN')
            summary['node_types'][node_type] = summary['node_types'].get(node_type, 0) + 1
        
        return summary
    
    def to_dict(self) -> Dict:
        """
        Convert graph to dictionary for serialization.
        """
        # Convert nodes
        nodes = []
        for node, data in self.G.nodes(data=True):
            nodes.append({
                'id': node,
                **data
            })
        
        # Convert edges
        edges = []
        for src, dst, data in self.G.edges(data=True):
            edges.append({
                'source': src,
                'target': dst,
                **data
            })
        
        return {
            'patient_id': self.patient_id,
            'created_at': self.created_at,
            'nodes': nodes,
            'edges': edges
        }
    
    def save(self, filepath: str):
        """Save graph to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str):
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = cls(data['patient_id'])
        graph.created_at = data['created_at']
        
        # Clear default nodes/edges
        graph.G.clear()
        
        # Add nodes
        for node in data['nodes']:
            node_id = node.pop('id')
            graph.G.add_node(node_id, **node)
        
        # Add edges
        for edge in data['edges']:
            src = edge.pop('source')
            dst = edge.pop('target')
            graph.G.add_edge(src, dst, **edge)
        
        return graph


def build_graph_from_patient_data(patient_data: Dict[str, Any]) -> PatientKnowledgeGraph:
    """
    Build a knowledge graph from processed patient data.
    
    Args:
        patient_data: Processed patient JSON with visits and extracted entities
    
    Returns:
        PatientKnowledgeGraph object
    """
    patient_id = patient_data['patient_id']
    pkg = PatientKnowledgeGraph(patient_id)
    
    visits = sorted(patient_data.get('visits', []), key=lambda v: v.get('date', ''))

    for visit in visits:
        date = visit.get('date', '')
        extracted = visit.get('extracted', {})

        # Fallback to top-level fields if extracted is missing/incomplete.
        conditions = extracted.get('conditions', []) or visit.get('diagnoses', [])
        medications = extracted.get('medications', []) or visit.get('medications', [])
        symptoms = extracted.get('symptoms', []) or visit.get('symptoms', [])
        lab_values = extracted.get('lab_values', {}) or visit.get('labs', {})
        procedures = extracted.get('procedures', []) or []
        relationships = extracted.get('relationships', []) or []
        
        # Process conditions
        condition_nodes: List[str] = []
        medication_nodes: List[str] = []
        symptom_nodes: List[str] = []
        lab_nodes: List[str] = []

        for condition in conditions:
            node_id = pkg.get_entity(condition, 'CONDITION')
            if node_id is None:
                node_id = pkg.add_entity(condition, 'CONDITION', date)
            else:
                pkg.update_entity(node_id, date, confirmed=True)
            condition_nodes.append(node_id)
        
        # Process medications
        for med in medications:
            node_id = pkg.get_entity(med, 'MEDICATION')
            if node_id is None:
                node_id = pkg.add_entity(med, 'MEDICATION', date)
            else:
                pkg.update_entity(node_id, date, confirmed=True)
            medication_nodes.append(node_id)
        
        # Process lab values
        for lab_name, lab_value in lab_values.items():
            node_id = pkg.get_entity(lab_name, 'LAB_VALUE')
            if node_id is None:
                node_id = pkg.add_entity(lab_name, 'LAB_VALUE', date, value=lab_value)
            else:
                pkg.update_entity(node_id, date, new_value=lab_value)
            lab_nodes.append(node_id)
        
        # Process symptoms
        for symptom in symptoms:
            node_id = pkg.get_entity(symptom, 'SYMPTOM')
            if node_id is None:
                node_id = pkg.add_entity(symptom, 'SYMPTOM', date)
            else:
                pkg.update_entity(node_id, date, confirmed=True)
            symptom_nodes.append(node_id)

        # Process procedures
        for procedure in procedures:
            node_id = pkg.get_entity(procedure, 'PROCEDURE')
            if node_id is None:
                node_id = pkg.add_entity(procedure, 'PROCEDURE', date)
            else:
                pkg.update_entity(node_id, date, confirmed=True)

        # Add fallback baseline relations only when extraction provides no explicit relationships.
        if not relationships:
            for c in condition_nodes:
                for m in medication_nodes:
                    pkg.add_relation(c, m, 'managed_by', date)
                for s in symptom_nodes:
                    pkg.add_relation(c, s, 'has_symptom', date)
                for l in lab_nodes:
                    pkg.add_relation(c, l, 'measured_by', date)
        
        # Process relationships
        for rel in relationships:
            subject = rel.get('subject', '')
            predicate = rel.get('predicate', '')
            obj = rel.get('object', '')
            confidence = float(rel.get('confidence', 1.0)) if isinstance(rel.get('confidence', 1.0), (int, float)) else 1.0
            
            if subject and obj and predicate:
                src_id = pkg.get_entity(subject)
                dst_id = pkg.get_entity(obj)
                
                if src_id and dst_id:
                    pkg.add_relation(src_id, dst_id, predicate, date, confidence=confidence)

        # Decay confidence after each visit to keep temporal trust dynamic.
        pkg.decay_confidence(date)
    
    return pkg


def build_graphs_for_all_patients(input_dir: str = "data/patients_processed",
                                   output_dir: str = "data/graphs",
                                   limit: int = None,
                                   create_visualizations: bool = True,
                                   viz_dir: str = "data/graphs/viz") -> List[str]:
    """
    Build knowledge graphs for all processed patients.
    
    Args:
        input_dir: Directory with processed patient JSON files
        output_dir: Directory to save graph JSON files
        limit: Maximum number of patients to process
        create_visualizations: Whether to generate PyVis HTML graph files
        viz_dir: Directory for visualization HTML files
    
    Returns:
        List of patient IDs that were processed
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all processed patient files
    patient_files = sorted(Path(input_dir).glob("*.json"))
    
    if limit is not None:
        patient_files = patient_files[:limit]

    if create_visualizations:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🏥 PHASE 3: KNOWLEDGE GRAPH CONSTRUCTION")
    print(f"{'='*60}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Patients: {len(patient_files)}")
    print(f"{'='*60}\n")
    
    processed = []
    errors = []
    
    for file_path in patient_files:
        try:
            # Load patient data
            with open(file_path, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            patient_id = patient_data['patient_id']
            print(f"Building graph for {patient_id}...", end=" ")
            
            # Build graph
            pkg = build_graph_from_patient_data(patient_data)
            
            # Save graph
            output_path = Path(output_dir) / f"{patient_id}_graph.json"
            pkg.save(str(output_path))

            if create_visualizations:
                viz_path = Path(viz_dir) / f"{patient_id}_graph.html"
                pkg.visualize(str(viz_path))
            
            # Print summary
            summary = pkg.summary()
            print(f"✅ {summary['total_nodes']} nodes, {summary['total_edges']} edges")
            
            processed.append(patient_id)
            
        except Exception as e:
            errors.append((file_path.name, str(e)))
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"📈 GRAPH CONSTRUCTION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Processed: {len(processed)}/{len(patient_files)}")
    print(f"❌ Errors: {len(errors)}")
    print(f"📁 Location: {output_dir}")
    print(f"{'='*60}")
    
    return processed


if __name__ == "__main__":
    # Quick test
    test_patient = {
        "patient_id": "TEST001",
        "visits": [
            {
                "date": "2024-01-15",
                "extracted": {
                    "conditions": ["Type 2 Diabetes", "Hypertension"],
                    "medications": ["Metformin"],
                    "lab_values": {"HbA1c": 7.2, "BP_systolic": 140, "BP_diastolic": 90},
                    "symptoms": ["Fatigue", "Polyuria"],
                    "relationships": []
                }
            }
        ]
    }
    
    print("Testing graph construction...")
    pkg = build_graph_from_patient_data(test_patient)
    print(f"Created graph with {pkg.G.number_of_nodes()} nodes")
    print(f"Patient summary: {pkg.summary()}")
    print("\n✅ Graph module ready!")