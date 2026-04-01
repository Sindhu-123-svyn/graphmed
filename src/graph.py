"""
Phase 3: Knowledge Graph Construction for GraphMed
Builds temporally-aware patient knowledge graphs with NetworkX
"""

import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path


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
                        added=self.created_at,
                        last_confirmed=self.created_at,
                        confidence=1.0)
    
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
        node_id = f"{entity_type}_{name}_{date}"
        
        # Clean name for consistency
        clean_name = name.lower().strip()
        
        self.G.add_node(node_id,
                        name=clean_name,
                        type=entity_type,
                        added=date,
                        last_confirmed=date,
                        confidence=confidence,
                        **kwargs)
        
        # Connect to patient
        self.G.add_edge(self.patient_id, node_id,
                        relation="has",
                        established=date)
        
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
        self.G.add_edge(src, dst,
                        relation=relation,
                        established=date,
                        confidence=confidence,
                        **kwargs)
    
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
        
        # Store history
        if 'history' not in self.G.nodes[node_id]:
            self.G.nodes[node_id]['history'] = []
        
        # Record old values before update
        old_values = {}
        for key in ['value', 'confidence']:
            if key in self.G.nodes[node_id]:
                old_values[key] = self.G.nodes[node_id][key]
        
        # Update node
        self.G.nodes[node_id]['last_confirmed'] = date
        if new_value is not None:
            self.G.nodes[node_id]['value'] = new_value
        for key, val in kwargs.items():
            self.G.nodes[node_id][key] = val
        
        # Record in history
        self.G.nodes[node_id]['history'].append({
            'date': date,
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
        cur_date = datetime.strptime(current_date, '%Y-%m-%d')
        
        for node in self.G.nodes:
            node_data = self.G.nodes[node]
            
            # Don't decay patient node
            if node_data.get('type') == 'PATIENT':
                continue
            
            last_confirmed = datetime.strptime(node_data.get('last_confirmed', current_date), '%Y-%m-%d')
            months_old = (cur_date - last_confirmed).days / 30.0
            
            # Calculate new confidence
            old_conf = node_data.get('confidence', 1.0)
            new_conf = max(0.2, old_conf - months_old * decay_rate)
            
            node_data['confidence'] = new_conf
            node_data['decayed'] = True
    
    def get_entity(self, name: str, entity_type: str = None) -> Optional[str]:
        """
        Find node ID for an entity by name.
        
        Args:
            name: Entity name to search for
            entity_type: Optional type filter
        
        Returns:
            node_id if found, None otherwise
        """
        clean_name = name.lower().strip()
        
        for node, data in self.G.nodes(data=True):
            if data.get('name', '').lower() == clean_name:
                if entity_type is None or data.get('type') == entity_type:
                    return node
        return None
    
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
    
    # Track seen entities to avoid duplicates
    seen_entities = {}
    
    for visit in patient_data['visits']:
        date = visit.get('date', '')
        extracted = visit.get('extracted', {})
        
        # Process conditions
        for condition in extracted.get('conditions', []):
            node_id = pkg.get_entity(condition, 'CONDITION')
            if node_id is None:
                node_id = pkg.add_entity(condition, 'CONDITION', date)
                seen_entities[condition.lower()] = node_id
            else:
                pkg.update_entity(node_id, date, confirmed=True)
        
        # Process medications
        for med in extracted.get('medications', []):
            node_id = pkg.get_entity(med, 'MEDICATION')
            if node_id is None:
                node_id = pkg.add_entity(med, 'MEDICATION', date)
                seen_entities[med.lower()] = node_id
            else:
                pkg.update_entity(node_id, date, confirmed=True)
        
        # Process lab values
        for lab_name, lab_value in extracted.get('lab_values', {}).items():
            node_id = pkg.get_entity(lab_name, 'LAB_VALUE')
            if node_id is None:
                node_id = pkg.add_entity(lab_name, 'LAB_VALUE', date, value=lab_value)
                seen_entities[lab_name.lower()] = node_id
            else:
                pkg.update_entity(node_id, date, new_value=lab_value)
        
        # Process symptoms
        for symptom in extracted.get('symptoms', []):
            node_id = pkg.get_entity(symptom, 'SYMPTOM')
            if node_id is None:
                node_id = pkg.add_entity(symptom, 'SYMPTOM', date)
                seen_entities[symptom.lower()] = node_id
            else:
                pkg.update_entity(node_id, date, confirmed=True)
        
        # Process relationships
        for rel in extracted.get('relationships', []):
            subject = rel.get('subject', '')
            predicate = rel.get('predicate', '')
            obj = rel.get('object', '')
            
            if subject and obj:
                src_id = pkg.get_entity(subject)
                dst_id = pkg.get_entity(obj)
                
                if src_id and dst_id:
                    pkg.add_relation(src_id, dst_id, predicate, date)
    
    return pkg


def build_graphs_for_all_patients(input_dir: str = "data/patients_processed",
                                   output_dir: str = "data/graphs",
                                   limit: int = None) -> List[str]:
    """
    Build knowledge graphs for all processed patients.
    
    Args:
        input_dir: Directory with processed patient JSON files
        output_dir: Directory to save graph JSON files
        limit: Maximum number of patients to process
    
    Returns:
        List of patient IDs that were processed
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all processed patient files
    patient_files = list(Path(input_dir).glob("*.json"))
    
    if limit:
        patient_files = patient_files[:limit]
    
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
                    "medications": ["Metformin 500mg"],
                    "lab_values": {"HbA1c": 7.2, "BP": "140/90"},
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