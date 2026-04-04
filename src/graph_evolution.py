"""
Phase 8: Graph Evolution & Conflict Resolution for GraphMed
Handles temporal evolution, conflict detection, and confidence decay
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import PatientKnowledgeGraph
from src.conflict_detector_simple import get_conflict_detector, detect_conflict


class GraphEvolution:
    """
    Handles evolution of patient knowledge graphs over time.
    Supports addition, update, conflict detection, and confidence decay.
    """
    
    def __init__(self, graph: PatientKnowledgeGraph):
        """
        Initialize graph evolution manager.
        
        Args:
            graph: PatientKnowledgeGraph instance
        """
        self.graph = graph
        self.conflict_detector = get_conflict_detector()
        self.evolution_log = []
    
    def add_new_entity(self, entity_name: str, entity_type: str, 
                       date: str, value: Any = None) -> Tuple[str, str]:
        """
        Add a new entity to the graph.
        
        Args:
            entity_name: Name of the entity
            entity_type: CONDITION, MEDICATION, SYMPTOM, LAB_VALUE, PROCEDURE
            date: Date of observation
            value: Optional value (for lab values)
        
        Returns:
            Tuple of (node_id, status) where status is "ADDED" or "UPDATED"
        """
        existing_node = self.graph.get_entity(entity_name, entity_type)
        
        if existing_node is None:
            # New entity - add it
            node_id = self.graph.add_entity(
                entity_name, entity_type, date, 
                confidence=1.0, value=value
            )
            self._log_evolution("ADDED", entity_name, entity_type, date)
            return node_id, "ADDED"
        else:
            # Entity exists - update it
            self.graph.update_entity(existing_node, date, new_value=value, confirmed=True)
            self._log_evolution("UPDATED", entity_name, entity_type, date)
            return existing_node, "UPDATED"
    
    def detect_and_resolve_conflicts(self, new_entity: Dict, 
                                      existing_entity: Dict) -> Dict:
        """
        Detect and resolve conflicts between new and existing information.
        
        Args:
            new_entity: New entity data
            existing_entity: Existing entity data
        
        Returns:
            Resolution result with status and recommendation
        """
        # Create statements for comparison
        new_stmt = f"{new_entity.get('name', '')} {new_entity.get('value', '')}"
        existing_stmt = f"{existing_entity.get('name', '')} {existing_entity.get('value', '')}"
        
        # Use conflict detector
        conflict_result = detect_conflict(new_stmt, existing_stmt)
        
        if conflict_result["is_conflict"] and conflict_result["confidence"] > 0.6:
            # Conflict detected - need resolution
            new_date = new_entity.get('date', '')
            existing_date = existing_entity.get('added', '')
            
            # Compare timestamps (newer information takes precedence)
            if new_date > existing_date:
                resolution = "NEWER_TAKES_PRECEDENCE"
                recommendation = f"Newer information ({new_date}) supersedes older ({existing_date})"
            else:
                resolution = "FLAG_FOR_REVIEW"
                recommendation = f"Conflict detected - manual review recommended"
            
            return {
                "is_conflict": True,
                "confidence": conflict_result["confidence"],
                "resolution": resolution,
                "recommendation": recommendation,
                "new_entity": new_entity,
                "existing_entity": existing_entity
            }
        
        return {
            "is_conflict": False,
            "confidence": conflict_result["confidence"],
            "resolution": "NO_CONFLICT",
            "recommendation": "Information is consistent"
        }
    
    def evolve_with_visit(self, visit_data: Dict) -> List[Dict]:
        """
        Evolve the graph with new visit data.
        
        Args:
            visit_data: New visit data with extracted entities
        
        Returns:
            List of evolution results for each entity
        """
        results = []
        date = visit_data.get('date', datetime.now().isoformat()[:10])
        extracted = visit_data.get('extracted', {})
        
        # Process conditions
        for condition in extracted.get('conditions', []):
            result = self._process_condition(condition, date)
            results.append(result)
        
        # Process medications
        for medication in extracted.get('medications', []):
            result = self._process_medication(medication, date)
            results.append(result)
        
        # Process symptoms
        for symptom in extracted.get('symptoms', []):
            result = self._process_symptom(symptom, date)
            results.append(result)
        
        # Process lab values
        for lab_name, lab_value in extracted.get('lab_values', {}).items():
            result = self._process_lab_value(lab_name, lab_value, date)
            results.append(result)
        
        # Apply confidence decay after processing
        self.apply_confidence_decay(date)
        
        return results
    
    def _process_condition(self, condition: str, date: str) -> Dict:
        """Process a condition entity."""
        existing = self.graph.get_entity(condition, "CONDITION")
        
        new_entity = {
            "name": condition,
            "type": "CONDITION",
            "date": date,
            "value": None
        }
        
        if existing:
            existing_data = self.graph.G.nodes[existing]
            conflict = self.detect_and_resolve_conflicts(new_entity, existing_data)
            
            if conflict["is_conflict"]:
                self.graph.G.nodes[existing]["status"] = "CONFLICTED"
                self.graph.G.nodes[existing]["conflict_note"] = conflict["recommendation"]
                return {
                    "entity": condition,
                    "type": "CONDITION",
                    "action": "CONFLICT_DETECTED",
                    "resolution": conflict["resolution"],
                    "recommendation": conflict["recommendation"]
                }
            else:
                self.graph.update_entity(existing, date, confirmed=True)
                return {
                    "entity": condition,
                    "type": "CONDITION",
                    "action": "UPDATED",
                    "resolution": "CONSISTENT"
                }
        else:
            self.graph.add_entity(condition, "CONDITION", date)
            return {
                "entity": condition,
                "type": "CONDITION",
                "action": "ADDED",
                "resolution": "NEW_ENTITY"
            }
    
    def _process_medication(self, medication: str, date: str) -> Dict:
        """Process a medication entity."""
        # Clean medication name (remove dosage if present)
        clean_name = medication.split()[0] if ' ' in medication else medication
        
        existing = self.graph.get_entity(clean_name, "MEDICATION")
        
        if existing:
            self.graph.update_entity(existing, date, confirmed=True)
            return {
                "entity": clean_name,
                "type": "MEDICATION",
                "action": "CONFIRMED",
                "resolution": "EXISTING"
            }
        else:
            self.graph.add_entity(medication, "MEDICATION", date)
            return {
                "entity": medication,
                "type": "MEDICATION",
                "action": "ADDED",
                "resolution": "NEW_ENTITY"
            }
    
    def _process_symptom(self, symptom: str, date: str) -> Dict:
        """Process a symptom entity."""
        existing = self.graph.get_entity(symptom, "SYMPTOM")
        
        if existing:
            self.graph.update_entity(existing, date, confirmed=True)
            return {
                "entity": symptom,
                "type": "SYMPTOM",
                "action": "CONFIRMED",
                "resolution": "EXISTING"
            }
        else:
            self.graph.add_entity(symptom, "SYMPTOM", date)
            return {
                "entity": symptom,
                "type": "SYMPTOM",
                "action": "ADDED",
                "resolution": "NEW_ENTITY"
            }
    
    def _process_lab_value(self, lab_name: str, lab_value: Any, date: str) -> Dict:
        """Process a lab value entity."""
        node_id = f"{lab_name}_{date}"
        
        existing = self.graph.get_entity(lab_name, "LAB_VALUE")
        
        if existing:
            # Check if value changed significantly
            existing_value = self.graph.G.nodes[existing].get('value')
            
            if existing_value != lab_value:
                # Track trend
                trend = self._calculate_trend(lab_name, lab_value)
                self.graph.update_entity(existing, date, new_value=lab_value, trend=trend)
                
                return {
                    "entity": lab_name,
                    "type": "LAB_VALUE",
                    "action": "UPDATED",
                    "old_value": existing_value,
                    "new_value": lab_value,
                    "trend": trend
                }
            else:
                self.graph.update_entity(existing, date, confirmed=True)
                return {
                    "entity": lab_name,
                    "type": "LAB_VALUE",
                    "action": "CONFIRMED",
                    "value": lab_value
                }
        else:
            self.graph.add_entity(lab_name, "LAB_VALUE", date, value=lab_value)
            return {
                "entity": lab_name,
                "type": "LAB_VALUE",
                "action": "ADDED",
                "value": lab_value
            }
    
    def _calculate_trend(self, lab_name: str, new_value: float) -> str:
        """Calculate trend for lab values."""
        # Get historical values
        trend_data = self.graph.get_lab_trend(lab_name)
        
        if len(trend_data) < 2:
            return "INSUFFICIENT_DATA"
        
        old_value = trend_data[-1].get('value', new_value)
        
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            if new_value > old_value * 1.1:
                return "INCREASING"
            elif new_value < old_value * 0.9:
                return "DECREASING"
            else:
                return "STABLE"
        
        return "UNKNOWN"
    
    def apply_confidence_decay(self, current_date: str, decay_rate: float = 0.05):
        """
        Apply confidence decay to entities not recently confirmed.
        
        Args:
            current_date: Current date for decay calculation
            decay_rate: Decay rate per month (default 5%)
        """
        self.graph.decay_confidence(current_date, decay_rate)
        self._log_evolution("DECAY_APPLIED", "ALL", "SYSTEM", current_date)
    
    def get_evolution_summary(self) -> Dict:
        """Get summary of graph evolution."""
        conflicts = self.graph.get_conflicts()
        
        # Get entities by confidence
        low_confidence = []
        for node, data in self.graph.G.nodes(data=True):
            if data.get('type') != 'PATIENT':
                conf = data.get('confidence', 1.0)
                if conf < 0.5:
                    low_confidence.append({
                        'name': data.get('name'),
                        'type': data.get('type'),
                        'confidence': conf
                    })
        
        return {
            'total_nodes': self.graph.G.number_of_nodes(),
            'total_edges': self.graph.G.number_of_edges(),
            'conflicts': len(conflicts),
            'low_confidence_entities': len(low_confidence),
            'evolution_steps': len(self.evolution_log)
        }
    
    def _log_evolution(self, action: str, entity: str, entity_type: str, date: str):
        """Log evolution event."""
        self.evolution_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'entity': entity,
            'entity_type': entity_type,
            'date': date
        })


def evolve_patient_graph(patient_id: str, new_visit: Dict, 
                         graphs_dir: str = "data/graphs") -> Dict:
    """
    Convenience function to evolve a patient's graph with new visit data.
    
    Args:
        patient_id: Patient identifier
        new_visit: New visit data with extracted entities
        graphs_dir: Directory containing graph files
    
    Returns:
        Evolution results
    """
    graphs_path = Path(graphs_dir)
    graph_path = graphs_path / f"{patient_id}_graph.json"
    
    # Load existing graph or create new
    if graph_path.exists():
        graph = PatientKnowledgeGraph.load(str(graph_path))
    else:
        graph = PatientKnowledgeGraph(patient_id)
    
    # Evolve graph
    evolution = GraphEvolution(graph)
    results = evolution.evolve_with_visit(new_visit)
    
    # Save updated graph
    graph.save(str(graph_path))
    
    return {
        "patient_id": patient_id,
        "evolution_results": results,
        "summary": evolution.get_evolution_summary()
    }


def simulate_patient_evolution(patient_id: str, visits: List[Dict]) -> List[Dict]:
    """
    Simulate graph evolution over multiple visits.
    
    Args:
        patient_id: Patient identifier
        visits: List of visits in chronological order
    
    Returns:
        Evolution history for each visit
    """
    graphs_path = Path("data/graphs")
    graphs_path.mkdir(exist_ok=True)
    
    graph = PatientKnowledgeGraph(patient_id)
    evolution = GraphEvolution(graph)
    
    history = []
    
    for i, visit in enumerate(visits, 1):
        print(f"\n📅 Processing Visit {i}: {visit.get('date', 'Unknown')}")
        
        results = evolution.evolve_with_visit(visit)
        
        history.append({
            "visit_number": i,
            "date": visit.get('date'),
            "results": results,
            "summary": evolution.get_evolution_summary()
        })
        
        # Show key changes
        for result in results:
            if result.get('action') in ['ADDED', 'UPDATED', 'CONFLICT_DETECTED']:
                print(f"   {result['action']}: {result.get('entity')} ({result.get('type')})")
                if result.get('trend'):
                    print(f"      Trend: {result['trend']}")
    
    # Save final graph
    graph.save(str(graphs_path / f"{patient_id}_graph.json"))
    
    return history


def test_evolution():
    """Test the graph evolution system."""
    
    print("\n" + "="*60)
    print("🧪 TESTING GRAPH EVOLUTION")
    print("="*60)
    
    # Create test patient
    patient_id = "TEST_EVOLUTION"
    
    # Simulate visits over time
    test_visits = [
        {
            "date": "2024-01-15",
            "extracted": {
                "conditions": ["Type 2 Diabetes", "Hypertension"],
                "medications": ["Metformin 500mg"],
                "symptoms": ["Fatigue", "Polydipsia"],
                "lab_values": {"HbA1c": 7.2, "BP": "140/90"}
            }
        },
        {
            "date": "2024-04-20",
            "extracted": {
                "conditions": ["Type 2 Diabetes", "Hypertension"],
                "medications": ["Metformin 1000mg"],
                "symptoms": ["Fatigue"],
                "lab_values": {"HbA1c": 7.8, "BP": "138/88"}
            }
        },
        {
            "date": "2024-08-10",
            "extracted": {
                "conditions": ["Type 2 Diabetes", "Hypertension", "Diabetic Neuropathy"],
                "medications": ["Metformin 1000mg", "Gabapentin 300mg"],
                "symptoms": ["Tingling in feet"],
                "lab_values": {"HbA1c": 8.1, "BP": "142/92"}
            }
        },
        {
            "date": "2024-12-05",
            "extracted": {
                "conditions": ["Type 2 Diabetes", "Hypertension", "Diabetic Neuropathy"],
                "medications": ["Metformin 1000mg", "Gabapentin 300mg", "Lisinopril 10mg"],
                "symptoms": [],
                "lab_values": {"HbA1c": 7.9, "BP": "135/85"}
            }
        }
    ]
    
    # Run evolution simulation
    history = simulate_patient_evolution(patient_id, test_visits)
    
    # Load final graph
    final_graph = PatientKnowledgeGraph.load(f"data/graphs/{patient_id}_graph.json")
    
    print("\n" + "="*60)
    print("📊 FINAL GRAPH SUMMARY")
    print("="*60)
    
    summary = final_graph.summary()
    print(f"Total Nodes: {summary['total_nodes']}")
    print(f"Total Edges: {summary['total_edges']}")
    print(f"Conflicts: {summary['conflicts']}")
    
    print("\nEntity Types:")
    for node_type, count in summary['node_types'].items():
        print(f"  {node_type}: {count}")
    
    # Show lab trends
    print("\n📈 Lab Trends:")
    for lab in ["HbA1c", "BP"]:
        trend = final_graph.get_lab_trend(lab)
        if trend:
            values = [f"{t['value']}" for t in trend]
            print(f"  {lab}: {' → '.join(values)}")
    
    return history


if __name__ == "__main__":
    test_evolution()