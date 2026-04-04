"""
Phase 4: Vector Memory Store for GraphMed
Stores patient visit summaries as embeddings for semantic search
"""

import os
import warnings
import logging
import re
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional


def _disable_chroma_telemetry_noise() -> None:
    """Disable telemetry calls that can fail with incompatible PostHog versions."""
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_TELEMETRY"] = "False"

    # Chroma 0.4.x may call posthog.capture(distinct_id, event, properties).
    # Newer PostHog clients changed signatures, causing noisy non-fatal errors.
    try:
        import posthog  # type: ignore

        def _capture_noop(*args, **kwargs):
            return None

        posthog.capture = _capture_noop
    except Exception:
        pass


_disable_chroma_telemetry_noise()

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Suppress ChromaDB telemetry warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global settings to ensure consistency
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)

EMOTION_TERMS = {
    "anxiety", "anxious", "worry", "worried", "fear", "afraid",
    "frustrated", "depressed", "overwhelmed", "distressed", "stress"
}

UNCERTAINTY_TERMS = {
    "possible", "possibly", "probable", "probably", "unclear",
    "suggests", "suggesting", "cannot rule out", "concern for",
    "suspect", "suspected", "question of"
}

CLINICIAN_INTENT_TERMS = {
    "plan", "monitor", "follow-up", "follow up", "refer", "referral",
    "counseled", "advised", "discussed", "consider", "observe"
}


class PatientMemoryStore:
    """
    Vector memory store for a single patient.
    Stores visit summaries as embeddings for semantic retrieval.
    """
    
    def __init__(self, patient_id: str, persist_directory: str = "data/chroma_db"):
        """
        Initialize memory store for a patient.
        
        Args:
            patient_id: Unique patient identifier
            persist_directory: Directory to store ChromaDB data
        """
        self.patient_id = patient_id
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent settings
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=CHROMA_SETTINGS
            )
        except Exception as e:
            # If there's an existing client with different settings, we need to handle it
            # For now, try to get the existing client
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
        
        # Create or get collection for this patient
        collection_name = f"patient_{patient_id}"
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": f"Visit memories for patient {patient_id}"}
            )
        except Exception as e:
            # If collection exists but metadata mismatch, just get it
            self.collection = self.client.get_collection(name=collection_name)
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Don't print to avoid clutter, but we can show count
        # print(f"✅ Memory store initialized for {patient_id}")
        # print(f"   Collection size: {self.collection.count()} existing memories")
    
    def create_visit_summary(self, visit: Dict[str, Any]) -> str:
        """
        Create a rich summary from visit data for embedding.
        
        Args:
            visit: Visit dictionary with note, diagnoses, medications, symptoms, labs
        
        Returns:
            Text summary for embedding
        """
        extracted = visit.get("extracted", {}) or {}
        note = (visit.get("note") or "").strip()

        diagnoses = self._merge_unique(
            visit.get("diagnoses", []),
            extracted.get("conditions", [])
        )
        medications = self._merge_unique(
            visit.get("medications", []),
            extracted.get("medications", [])
        )
        symptoms = self._merge_unique(
            visit.get("symptoms", []),
            extracted.get("symptoms", [])
        )

        dosages = self._merge_unique(
            visit.get("dosages", []),
            extracted.get("dosages", [])
        )
        procedures = self._to_clean_list(extracted.get("procedures", []))
        relationships = extracted.get("relationships", [])
        labs = visit.get("labs", {}) or extracted.get("lab_values", {})

        signal_info = self._extract_narrative_signals(note)

        summary_parts = [
            f"Patient ID: {self.patient_id}",
            f"Visit ID: {visit.get('visit_id', 'Unknown')}",
            f"Visit Date: {visit.get('date', 'Unknown')}",
        ]

        if note:
            summary_parts.append("Narrative Note:")
            summary_parts.append(note)

        summary_parts.append("Clinical Snapshot:")

        if diagnoses:
            summary_parts.append(f"- Diagnoses/Conditions: {', '.join(diagnoses)}")
        if symptoms:
            summary_parts.append(f"- Symptoms: {', '.join(symptoms)}")
        if medications:
            summary_parts.append(f"- Medications: {', '.join(medications)}")
        if dosages:
            summary_parts.append(f"- Dosages: {', '.join(dosages)}")
        if procedures:
            summary_parts.append(f"- Procedures/Referrals: {', '.join(procedures)}")

        if isinstance(labs, dict) and labs:
            lab_items = [f"{k}: {v}" for k, v in labs.items()]
            summary_parts.append(f"- Labs: {', '.join(lab_items)}")

        if relationships:
            relationship_lines = []
            for rel in relationships[:8]:
                if not isinstance(rel, dict):
                    continue
                subj = rel.get("subject", "?")
                pred = rel.get("predicate", "related_to")
                obj = rel.get("object", "?")
                relationship_lines.append(f"{subj} {pred} {obj}")
            if relationship_lines:
                summary_parts.append(f"- Extracted Relationships: {'; '.join(relationship_lines)}")

        summary_parts.append("Context Signals:")
        summary_parts.append(
            f"- Emotional cues: {', '.join(signal_info['emotions']) if signal_info['emotions'] else 'none explicitly stated'}"
        )
        summary_parts.append(
            f"- Ambiguity/uncertainty cues: {', '.join(signal_info['uncertainty']) if signal_info['uncertainty'] else 'none explicitly stated'}"
        )
        summary_parts.append(
            f"- Clinician intent/action cues: {', '.join(signal_info['clinician_intent']) if signal_info['clinician_intent'] else 'none explicitly stated'}"
        )

        return "\n".join(summary_parts)

    def _to_clean_list(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, (str, int, float, bool)):
            value = [value]
        elif not isinstance(value, list):
            return []

        cleaned: List[str] = []
        seen = set()
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    def _merge_unique(self, *values: Any) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in values:
            for item in self._to_clean_list(value):
                key = item.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _extract_narrative_signals(self, note: str) -> Dict[str, List[str]]:
        note_lower = note.lower()

        def find_terms(term_set: set) -> List[str]:
            found = []
            for term in sorted(term_set):
                if re.search(r"\\b" + re.escape(term) + r"\\b", note_lower):
                    found.append(term)
            return found

        return {
            "emotions": find_terms(EMOTION_TERMS),
            "uncertainty": find_terms(UNCERTAINTY_TERMS),
            "clinician_intent": find_terms(CLINICIAN_INTENT_TERMS),
        }
    
    def store_visit(self, visit: Dict[str, Any], visit_id: Optional[str] = None) -> str:
        """
        Store a visit in the vector memory.
        
        Args:
            visit: Visit dictionary with clinical data
            visit_id: Optional custom ID (uses visit_id from data if not provided)
        
        Returns:
            ID of the stored document
        """
        # Generate summary
        summary = self.create_visit_summary(visit)
        
        # Generate embedding
        embedding = self.embedder.encode(summary).tolist()
        
        # Use provided ID or from visit data or generate
        doc_id = visit_id or visit.get('visit_id', str(uuid.uuid4()))
        
        signal_info = self._extract_narrative_signals(visit.get("note", "") or "")

        # Prepare metadata
        metadata = {
            "visit_id": doc_id,
            "date": visit.get('date', ''),
            "patient_id": self.patient_id,
            "has_emotion_signal": len(signal_info["emotions"]) > 0,
            "has_uncertainty_signal": len(signal_info["uncertainty"]) > 0,
            "has_clinician_intent": len(signal_info["clinician_intent"]) > 0,
            "emotion_terms": ", ".join(signal_info["emotions"]),
            "uncertainty_terms": ", ".join(signal_info["uncertainty"]),
            "intent_terms": ", ".join(signal_info["clinician_intent"]),
        }

        # Upsert to make repeated builds idempotent.
        try:
            self.collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[summary],
                metadatas=[metadata]
            )
        except Exception:
            pass
        
        return doc_id
    
    def store_all_visits(self, patient_data: Dict[str, Any], verbose: bool = False) -> int:
        """
        Store all visits for a patient.
        
        Args:
            patient_data: Complete patient data with visits
            verbose: Whether to print progress
        
        Returns:
            Number of visits stored
        """
        visits = patient_data.get('visits', [])
        
        if not visits:
            return 0
        
        stored = 0
        for visit in visits:
            try:
                self.store_visit(visit)
                stored += 1
            except Exception:
                # Silently skip errors during build
                pass
        
        return stored
    
    def retrieve_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most similar past visits based on semantic similarity.
        
        Args:
            query: Search query (e.g., "patient with fatigue and high blood pressure")
            top_k: Number of results to return
        
        Returns:
            List of similar visits with their documents and metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Query the collection
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
        except Exception as e:
            return []
        
        # Format results
        retrieved = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                retrieved.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i] if results['documents'] else '',
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results.get('distances') else None
                })
        
        return retrieved
    
    def search_by_symptom(self, symptom: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for visits mentioning a specific symptom."""
        query = f"Patient experiences {symptom}"
        return self.retrieve_similar(query, top_k)
    
    def search_by_medication(self, medication: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for visits involving a specific medication."""
        query = f"Patient prescribed {medication}"
        return self.retrieve_similar(query, top_k)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            return {
                'patient_id': self.patient_id,
                'collection_name': f"patient_{self.patient_id}",
                'count': self.collection.count(),
                'persist_directory': self.persist_directory
            }
        except Exception:
            return {
                'patient_id': self.patient_id,
                'collection_name': f"patient_{self.patient_id}",
                'count': 0,
                'persist_directory': self.persist_directory
            }
    
    def delete_collection(self):
        """Delete the entire collection (use with caution)."""
        try:
            self.client.delete_collection(f"patient_{self.patient_id}")
        except Exception:
            pass


class GlobalMemoryManager:
    """
    Manages memory stores for all patients.
    """
    
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize global memory manager.
        
        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = persist_directory
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=CHROMA_SETTINGS
            )
        except Exception:
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._stores = {}
    
    def get_patient_store(self, patient_id: str) -> PatientMemoryStore:
        """Get or create memory store for a patient."""
        if patient_id not in self._stores:
            self._stores[patient_id] = PatientMemoryStore(
                patient_id,
                self.persist_directory
            )
        return self._stores[patient_id]
    
    def list_patients(self) -> List[str]:
        """List all patients with memory stores."""
        try:
            collections = self.client.list_collections()
            patients = []
            for col in collections:
                if col.name.startswith("patient_"):
                    patient_id = col.name.replace("patient_", "")
                    patients.append(patient_id)
            return patients
        except Exception:
            return []
    
    def global_search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Search across all patients."""
        query_embedding = self.embedder.encode(query).tolist()
        results = {}
        
        for patient_id in self.list_patients():
            try:
                store = self.get_patient_store(patient_id)
                patient_results = store.retrieve_similar(query, top_k)
                if patient_results:
                    results[patient_id] = patient_results
            except Exception:
                continue
        
        return results


def build_memory_for_all_patients(input_dir: str = "data/patients_processed",
                                   persist_dir: str = "data/chroma_db",
                                   limit: int = None,
                                   verbose: bool = False) -> Dict[str, int]:
    """
    Build memory stores for all processed patients.
    
    Args:
        input_dir: Directory with processed patient JSON files
        persist_dir: Directory to store ChromaDB data
        limit: Maximum number of patients to process
        verbose: Whether to print progress
    
    Returns:
        Dictionary mapping patient_id to number of visits stored
    """
    # Create persist directory
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all patient files
    patient_files = sorted(Path(input_dir).glob("*.json"))
    
    if limit:
        patient_files = patient_files[:limit]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🏥 PHASE 4: BUILDING VECTOR MEMORY STORES")
        print(f"{'='*60}")
        print(f"Patients: {len(patient_files)}")
        print(f"{'='*60}\n")
    
    results = {}
    total_visits = 0
    
    for i, file_path in enumerate(patient_files, 1):
        try:
            # Load patient data
            with open(file_path, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
            
            patient_id = patient_data['patient_id']
            
            if verbose:
                print(f"[{i}/{len(patient_files)}] Processing {patient_id}...", end=" ")
            
            # Create memory store
            memory_store = PatientMemoryStore(patient_id, persist_dir)
            
            # Store all visits
            num_stored = memory_store.store_all_visits(patient_data)
            
            results[patient_id] = num_stored
            total_visits += num_stored
            
            if verbose:
                print(f"✅ ({num_stored} visits)")
            
        except Exception as e:
            if verbose:
                print(f"❌ Error: {e}")
            results[file_path.stem] = 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📈 MEMORY BUILD COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Patients processed: {len(results)}")
        print(f"✅ Total visits stored: {total_visits}")
        print(f"📁 Location: {persist_dir}")
        print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    # Quick test
    print("Testing PatientMemoryStore...")
    
    # Create memory store
    store = PatientMemoryStore("TEST001", "data/test_chroma_db")
    
    print("\n✅ Memory module ready!")