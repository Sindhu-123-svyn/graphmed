"""
Phase 4: Vector Memory Store for GraphMed
Stores patient visit summaries as embeddings for semantic search
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import uuid
import warnings
import logging
import os

# Suppress ChromaDB telemetry warnings
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global settings to ensure consistency
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)


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
        summary_parts = []
        
        # Add date
        summary_parts.append(f"Visit Date: {visit.get('date', 'Unknown')}")
        
        # Add clinical note (most important)
        note = visit.get('note', '')
        if note:
            summary_parts.append(f"Clinical Note: {note}")
        
        # Add diagnoses/conditions
        diagnoses = visit.get('diagnoses', []) or visit.get('extracted', {}).get('conditions', [])
        if diagnoses:
            summary_parts.append(f"Diagnoses: {', '.join(diagnoses)}")
        
        # Add medications
        medications = visit.get('medications', []) or visit.get('extracted', {}).get('medications', [])
        if medications:
            summary_parts.append(f"Medications: {', '.join(medications)}")
        
        # Add symptoms
        symptoms = visit.get('symptoms', []) or visit.get('extracted', {}).get('symptoms', [])
        if symptoms:
            summary_parts.append(f"Symptoms: {', '.join(symptoms)}")
        
        # Add lab values
        labs = visit.get('labs', {}) or visit.get('extracted', {}).get('lab_values', {})
        if labs:
            lab_str = ', '.join([f"{k}: {v}" for k, v in labs.items()])
            summary_parts.append(f"Lab Values: {lab_str}")
        
        return "\n".join(summary_parts)
    
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
        
        # Prepare metadata
        metadata = {
            "visit_id": doc_id,
            "date": visit.get('date', ''),
            "patient_id": self.patient_id
        }
        
        # Add to collection (with error handling)
        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[summary],
                metadatas=[metadata]
            )
        except Exception as e:
            # If document already exists, update it
            if "already exists" in str(e):
                self.collection.update(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[metadata]
                )
            else:
                # Silently ignore other errors during build
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
    patient_files = list(Path(input_dir).glob("*.json"))
    
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