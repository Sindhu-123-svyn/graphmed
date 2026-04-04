"""
Baseline RAG System for Comparison
Simple RAG without knowledge graph, memory evolution, or conflict detection
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


class BaselineRAG:
    """
    Simple RAG baseline without graph, memory evolution, or conflict detection.
    Used for comparison experiments.
    """
    
    def __init__(self, persist_dir: str = "data"):
        self.persist_dir = Path(persist_dir)
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB for simple retrieval
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir / "baseline_chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize LLM
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Store for patient data
        self.patient_data = {}
        self._load_patient_data()
        
        print("✅ Baseline RAG initialized (no graph, no memory evolution)")
    
    def _load_patient_data(self):
        """Load all patient data."""
        patients_dir = self.persist_dir / "patients_processed"
        if patients_dir.exists():
            for file in patients_dir.glob("*.json"):
                with open(file, 'r') as f:
                    data = json.load(f)
                    self.patient_data[data['patient_id']] = data
                    
                    # Index all notes for retrieval
                    self._index_patient(data)
    
    def _index_patient(self, patient_data: Dict):
        """Index patient notes for retrieval."""
        collection_name = f"baseline_{patient_data['patient_id']}"
        
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
        except:
            collection = self.client.get_collection(name=collection_name)
        
        for i, visit in enumerate(patient_data.get('visits', [])):
            doc_id = f"{patient_data['patient_id']}_visit_{i}"
            text = f"Date: {visit.get('date', '')}\nNote: {visit.get('note', '')}"
            embedding = self.embedder.encode(text).tolist()
            
            try:
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[{'date': visit.get('date', ''), 'visit_id': i}]
                )
            except:
                pass
    
    def retrieve(self, patient_id: str, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant passages."""
        collection_name = f"baseline_{patient_id}"
        
        try:
            collection = self.client.get_collection(name=collection_name)
            query_embedding = self.embedder.encode(query).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
            
            if results['documents'] and results['documents'][0]:
                return results['documents'][0]
        except:
            pass
        
        return []
    
    def answer(self, patient_id: str, query: str) -> str:
        """Generate answer using retrieved context."""
        # Retrieve relevant passages
        retrieved_docs = self.retrieve(patient_id, query)
        
        if not retrieved_docs:
            context = "No relevant patient information found."
        else:
            context = "\n\n".join(retrieved_docs)
        
        # Generate answer
        prompt = f"""You are a clinical assistant. Answer the question based ONLY on the provided patient context.

Patient Context:
{context}

Question: {query}

Answer (be concise and cite the source if possible):"""
        
        try:
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"
    
    def get_patient_info(self, patient_id: str, info_type: str) -> str:
        """Simple patient info retrieval without graph."""
        if patient_id not in self.patient_data:
            return f"No data for {patient_id}"
        
        patient = self.patient_data[patient_id]
        visits = patient.get('visits', [])
        
        if info_type == "conditions":
            conditions = set()
            for v in visits:
                for d in v.get('diagnoses', []):
                    conditions.add(d)
            return f"Conditions: {', '.join(conditions)}" if conditions else "No conditions found"
        
        elif info_type == "medications":
            medications = set()
            for v in visits:
                for m in v.get('medications', []):
                    medications.add(m)
            return f"Medications: {', '.join(medications)}" if medications else "No medications found"
        
        elif info_type == "symptoms":
            symptoms = set()
            for v in visits:
                for s in v.get('symptoms', []):
                    symptoms.add(s)
            return f"Symptoms: {', '.join(symptoms)}" if symptoms else "No symptoms documented"
        
        else:
            return "Info type not recognized"