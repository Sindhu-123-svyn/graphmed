"""
Baseline RAG System for Comparison
Simple RAG without knowledge graph, memory evolution, or conflict detection
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import os
import requests
import warnings
import logging


def _disable_chroma_telemetry_noise() -> None:
    """Disable telemetry calls that can fail with incompatible PostHog versions."""
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_TELEMETRY"] = "False"

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
from groq import Groq
from dotenv import load_dotenv

# Suppress telemetry warning logs.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.ERROR)

load_dotenv()


class BaselineRAG:
    """
    Simple RAG baseline without graph, memory evolution, or conflict detection.
    Used for comparison experiments.
    """
    
    def __init__(self, persist_dir: str = "data", llm_provider: str = None):
        self.persist_dir = Path(persist_dir)
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB for simple retrieval
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir / "baseline_chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize LLM providers
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        self.openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.openrouter_model = os.getenv("BASELINE_OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
        self.baseline_provider = (llm_provider or os.getenv("BASELINE_LLM_PROVIDER") or "groq").strip().lower()
        
        # Store for patient data
        self.patient_data = {}
        self._load_patient_data()
        
        print("[OK] Baseline RAG initialized (no graph, no memory evolution)")
        print(f"[OK] Baseline LLM provider preference: {self.baseline_provider}")
    
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
        
        if self.baseline_provider == "openrouter":
            out = self._answer_with_openrouter(prompt)
            if not out.lower().startswith("error:"):
                return out
            fallback = os.getenv("BASELINE_OPENROUTER_FALLBACK_TO_GROQ", "1").strip() == "1"
            if fallback:
                return self._answer_with_groq(prompt)
            return out

        return self._answer_with_groq(prompt)

    def _answer_with_groq(self, prompt: str) -> str:
        if self.llm is None:
            return "Error: GROQ_API_KEY is not configured"
        try:
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def _answer_with_openrouter(self, prompt: str) -> str:
        if not self.openrouter_api_key:
            return "Error: OPEN_ROUTER_API_KEY is not configured"

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        try:
            resp = requests.post(self.openrouter_api_url, headers=headers, json=payload, timeout=45)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
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