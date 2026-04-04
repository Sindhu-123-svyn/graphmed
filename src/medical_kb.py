"""
Phase 5: Medical RAG Knowledge Base for GraphMed
External medical knowledge base for drug interactions, guidelines, and disease information
"""

import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Medical knowledge categories
MEDICAL_CATEGORIES = {
    "drug_interaction": "Drug-drug interactions and contraindications",
    "drug_disease": "Drug-disease contraindications",
    "disease_info": "Disease symptoms, diagnosis, treatment guidelines",
    "clinical_guideline": "Clinical practice guidelines and recommendations",
    "lab_interpretation": "Lab value interpretation and reference ranges",
    "medication_info": "Medication indications, dosages, side effects"
}

class MedicalKnowledgeBase:
    """
    External medical knowledge base for grounding clinical answers.
    Contains drug interactions, disease information, and clinical guidelines.
    """
    
    def __init__(self, persist_directory: str = "data/medical_kb"):
        """
        Initialize the medical knowledge base.
        
        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection for medical knowledge
        try:
            self.collection = self.client.get_or_create_collection(
                name="medical_knowledge_base",
                metadata={"description": "External medical knowledge for grounding"}
            )
        except Exception:
            self.collection = self.client.get_collection(name="medical_knowledge_base")
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✅ Medical Knowledge Base initialized")
        print(f"   Existing entries: {self.collection.count()}")
    
    def add_drug_interaction(self, drug1: str, drug2: str, 
                             severity: str, description: str, 
                             recommendation: str) -> str:
        """
        Add a drug-drug interaction to the knowledge base.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            severity: severe, moderate, mild
            description: Description of the interaction
            recommendation: Clinical recommendation
        
        Returns:
            Document ID
        """
        doc_id = f"interaction_{drug1}_{drug2}".lower().replace(" ", "_")
        
        document = f"""
DRUG INTERACTION: {drug1} and {drug2}
Severity: {severity.upper()}
Description: {description}
Recommendation: {recommendation}
        """.strip()
        
        metadata = {
            "type": "drug_interaction",
            "drug1": drug1.lower(),
            "drug2": drug2.lower(),
            "severity": severity,
            "category": "drug_interaction"
        }
        
        embedding = self.embedder.encode(document).tolist()
        
        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        
        return doc_id
    
    def add_disease_info(self, disease: str, symptoms: List[str],
                         diagnosis: str, treatment: str,
                         complications: List[str] = None) -> str:
        """
        Add disease information to the knowledge base.
        
        Args:
            disease: Disease name
            symptoms: List of common symptoms
            diagnosis: Diagnostic criteria/methods
            treatment: Treatment approaches
            complications: Potential complications
        
        Returns:
            Document ID
        """
        doc_id = f"disease_{disease}".lower().replace(" ", "_")
        
        symptoms_str = ", ".join(symptoms) if symptoms else "Not specified"
        complications_str = ", ".join(complications) if complications else "Not specified"
        
        document = f"""
DISEASE: {disease}
Common Symptoms: {symptoms_str}
Diagnosis: {diagnosis}
Treatment: {treatment}
Potential Complications: {complications_str}
        """.strip()
        
        metadata = {
            "type": "disease_info",
            "disease": disease.lower(),
            "category": "disease_info"
        }
        
        embedding = self.embedder.encode(document).tolist()
        
        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        
        return doc_id
    
    def add_medication_info(self, medication: str, indications: List[str],
                            dosages: List[str], side_effects: List[str],
                            contraindications: List[str]) -> str:
        """
        Add medication information to the knowledge base.
        
        Args:
            medication: Medication name
            indications: What it treats
            dosages: Common dosages
            side_effects: Common side effects
            contraindications: When not to use
        
        Returns:
            Document ID
        """
        doc_id = f"medication_{medication}".lower().replace(" ", "_")
        
        document = f"""
MEDICATION: {medication}
Indications: {', '.join(indications)}
Common Dosages: {', '.join(dosages)}
Side Effects: {', '.join(side_effects)}
Contraindications: {', '.join(contraindications)}
        """.strip()
        
        metadata = {
            "type": "medication_info",
            "medication": medication.lower(),
            "category": "medication_info"
        }
        
        embedding = self.embedder.encode(document).tolist()
        
        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        
        return doc_id
    
    def add_clinical_guideline(self, condition: str, guideline: str,
                                source: str, year: int) -> str:
        """
        Add clinical guideline to the knowledge base.
        
        Args:
            condition: Medical condition
            guideline: Guideline content
            source: Source organization
            year: Year of guideline
        
        Returns:
            Document ID
        """
        doc_id = f"guideline_{condition}".lower().replace(" ", "_")
        
        document = f"""
CLINICAL GUIDELINE: {condition}
Source: {source} ({year})
Guideline: {guideline}
        """.strip()
        
        metadata = {
            "type": "clinical_guideline",
            "condition": condition.lower(),
            "source": source,
            "year": year,
            "category": "clinical_guideline"
        }
        
        embedding = self.embedder.encode(document).tolist()
        
        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        
        return doc_id
    
    def add_lab_reference(self, lab_name: str, normal_range: str,
                          critical_range: str, interpretation: str) -> str:
        """
        Add lab value reference information.
        
        Args:
            lab_name: Name of the lab test
            normal_range: Normal reference range
            critical_range: Critical/abnormal range
            interpretation: How to interpret values
        
        Returns:
            Document ID
        """
        doc_id = f"lab_{lab_name}".lower().replace(" ", "_")
        
        document = f"""
LAB TEST: {lab_name}
Normal Range: {normal_range}
Critical/Abnormal Range: {critical_range}
Interpretation: {interpretation}
        """.strip()
        
        metadata = {
            "type": "lab_interpretation",
            "lab_name": lab_name.lower(),
            "category": "lab_interpretation"
        }
        
        embedding = self.embedder.encode(document).tolist()
        
        try:
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        except Exception:
            self.collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
        
        return doc_id
    
    def query(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the medical knowledge base for relevant information.
        
        Args:
            question: Clinical question or query
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with metadata
        """
        # Generate embedding for the question
        query_embedding = self.embedder.encode(question).tolist()
        
        # Query the collection
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
        except Exception as e:
            print(f"Query error: {e}")
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
    
    def check_drug_interaction(self, drug1: str, drug2: str) -> Optional[Dict]:
        """
        Check if two drugs have a known interaction.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
        
        Returns:
            Interaction information if found, None otherwise
        """
        # Try exact match
        doc_id = f"interaction_{drug1}_{drug2}".lower().replace(" ", "_")
        
        # Search for interactions involving either drug
        query = f"Drug interaction between {drug1} and {drug2}"
        results = self.query(query, top_k=3)
        
        for result in results:
            metadata = result['metadata']
            if metadata.get('type') == 'drug_interaction':
                d1 = metadata.get('drug1', '')
                d2 = metadata.get('drug2', '')
                if (drug1.lower() in d1 and drug2.lower() in d2) or \
                   (drug1.lower() in d2 and drug2.lower() in d1):
                    return result
        
        return None
    
    def get_disease_info(self, disease: str) -> List[Dict]:
        """
        Get information about a disease.
        
        Args:
            disease: Disease name
        
        Returns:
            List of relevant documents
        """
        query = f"Information about {disease} including symptoms, diagnosis, and treatment"
        results = self.query(query, top_k=3)
        
        return [r for r in results if r['metadata'].get('type') == 'disease_info']
    
    def get_medication_info(self, medication: str) -> List[Dict]:
        """
        Get information about a medication.
        
        Args:
            medication: Medication name
        
        Returns:
            List of relevant documents
        """
        query = f"Information about {medication} including indications, dosages, and side effects"
        results = self.query(query, top_k=3)
        
        return [r for r in results if r['metadata'].get('type') == 'medication_info']
    
    def verify_fact(self, claim: str) -> Dict[str, Any]:
        """
        Verify a clinical claim against the knowledge base.
        
        Args:
            claim: Clinical statement to verify
        
        Returns:
            Dictionary with verification result and supporting evidence
        """
        results = self.query(claim, top_k=3)
        
        if not results:
            return {
                "verified": False,
                "confidence": 0.0,
                "evidence": None,
                "message": "No relevant medical knowledge found"
            }
        
        # Get the most relevant result
        best = results[0]
        
        return {
            "verified": True,
            "confidence": 1.0 - (best.get('distance', 1.0) / 2),
            "evidence": best['document'],
            "source_type": best['metadata'].get('type', 'unknown'),
            "message": "Found supporting medical evidence"
        }
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of knowledge base contents."""
        try:
            all_docs = self.collection.get()
            type_counts = {}
            if all_docs.get('metadatas'):
                for meta in all_docs['metadatas']:
                    doc_type = meta.get('type', 'unknown')
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            return {
                'total_entries': self.collection.count(),
                'by_type': type_counts
            }
        except Exception:
            return {'total_entries': 0, 'by_type': {}}


def load_default_medical_knowledge(kb: MedicalKnowledgeBase):
    """
    Load default medical knowledge into the knowledge base.
    This includes common drug interactions, disease information, and guidelines.
    """
    
    print("\n📚 Loading default medical knowledge...")
    
    # Drug Interactions
    drug_interactions = [
        ("Metformin", "Ibuprofen", "moderate",
         "NSAIDs like ibuprofen can reduce the effectiveness of metformin and may increase risk of lactic acidosis in patients with renal impairment.",
         "Use alternative pain reliever like acetaminophen. Monitor renal function if NSAID use is necessary."),
        
        ("Metformin", "Lisinopril", "mild",
         "Generally safe combination, but monitor renal function as both can affect kidney function.",
         "Regular monitoring of renal function and potassium levels recommended."),
        
        ("Lisinopril", "Losartan", "severe",
         "Dual blockade of the renin-angiotensin system increases risk of hypotension, hyperkalemia, and renal failure.",
         "Avoid concurrent use. Choose one ACE inhibitor or ARB."),
        
        ("Warfarin", "Ibuprofen", "severe",
         "NSAIDs increase bleeding risk in patients taking warfarin.",
         "Avoid ibuprofen. Consider acetaminophen for pain management."),
        
        ("Metformin", "Gabapentin", "mild",
         "No significant interaction, but both may cause dizziness and fatigue.",
         "Monitor for additive CNS depression, especially in elderly patients."),
        
        ("Lisinopril", "Potassium", "moderate",
         "ACE inhibitors can increase potassium levels.",
         "Monitor potassium levels. Avoid potassium supplements unless prescribed."),
        
        ("Atorvastatin", "Clopidogrel", "mild",
         "Potential for reduced antiplatelet effect of clopidogrel.",
         "No dose adjustment needed but monitor for cardiovascular events."),
        
        ("Amlodipine", "Metformin", "mild",
         "No significant interaction.",
         "Safe to use together. Monitor blood pressure and glucose."),
    ]
    
    for interaction in drug_interactions:
        kb.add_drug_interaction(*interaction)
        print(f"  ✓ Added interaction: {interaction[0]} + {interaction[1]}")
    
    # Disease Information
    diseases = [
        ("Type 2 Diabetes",
         ["Increased thirst", "Frequent urination", "Fatigue", "Blurred vision", "Slow-healing sores"],
         "Fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, or random glucose ≥200 mg/dL with symptoms",
         "Lifestyle modifications, Metformin first-line, then add sulfonylureas, DPP-4 inhibitors, SGLT2 inhibitors, or insulin",
         ["Cardiovascular disease", "Neuropathy", "Nephropathy", "Retinopathy"]),
        
        ("Hypertension",
         ["Often asymptomatic", "Headaches", "Dizziness", "Blurred vision", "Shortness of breath"],
         "Blood pressure ≥130/80 mmHg on at least two occasions",
         "Lifestyle modifications, ACE inhibitors, ARBs, CCBs, or thiazide diuretics as first-line",
         ["Heart attack", "Stroke", "Kidney disease", "Heart failure"]),
        
        ("Chronic Kidney Disease",
         ["Often asymptomatic early", "Fatigue", "Swelling in ankles", "Urine changes", "Shortness of breath"],
         "eGFR <60 mL/min for >3 months or albuminuria >30 mg/g creatinine",
         "Control BP, ACE inhibitors/ARBs, manage diabetes, avoid NSAIDs, consider renal diet",
         ["End-stage renal disease", "Cardiovascular disease", "Anemia", "Mineral bone disorder"]),
        
        ("Diabetic Neuropathy",
         ["Numbness", "Tingling", "Burning pain", "Sharp pains", "Muscle weakness"],
         "Clinical diagnosis based on symptoms and physical exam in diabetic patient",
         "Blood glucose control, Gabapentin/Pregabalin for pain, Duloxetine/amitriptyline",
         ["Foot ulcers", "Infections", "Amputation", "Falls"]),
        
        ("Heart Failure",
         ["Shortness of breath", "Fatigue", "Swelling in legs", "Rapid heartbeat", "Persistent cough"],
         "Symptoms plus objective evidence of cardiac dysfunction (echo, BNP)",
         "ACE inhibitors/ARBs, beta-blockers, diuretics, SGLT2 inhibitors",
         ["Arrhythmias", "Kidney damage", "Liver damage", "Sudden cardiac death"]),
    ]
    
    for disease in diseases:
        kb.add_disease_info(*disease)
        print(f"  ✓ Added disease: {disease[0]}")
    
    # Medication Information
    medications = [
        ("Metformin",
         ["Type 2 Diabetes"],
         ["500mg daily", "850mg twice daily", "1000mg twice daily"],
         ["GI upset", "Nausea", "Diarrhea", "Vitamin B12 deficiency"],
         ["Severe renal impairment (eGFR <30)", "Metabolic acidosis", "Severe liver disease"]),
        
        ("Lisinopril",
         ["Hypertension", "Heart Failure", "Diabetic Nephropathy"],
         ["10mg daily", "20mg daily", "40mg daily"],
         ["Cough", "Dizziness", "Headache", "Hyperkalemia"],
         ["Angioedema history", "Pregnancy", "Bilateral renal artery stenosis"]),
        
        ("Atorvastatin",
         ["Hyperlipidemia", "Cardiovascular disease prevention"],
         ["10mg daily", "20mg daily", "40mg daily", "80mg daily"],
         ["Muscle pain", "Liver enzyme elevation", "GI symptoms"],
         ["Active liver disease", "Pregnancy", "Concurrent cyclosporine"]),
        
        ("Gabapentin",
         ["Neuropathic pain", "Postherpetic neuralgia"],
         ["300mg nightly", "300mg TID", "600mg TID"],
         ["Dizziness", "Somnolence", "Peripheral edema", "Ataxia"],
         ["Hypersensitivity", "Myasthenia gravis"]),
        
        ("Amlodipine",
         ["Hypertension", "Coronary artery disease"],
         ["5mg daily", "10mg daily"],
         ["Edema", "Dizziness", "Flushing", "Palpitations"],
         ["Hypersensitivity", "Severe aortic stenosis"]),
    ]
    
    for med in medications:
        kb.add_medication_info(*med)
        print(f"  ✓ Added medication: {med[0]}")
    
    # Lab References
    lab_references = [
        ("HbA1c", "4-5.6%", ">6.5% indicates diabetes, <7% good control",
         "Higher values indicate poorer glycemic control. Target <7% for most diabetics."),
        
        ("eGFR", ">90 mL/min", "<60 mL/min indicates CKD, <15 mL/min indicates kidney failure",
         "Lower values indicate worse kidney function. Monitor trends for CKD progression."),
        
        ("LDL Cholesterol", "<100 mg/dL", ">190 mg/dL very high risk, >70 mg/dL in high-risk patients",
         "Lower is better. Target <70 mg/dL for high-risk patients."),
        
        ("Blood Pressure", "<120/80 mmHg", ">130/80 mmHg indicates hypertension",
         "Elevated BP increases cardiovascular risk. Target <130/80 for most adults."),
    ]
    
    for lab in lab_references:
        kb.add_lab_reference(*lab)
        print(f"  ✓ Added lab reference: {lab[0]}")
    
    # Clinical Guidelines
    guidelines = [
        ("Type 2 Diabetes",
         "Metformin is first-line therapy unless contraindicated. Add second agent if HbA1c >7% after 3 months.",
         "American Diabetes Association", 2024),
        
        ("Hypertension",
         "Start treatment if BP ≥130/80. First-line: thiazide, ACE inhibitor, ARB, or CCB.",
         "American College of Cardiology", 2023),
        
        ("Hyperlipidemia",
         "Statin therapy for primary prevention if LDL ≥190 or 10-year risk >7.5%.",
         "American Heart Association", 2023),
    ]
    
    for guideline in guidelines:
        kb.add_clinical_guideline(*guideline)
        print(f"  ✓ Added guideline: {guideline[0]}")
    
    print(f"\n✅ Default knowledge loaded successfully!")
    print(f"   Total entries: {kb.collection.count()}")


def create_medical_knowledge_base():
    """Create and populate the medical knowledge base."""
    
    print("\n" + "="*60)
    print("🏥 PHASE 5: MEDICAL KNOWLEDGE BASE")
    print("="*60)
    
    # Initialize knowledge base
    kb = MedicalKnowledgeBase("data/medical_kb")
    
    # Check if already populated
    summary = kb.get_summary()
    if summary['total_entries'] > 0:
        print(f"\n📊 Knowledge base already contains {summary['total_entries']} entries")
        return kb
    
    # Load default knowledge
    load_default_medical_knowledge(kb)
    
    # Show summary
    summary = kb.get_summary()
    print(f"\n📊 Knowledge Base Summary:")
    print(f"   Total entries: {summary['total_entries']}")
    print(f"   By type:")
    for doc_type, count in summary['by_type'].items():
        print(f"     - {doc_type}: {count}")
    
    return kb


if __name__ == "__main__":
    # Test the medical knowledge base
    kb = create_medical_knowledge_base()
    
    # Test queries
    print("\n" + "="*60)
    print("🔍 Testing Knowledge Base Queries")
    print("="*60)
    
    # Test drug interaction
    print("\n1. Checking drug interaction: Metformin + Ibuprofen")
    result = kb.check_drug_interaction("Metformin", "Ibuprofen")
    if result:
        print(f"   Found: {result['document'][:200]}...")
    
    # Test disease info
    print("\n2. Getting disease info: Type 2 Diabetes")
    results = kb.get_disease_info("Type 2 Diabetes")
    if results:
        print(f"   Found: {results[0]['document'][:200]}...")
    
    # Test general query
    print("\n3. General query: 'treatment for high blood pressure'")
    results = kb.query("treatment for high blood pressure", top_k=2)
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r['metadata'].get('type', 'unknown')}] {r['document'][:150]}...")
    
    # Test fact verification
    print("\n4. Verifying claim: 'Metformin is first-line for Type 2 Diabetes'")
    verification = kb.verify_fact("Metformin is first-line for Type 2 Diabetes")
    print(f"   Verified: {verification['verified']}")
    print(f"   Confidence: {verification['confidence']:.2f}")
    
    print("\n✅ Medical Knowledge Base ready!")