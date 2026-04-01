"""
Phase 2: NLP Extraction Pipeline for GraphMed
Enhanced version with better regex patterns for clinical entities
"""

import json
import re
import spacy
from typing import Dict, List, Any, Optional
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load spaCy models
try:
    nlp = spacy.load("en_core_sci_md")
    print("✅ Loaded SciSpacy biomedical model")
except:
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ Loaded spaCy model")
    except:
        print("⚠️  No spaCy model found. Run: python -m spacy download en_core_web_sm")
        nlp = None

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ClinicalExtractor:
    """Extract structured clinical entities from medical notes."""
    
    def __init__(self, use_llm: bool = False):  # Default to False for speed
        """
        Initialize the extractor.
        
        Args:
            use_llm: Whether to use LLM for relation extraction (slower but more accurate)
        """
        self.use_llm = use_llm
        
        # Condition patterns
        self.condition_patterns = {
            'hypertension': r'\b(hypertension|htn|high blood pressure)\b',
            'diabetes': r'\b(diabetes|type 2 diabetes|t2dm|dm|type ii diabetes)\b',
            'hyperlipidemia': r'\b(hyperlipidemia|high cholesterol|dyslipidemia)\b',
            'ckd': r'\b(chronic kidney disease|ckd|renal impairment|kidney disease)\b',
            'neuropathy': r'\b(neuropathy|diabetic neuropathy)\b',
            'heart_failure': r'\b(heart failure|hf|congestive heart failure)\b',
            'cad': r'\b(coronary artery disease|cad)\b',
            'anemia': r'\b(anemia|low hemoglobin)\b',
            'copd': r'\b(copd|chronic obstructive pulmonary disease)\b',
            'asthma': r'\b(asthma)\b'
        }
        
        # Medication patterns
        self.medication_patterns = [
            # Common antihypertensives
            r'\b(metformin|lisinopril|losartan|amlodipine|metoprolol|atenolol|hydrochlorothiazide)\b',
            # Statins
            r'\b(atorvastatin|simvastatin|rosuvastatin)\b',
            # Other
            r'\b(gabapentin|warfarin|clopidogrel|insulin)\b',
            # Dosages (capture as separate entities)
            r'\b(\d+(?:\.\d+)?\s*(?:mg|mcg|g|mg/day|mg daily|mg bid|mg tid))\b'
        ]
        
        # Symptom patterns
        self.symptom_patterns = {
            'fatigue': r'\b(fatigue|tiredness|exhaustion|low energy)\b',
            'sob': r'\b(shortness of breath|sob|dyspnea|breathlessness)\b',
            'chest_pain': r'\b(chest pain|angina|chest discomfort)\b',
            'polyuria': r'\b(polyuria|frequent urination)\b',
            'polydipsia': r'\b(polydipsia|increased thirst|excessive thirst)\b',
            'tingling': r'\b(tingling|numbness|paresthesia)\b',
            'edema': r'\b(edema|swelling|fluid retention)\b',
            'dizziness': r'\b(dizziness|lightheadedness|vertigo)\b',
            'headache': r'\b(headache)\b',
            'nausea': r'\b(nausea|vomiting)\b'
        }
        
        # Lab patterns
        self.lab_patterns = {
            'HbA1c': r'\b(HbA1c|A1c|hemoglobin A1c)\s*:?\s*(\d+\.?\d*)\s*%?',
            'BP_systolic': r'\b(blood pressure|BP)\s*:?\s*(\d+)/(\d+)',
            'eGFR': r'\b(eGFR|GFR)\s*:?\s*(\d+)\s*(?:mL/min)?',
            'LDL': r'\b(LDL|LDL-C)\s*:?\s*(\d+)\s*(?:mg/dL)?',
            'HDL': r'\b(HDL|HDL-C)\s*:?\s*(\d+)',
            'Creatinine': r'\b(creatinine|Cr)\s*:?\s*(\d+\.?\d*)\s*(?:mg/dL)?',
            'Glucose': r'\b(glucose|blood sugar|BG)\s*:?\s*(\d+)\s*(?:mg/dL)?',
            'Potassium': r'\b(potassium|K\+)\s*:?\s*(\d+\.?\d*)',
            'Sodium': r'\b(sodium|Na\+)\s*:?\s*(\d+)'
        }
    
    def extract_conditions(self, text: str) -> List[str]:
        """Extract conditions using regex patterns."""
        conditions = []
        text_lower = text.lower()
        
        for condition_name, pattern in self.condition_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Format condition name nicely
                if condition_name == 'hypertension':
                    conditions.append('Hypertension')
                elif condition_name == 'diabetes':
                    conditions.append('Type 2 Diabetes')
                elif condition_name == 'hyperlipidemia':
                    conditions.append('Hyperlipidemia')
                elif condition_name == 'ckd':
                    conditions.append('Chronic Kidney Disease')
                elif condition_name == 'neuropathy':
                    conditions.append('Diabetic Neuropathy')
                elif condition_name == 'heart_failure':
                    conditions.append('Heart Failure')
                elif condition_name == 'cad':
                    conditions.append('Coronary Artery Disease')
                elif condition_name == 'anemia':
                    conditions.append('Anemia')
                elif condition_name == 'copd':
                    conditions.append('COPD')
                elif condition_name == 'asthma':
                    conditions.append('Asthma')
        
        return list(set(conditions))
    
    def extract_medications(self, text: str) -> List[str]:
        """Extract medications using regex patterns."""
        medications = []
        text_lower = text.lower()
        
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                med = match.group(0).lower()
                if med not in medications:
                    medications.append(med)
        
        return medications
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms using regex patterns."""
        symptoms = []
        text_lower = text.lower()
        
        for symptom_name, pattern in self.symptom_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Format symptom name nicely
                if symptom_name == 'sob':
                    symptoms.append('Shortness of breath')
                elif symptom_name == 'chest_pain':
                    symptoms.append('Chest pain')
                else:
                    symptoms.append(symptom_name.capitalize())
        
        return list(set(symptoms))
    
    def extract_lab_values(self, text: str) -> Dict[str, Any]:
        """Extract lab values using regex patterns."""
        labs = {}
        
        for lab_name, pattern in self.lab_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if lab_name == 'BP_systolic':
                    # Handle blood pressure specially
                    if len(match.groups()) >= 2:
                        labs['BP'] = f"{match.group(2)}/{match.group(3)}"
                else:
                    # Extract the numeric value
                    value_str = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    try:
                        # Try to convert to float if numeric
                        if value_str and value_str.replace('.', '').replace('-', '').isdigit():
                            labs[lab_name] = float(value_str)
                        else:
                            labs[lab_name] = value_str
                    except:
                        labs[lab_name] = value_str
        
        return labs
    
    def extract_with_scispacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using SciSpacy NER (backup)."""
        if not nlp:
            return {"conditions": [], "medications": [], "symptoms": []}
        
        doc = nlp(text)
        
        entities = {
            "conditions": [],
            "medications": [],
            "symptoms": []
        }
        
        # Map spaCy entity labels
        label_map = {
            "DISEASE": "conditions",
            "CONDITION": "conditions",
            "MEDICATION": "medications",
            "DRUG": "medications",
            "SYMPTOM": "symptoms"
        }
        
        for ent in doc.ents:
            category = label_map.get(ent.label_, None)
            if category and ent.text not in entities[category]:
                entities[category].append(ent.text)
        
        return entities
    
    def extract(self, text: str, use_llm: bool = False) -> Dict[str, Any]:
        """
        Complete extraction pipeline using regex patterns (fast and reliable).
        
        Args:
            text: Clinical note text
            use_llm: Whether to use LLM (default False for speed)
        
        Returns:
            Dictionary with extracted entities
        """
        # Extract using regex patterns
        conditions = self.extract_conditions(text)
        medications = self.extract_medications(text)
        symptoms = self.extract_symptoms(text)
        lab_values = self.extract_lab_values(text)
        
        # Use SciSpacy as backup if regex didn't find anything
        if not conditions and not medications and not symptoms:
            spacy_results = self.extract_with_scispacy(text)
            conditions = conditions or spacy_results.get('conditions', [])
            medications = medications or spacy_results.get('medications', [])
            symptoms = symptoms or spacy_results.get('symptoms', [])
        
        return {
            "conditions": conditions,
            "medications": medications,
            "lab_values": lab_values,
            "symptoms": symptoms,
            "procedures": [],
            "relationships": []
        }

def extract_entities(note_text: str, use_llm: bool = False) -> Dict[str, Any]:
    """Convenience function to extract entities from a clinical note."""
    extractor = ClinicalExtractor(use_llm=use_llm)
    return extractor.extract(note_text, use_llm)

def process_patient_visits(patient_data: Dict[str, Any], use_llm: bool = False) -> Dict[str, Any]:
    """Process all visits for a patient and add extracted entities."""
    extractor = ClinicalExtractor(use_llm=use_llm)
    
    for visit in patient_data.get('visits', []):
        note = visit.get('note', '')
        
        # Extract entities from note
        extracted = extractor.extract(note, use_llm)
        
        # Add extracted data to visit
        visit['extracted'] = extracted
        
        # Update top-level fields
        if extracted.get('conditions'):
            visit['diagnoses'] = extracted['conditions']
        if extracted.get('medications'):
            visit['medications'] = extracted['medications']
        if extracted.get('symptoms'):
            visit['symptoms'] = extracted['symptoms']
        if extracted.get('lab_values'):
            visit['labs'] = extracted['lab_values']
    
    return patient_data

if __name__ == "__main__":
    # Test the enhanced extractor
    test_note = "Patient has hypertension and diabetes. BP 140/90, HbA1c 7.2. Reports fatigue and shortness of breath."
    
    extractor = ClinicalExtractor()
    result = extractor.extract(test_note)
    
    print("Test extraction:")
    print(f"  Conditions: {result['conditions']}")
    print(f"  Medications: {result['medications']}")
    print(f"  Lab values: {result['lab_values']}")
    print(f"  Symptoms: {result['symptoms']}")