"""
Phase 2: NLP Extraction Pipeline for GraphMed
Extracts structured entities from clinical notes using SciSpacy + LLM
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
    # Try SciSpacy first (biomedical)
    nlp = spacy.load("en_core_sci_md")
    print("✅ Loaded SciSpacy biomedical model")
except:
    try:
        # Fallback to regular spaCy
        nlp = spacy.load("en_core_web_sm")
        print("✅ Loaded spaCy model")
    except:
        print("⚠️  No spaCy model found. Run: python -m spacy download en_core_web_sm")
        nlp = None

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ClinicalExtractor:
    """Extract structured clinical entities from medical notes."""
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the extractor.
        
        Args:
            use_llm: Whether to use LLM for relation extraction (slower but more accurate)
        """
        self.use_llm = use_llm
        
        # Medical term patterns for rule-based extraction
        self.medication_patterns = [
            r'\b(metformin|lisinopril|gabapentin|insulin|atorvastatin|amlodipine|losartan|hydrochlorothiazide)\b',
            r'\b(\d+mg|\d+ mcg|\d+ g)\b',
            r'\b(once daily|twice daily|three times daily|QD|BID|TID)\b'
        ]
        
        self.lab_patterns = {
            'HbA1c': r'\b(HbA1c|A1c|hemoglobin A1c)\s*:?\s*(\d+\.?\d*)',
            'BP': r'\b(blood pressure|BP)\s*:?\s*(\d+)/(\d+)',
            'eGFR': r'\b(eGFR|GFR)\s*:?\s*(\d+)',
            'LDL': r'\b(LDL|LDL-C)\s*:?\s*(\d+)',
            'HDL': r'\b(HDL|HDL-C)\s*:?\s*(\d+)',
            'Glucose': r'\b(glucose|blood sugar|BG)\s*:?\s*(\d+)',
            'Creatinine': r'\b(creatinine|Cr)\s*:?\s*(\d+\.?\d*)'
        }
    
    def extract_with_scispacy(self, text: str) -> Dict[str, List[str]]:
        """
        Stage 1: Extract entities using SciSpacy NER.
        
        Returns:
            Dictionary with entity types and their mentions
        """
        if not nlp:
            return {"conditions": [], "medications": [], "symptoms": []}
        
        doc = nlp(text)
        
        entities = {
            "conditions": [],
            "medications": [],
            "symptoms": [],
            "procedures": []
        }
        
        # Map spaCy entity labels to our categories
        label_map = {
            "DISEASE": "conditions",
            "CONDITION": "conditions",
            "MEDICATION": "medications",
            "DRUG": "medications",
            "SYMPTOM": "symptoms",
            "PROCEDURE": "procedures"
        }
        
        for ent in doc.ents:
            category = label_map.get(ent.label_, None)
            if category and ent.text not in entities[category]:
                entities[category].append(ent.text)
        
        return entities
    
    def extract_lab_values(self, text: str) -> Dict[str, Any]:
        """
        Extract lab values using regex patterns.
        """
        labs = {}
        
        for lab_name, pattern in self.lab_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if lab_name == 'BP':
                    labs[lab_name] = f"{match.group(2)}/{match.group(3)}"
                else:
                    value = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    labs[lab_name] = float(value) if value.replace('.', '').isdigit() else value
        
        return labs
    
    def extract_medications_regex(self, text: str) -> List[str]:
        """
        Extract medications using regex patterns.
        """
        medications = []
        
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                med = match.group(0).lower()
                if med not in medications:
                    medications.append(med)
        
        return medications
    
    def extract_with_llm(self, text: str, context_entities: List[str] = None) -> Dict[str, Any]:
        """
        Stage 2: Use LLM for structured extraction with relationships.
        
        Args:
            text: Clinical note text
            context_entities: Entities already detected by SciSpacy (for context)
        
        Returns:
            Structured JSON with extracted entities
        """
        context_str = f"Previously detected terms: {context_entities}" if context_entities else ""
        
        prompt = f"""Extract medical entities from this clinical note and return ONLY valid JSON.

Clinical note: {text}

{context_str}

Return JSON with exactly these keys:
{{
    "conditions": ["list of medical conditions/diseases"],
    "medications": ["list of medications with dosages"],
    "lab_values": {{"lab_name": value}},
    "symptoms": ["list of symptoms"],
    "procedures": ["list of procedures/tests"],
    "relationships": [
        {{"subject": "condition/med", "predicate": "managed_by/causes", "object": "medication/condition"}}
    ]
}}

Extract only what's explicitly mentioned. Use null for missing values.
"""
        
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            
            # Extract JSON from response
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = result[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self.get_empty_extraction()
                
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return self.get_empty_extraction()
    
    def get_empty_extraction(self) -> Dict[str, Any]:
        """Return empty extraction structure."""
        return {
            "conditions": [],
            "medications": [],
            "lab_values": {},
            "symptoms": [],
            "procedures": [],
            "relationships": []
        }
    
    def extract(self, note: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Complete extraction pipeline.
        
        Args:
            note: Clinical note text
            use_llm: Whether to use LLM for extraction
        
        Returns:
            Combined extraction results
        """
        # Stage 1: SciSpacy extraction
        spacy_results = self.extract_with_scispacy(note)
        
        # Extract lab values with regex
        lab_values = self.extract_lab_values(note)
        
        # Extract medications with regex
        medications_regex = self.extract_medications_regex(note)
        
        # Stage 2: LLM extraction (optional)
        if use_llm and self.use_llm:
            llm_results = self.extract_with_llm(note, spacy_results["conditions"])
            
            # Merge results (LLM takes precedence for complex entities)
            merged = {
                "conditions": list(set(spacy_results["conditions"] + llm_results.get("conditions", []))),
                "medications": list(set(medications_regex + llm_results.get("medications", []))),
                "lab_values": {**lab_values, **llm_results.get("lab_values", {})},
                "symptoms": list(set(spacy_results["symptoms"] + llm_results.get("symptoms", []))),
                "procedures": list(set(spacy_results["procedures"] + llm_results.get("procedures", []))),
                "relationships": llm_results.get("relationships", [])
            }
        else:
            # Use only rule-based extraction
            merged = {
                "conditions": spacy_results["conditions"],
                "medications": medications_regex,
                "lab_values": lab_values,
                "symptoms": spacy_results["symptoms"],
                "procedures": spacy_results["procedures"],
                "relationships": []
            }
        
        return merged

def extract_entities(note_text: str, use_llm: bool = True) -> Dict[str, Any]:
    """
    Convenience function to extract entities from a clinical note.
    
    Args:
        note_text: The clinical note text
        use_llm: Whether to use LLM for extraction
    
    Returns:
        Dictionary with extracted entities
    """
    extractor = ClinicalExtractor(use_llm=use_llm)
    return extractor.extract(note_text, use_llm)

def process_patient_visits(patient_data: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
    """
    Process all visits for a patient and add extracted entities.
    
    Args:
        patient_data: Patient JSON data with visits
        use_llm: Whether to use LLM for extraction
    
    Returns:
        Patient data with extraction results added
    """
    extractor = ClinicalExtractor(use_llm=use_llm)
    
    for visit in patient_data.get('visits', []):
        note = visit.get('note', '')
        
        # Extract entities from note
        extracted = extractor.extract(note, use_llm)
        
        # Add extracted data to visit
        visit['extracted'] = extracted
        
        # Also update top-level fields if they're empty (helpful for graph construction)
        if not visit.get('symptoms'):
            visit['symptoms'] = extracted.get('symptoms', [])
        if not visit.get('medications'):
            visit['medications'] = extracted.get('medications', [])
        if not visit.get('diagnoses'):
            visit['diagnoses'] = extracted.get('conditions', [])
        if not visit.get('labs'):
            visit['labs'] = extracted.get('lab_values', {})
    
    return patient_data

# Test function
def test_extraction():
    """Test the extraction pipeline on sample notes."""
    
    test_notes = [
        "Patient diagnosed with Type 2 Diabetes. Started on Metformin 500mg. HbA1c: 7.2",
        "Blood pressure 140/90. Patient reports fatigue and polyuria. Continue Metformin.",
        "eGFR 68. Added Lisinopril 10mg for hypertension. Neuropathy symptoms improving."
    ]
    
    print("\n" + "="*60)
    print("Testing Extraction Pipeline")
    print("="*60)
    
    for i, note in enumerate(test_notes, 1):
        print(f"\nTest {i}:")
        print(f"Note: {note[:80]}...")
        
        result = extract_entities(note, use_llm=False)
        
        print(f"  Conditions: {result['conditions']}")
        print(f"  Medications: {result['medications']}")
        print(f"  Lab values: {result['lab_values']}")
        print(f"  Symptoms: {result['symptoms']}")

if __name__ == "__main__":
    test_extraction()