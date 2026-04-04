"""
Phase 2: NLP Extraction Pipeline for GraphMed
Enhanced version with better regex patterns for clinical entities
"""

import json
import re
import spacy
from typing import Dict, List, Any, Optional, Tuple
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
        self._llm_available = bool(os.getenv("GROQ_API_KEY"))
        
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
            r'\b(gabapentin|warfarin|clopidogrel|insulin|furosemide|erythropoietin|aspirin|eplerenone)\b'
        ]
        self.dosage_pattern = r'\b(\d+(?:\.\d+)?\s*(?:mg|mcg|g)(?:\s*(?:daily|bid|tid|qid))?)\b'
        
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
        self.allowed_lab_keys = {
            "HbA1c", "BP_systolic", "BP_diastolic", "eGFR", "LDL", "HDL", "Creatinine", "Glucose", "Potassium", "Sodium"
        }
        self.lab_key_aliases = {
            "blood pressure": "BP",
            "blood_pressure": "BP",
            "bp": "BP",
            "a1c": "HbA1c",
            "hemoglobin a1c": "HbA1c",
            "gfr": "eGFR",
            "ldl-c": "LDL",
            "hdl-c": "HDL",
            "cr": "Creatinine",
            "blood sugar": "Glucose",
            "na+": "Sodium",
            "k+": "Potassium"
        }
        self.canonical_conditions = {
            "hypertension": "Hypertension",
            "high blood pressure": "Hypertension",
            "type 2 diabetes": "Type 2 Diabetes",
            "diabetes": "Type 2 Diabetes",
            "t2dm": "Type 2 Diabetes",
            "hyperlipidemia": "Hyperlipidemia",
            "dyslipidemia": "Hyperlipidemia",
            "chronic kidney disease": "Chronic Kidney Disease",
            "renal dysfunction": "Chronic Kidney Disease",
            "renal impairment": "Chronic Kidney Disease",
            "coronary artery disease": "Coronary Artery Disease",
            "cad": "Coronary Artery Disease",
            "heart failure": "Heart Failure",
            "congestive heart failure": "Heart Failure",
            "anemia": "Anemia",
            "copd": "COPD",
            "asthma": "Asthma"
        }
        self.canonical_symptoms = {
            "sob": "Shortness of breath",
            "dyspnea": "Shortness of breath",
            "breathlessness": "Shortness of breath",
            "shortness of breath": "Shortness of breath",
            "chest pain": "Chest pain",
            "fatigue": "Fatigue",
            "tiredness": "Fatigue",
            "increased thirst": "Increased thirst",
            "excessive thirst": "Increased thirst",
            "polyuria": "Frequent urination",
            "frequent urination": "Frequent urination",
            "edema": "Edema",
            "swelling": "Edema",
            "dizziness": "Dizziness",
            "nausea": "Nausea",
            "palpitations": "Palpitations"
        }
        self.relationship_rules = {
            "managed_by": {("CONDITION", "MEDICATION")},
            "has_symptom": {("CONDITION", "SYMPTOM")},
            "measured_by": {("CONDITION", "LAB_VALUE")},
            "contraindicated_with": {("MEDICATION", "MEDICATION")},
            "improves": {("MEDICATION", "CONDITION"), ("MEDICATION", "SYMPTOM"), ("PROCEDURE", "CONDITION"), ("PROCEDURE", "SYMPTOM")},
            "worsens": {("CONDITION", "CONDITION"), ("CONDITION", "SYMPTOM"), ("MEDICATION", "SYMPTOM"), ("LAB_VALUE", "CONDITION")}
        }

    def _unique_preserve_order(self, items: List[Any]) -> List[Any]:
        """Return unique values while preserving first occurrence order."""
        seen = set()
        result = []

        for item in items:
            if item is None:
                continue
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key not in seen:
                seen.add(key)
                result.append(value)

        return result

    def _safe_parse_llm_json(self, llm_text: str) -> Dict[str, Any]:
        """Parse JSON from raw LLM output, tolerating fenced code blocks."""
        if not llm_text:
            return {}

        cleaned = llm_text.strip()

        # Remove markdown fences if present.
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except Exception:
            # Try to recover by grabbing the first JSON object.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except Exception:
                    return {}
            return {}

    def _coerce_numeric(self, value: Any) -> Any:
        """Convert numeric-looking strings to float/int where possible."""
        if isinstance(value, (int, float)):
            return value

        if isinstance(value, str):
            stripped = value.strip().replace('%', '')
            if re.fullmatch(r"\d+", stripped):
                return int(stripped)
            if re.fullmatch(r"\d+\.\d+", stripped):
                return float(stripped)
        return value

    def _normalize_condition(self, value: str) -> str:
        key = value.strip().lower()
        return self.canonical_conditions.get(key, value.strip().title())

    def _normalize_symptom(self, value: str) -> str:
        key = value.strip().lower()
        return self.canonical_symptoms.get(key, value.strip().capitalize())

    def _normalize_medication(self, value: str) -> str:
        text = value.strip().lower()
        text = re.sub(self.dosage_pattern, "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _split_medications_and_dosages(self, values: List[str]) -> Tuple[List[str], List[str]]:
        medications: List[str] = []
        dosages: List[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            dosage_hits = re.findall(self.dosage_pattern, value, flags=re.IGNORECASE)
            for d in dosage_hits:
                dosages.append(d.strip())
            med = self._normalize_medication(value)
            if med:
                medications.append(med)
        return self._unique_preserve_order(medications), self._unique_preserve_order(dosages)

    def _extract_dosages_from_text(self, text: str) -> List[str]:
        return self._unique_preserve_order(re.findall(self.dosage_pattern, text, flags=re.IGNORECASE))

    def _normalize_lab_values(self, labs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(labs, dict):
            return {}

        normalized: Dict[str, Any] = {}

        for raw_key, raw_value in labs.items():
            if raw_key is None:
                continue
            key = str(raw_key).strip()
            key_lower = key.lower()
            canonical = self.lab_key_aliases.get(key_lower, key)

            # Convert BP-like free text into systolic/diastolic.
            if canonical == "BP" or canonical.lower().startswith("blood pressure"):
                bp_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", str(raw_value))
                if bp_match:
                    normalized["BP_systolic"] = int(bp_match.group(1))
                    normalized["BP_diastolic"] = int(bp_match.group(2))
                continue

            if canonical not in self.allowed_lab_keys:
                # Some LLM outputs use e.g., blood_pressure keys.
                if key_lower in ("blood pressure", "blood_pressure", "bp"):
                    bp_match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", str(raw_value))
                    if bp_match:
                        normalized["BP_systolic"] = int(bp_match.group(1))
                        normalized["BP_diastolic"] = int(bp_match.group(2))
                continue

            normalized[canonical] = self._coerce_numeric(raw_value)

        return normalized

    def _normalize_predicate(self, predicate: str) -> str:
        p = predicate.strip().lower().replace("-", "_").replace(" ", "_")
        alias = {
            "treated_by": "managed_by",
            "prescribed_for": "managed_by",
            "indicated_for": "managed_by",
            "hassign": "has_symptom",
            "associated_with": "has_symptom",
            "interaction_with": "contraindicated_with",
            "increased": "worsens",
            "decreased": "improves"
        }
        return alias.get(p, p)

    def _infer_entity_type(self, entity: str, extraction: Dict[str, Any]) -> str:
        value = entity.strip().lower()

        for c in extraction.get("conditions", []):
            if value == str(c).strip().lower():
                return "CONDITION"
        for m in extraction.get("medications", []):
            if value == str(m).strip().lower():
                return "MEDICATION"
        for s in extraction.get("symptoms", []):
            if value == str(s).strip().lower():
                return "SYMPTOM"
        for p in extraction.get("procedures", []):
            if value == str(p).strip().lower():
                return "PROCEDURE"
        for lab_name in extraction.get("lab_values", {}).keys():
            if value == str(lab_name).strip().lower():
                return "LAB_VALUE"

        return "UNKNOWN"

    def _resolve_entity_name(self, raw_entity: str, extraction: Dict[str, Any]) -> str:
        """Resolve an entity string to canonical names already present in extraction."""
        value = raw_entity.strip()
        if not value:
            return ""

        value_lower = value.lower()

        # Resolve from known extracted entities first.
        for key in ("conditions", "medications", "symptoms", "procedures"):
            for item in extraction.get(key, []):
                if value_lower == str(item).strip().lower():
                    return str(item)

        for lab_key in extraction.get("lab_values", {}).keys():
            if value_lower == str(lab_key).strip().lower():
                return str(lab_key)

        # Fallback normalization by domain.
        med = self._normalize_medication(value)
        if med:
            for item in extraction.get("medications", []):
                if med == str(item).strip().lower():
                    return str(item)

        cond = self._normalize_condition(value)
        if cond:
            for item in extraction.get("conditions", []):
                if cond.lower() == str(item).strip().lower():
                    return str(item)

        sym = self._normalize_symptom(value)
        if sym:
            for item in extraction.get("symptoms", []):
                if sym.lower() == str(item).strip().lower():
                    return str(item)

        # Keep original if we cannot resolve safely.
        return value

    def _validate_relationship(self, rel: Dict[str, Any], extraction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        subj = self._resolve_entity_name(str(rel.get("subject", "")), extraction)
        obj = self._resolve_entity_name(str(rel.get("object", "")), extraction)
        pred = self._normalize_predicate(str(rel.get("predicate", "")))

        if not subj or not obj or not pred:
            return None

        subj_type = self._infer_entity_type(subj, extraction)
        obj_type = self._infer_entity_type(obj, extraction)
        allowed_pairs = self.relationship_rules.get(pred)

        if not allowed_pairs:
            return None

        if (subj_type, obj_type) in allowed_pairs:
            return {"subject": subj, "predicate": pred, "object": obj}

        # Try reversible predicates by flipping direction.
        if pred in {"managed_by", "measured_by"} and (obj_type, subj_type) in allowed_pairs:
            return {"subject": obj, "predicate": pred, "object": subj}

        return None

    def _llm_enrich(self, text: str, stage1: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: LLM enrichment for context/procedures/relationships."""
        if not self._llm_available:
            return {}

        prompt = f"""You are a clinical information extraction engine.
Given a clinical note and Stage 1 extracted entities, enrich extraction with contextual relations.

Return ONLY valid JSON with this exact schema:
{{
  "conditions": ["..."],
  "medications": ["..."],
    "dosages": ["..."],
  "lab_values": {{"name": value}},
  "symptoms": ["..."],
  "procedures": ["..."],
  "relationships": [
    {{"subject": "...", "predicate": "managed_by|has_symptom|measured_by|contraindicated_with|improves|worsens", "object": "...", "confidence": 0.0}}
  ]
}}

Rules:
- Keep entities clinically grounded in the note.
- Prefer concise canonical names.
- For labs, only use keys: HbA1c, BP_systolic, BP_diastolic, eGFR, LDL, HDL, Creatinine, Glucose, Potassium, Sodium.
- Include relationships only when strongly implied.
- confidence must be between 0 and 1.

Clinical Note:
{text}

Stage 1 Extraction:
{json.dumps(stage1, ensure_ascii=False)}
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            content = response.choices[0].message.content
            parsed = self._safe_parse_llm_json(content)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _merge_extractions(self, stage1: Dict[str, Any], llm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge Stage 1 deterministic output with Stage 2 LLM enrichment."""
        merged = {
            "conditions": self._unique_preserve_order([self._normalize_condition(x) for x in (stage1.get("conditions", []) + llm_data.get("conditions", []))]),
            "medications": [],
            "dosages": [],
            "symptoms": self._unique_preserve_order([self._normalize_symptom(x) for x in (stage1.get("symptoms", []) + llm_data.get("symptoms", []))]),
            "procedures": self._unique_preserve_order(llm_data.get("procedures", [])),
            "relationships": [],
            "lab_values": {}
        }

        merged_meds, merged_dosages = self._split_medications_and_dosages(
            stage1.get("medications", []) + llm_data.get("medications", [])
        )
        llm_dosages = llm_data.get("dosages", []) if isinstance(llm_data.get("dosages", []), list) else []
        merged["medications"] = merged_meds
        merged["dosages"] = self._unique_preserve_order(merged_dosages + llm_dosages)

        # Stage 1 labs are generally more reliable for exact values; LLM fills gaps.
        stage1_labs = stage1.get("lab_values", {}) if isinstance(stage1.get("lab_values", {}), dict) else {}
        llm_labs = llm_data.get("lab_values", {}) if isinstance(llm_data.get("lab_values", {}), dict) else {}

        merged_labs: Dict[str, Any] = self._normalize_lab_values(stage1_labs)
        llm_labs_norm = self._normalize_lab_values(llm_labs)
        for key, value in llm_labs_norm.items():
            if key not in merged_labs:
                merged_labs[key] = value

        merged["lab_values"] = merged_labs

        # Validate relationship objects.
        extraction_for_types = {
            "conditions": merged["conditions"],
            "medications": merged["medications"],
            "symptoms": merged["symptoms"],
            "procedures": merged["procedures"],
            "lab_values": merged["lab_values"]
        }
        for rel in llm_data.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            validated = self._validate_relationship(rel, extraction_for_types)
            if validated:
                relationship = validated
                conf = rel.get("confidence")
                if isinstance(conf, (int, float)):
                    relationship["confidence"] = max(0.0, min(1.0, float(conf)))
                merged["relationships"].append(relationship)

        return merged
    
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
        
        normalized = [self._normalize_condition(c) for c in conditions]
        return self._unique_preserve_order(normalized)
    
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
        
        meds, _ = self._split_medications_and_dosages(medications)
        return meds
    
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
        
        normalized = [self._normalize_symptom(s) for s in symptoms]
        return self._unique_preserve_order(normalized)
    
    def extract_lab_values(self, text: str) -> Dict[str, Any]:
        """Extract lab values using regex patterns."""
        labs = {}
        
        for lab_name, pattern in self.lab_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if lab_name == 'BP_systolic':
                    # Handle blood pressure specially
                    if len(match.groups()) >= 2:
                        labs['BP_systolic'] = int(match.group(2))
                        labs['BP_diastolic'] = int(match.group(3))
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
        
        return self._normalize_lab_values(labs)
    
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

        entities["conditions"] = [self._normalize_condition(x) for x in entities["conditions"]]
        entities["symptoms"] = [self._normalize_symptom(x) for x in entities["symptoms"]]
        entities["medications"], _ = self._split_medications_and_dosages(entities["medications"])
        entities["conditions"] = self._unique_preserve_order(entities["conditions"])
        entities["symptoms"] = self._unique_preserve_order(entities["symptoms"])
        
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
        # Stage 1: deterministic extraction (regex + SciSpacy fallback)
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
        
        stage1 = {
            "conditions": conditions,
            "medications": medications,
            "dosages": self._extract_dosages_from_text(text),
            "lab_values": lab_values,
            "symptoms": symptoms,
            "procedures": [],
            "relationships": []
        }

        # Stage 2: LLM enrichment for relationships/context.
        should_use_llm = use_llm if use_llm is not None else self.use_llm
        if should_use_llm:
            llm_data = self._llm_enrich(text, stage1)
            return self._merge_extractions(stage1, llm_data)

        return stage1

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
        
        # Merge top-level fields to avoid losing pre-existing structured data.
        existing_diagnoses = [extractor._normalize_condition(x) for x in visit.get('diagnoses', [])]
        existing_meds, existing_dosages = extractor._split_medications_and_dosages(visit.get('medications', []))
        existing_symptoms = [extractor._normalize_symptom(x) for x in visit.get('symptoms', [])]
        existing_labs = extractor._normalize_lab_values(visit.get('labs', {}))

        visit['diagnoses'] = extractor._unique_preserve_order(existing_diagnoses + extracted.get('conditions', []))
        visit['medications'] = extractor._unique_preserve_order(existing_meds + extracted.get('medications', []))
        visit['symptoms'] = extractor._unique_preserve_order(existing_symptoms + extracted.get('symptoms', []))
        visit['dosages'] = extractor._unique_preserve_order(existing_dosages + extracted.get('dosages', []))

        merged_labs = {}
        if isinstance(existing_labs, dict):
            merged_labs.update(existing_labs)
        extracted_labs = extractor._normalize_lab_values(extracted.get('lab_values', {}))
        if isinstance(extracted_labs, dict):
            for key, value in extracted_labs.items():
                if key not in merged_labs:
                    merged_labs[key] = value
        if merged_labs:
            visit['labs'] = merged_labs
    
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