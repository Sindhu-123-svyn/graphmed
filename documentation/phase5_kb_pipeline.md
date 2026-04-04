# Phase 5 KB Pipeline (Priority Ingestion)

## Goal
Build a non-patient-specific external medical knowledge base for grounded agent responses.

Priority order:
1. DrugBank Open Data
2. CDC Clinical Guidelines
3. PubMed (targeted topics)
4. MedlinePlus Drug Information (optional)

## Expected data layout
Create this structure under data/medical_sources:

- data/medical_sources/drugbank/drugbank_open.json
- data/medical_sources/cdc_guidelines/
  - .txt / .md / .pdf files
- data/medical_sources/medlineplus/medlineplus_drugs.csv

Optional env var for PubMed API:
- PUBMED_EMAIL=<your_email>

## Supported source schemas

### 1) DrugBank JSON
Top-level can be list or object with key drugs.
Each drug object should include as many fields as available:
- name
- description
- mechanism_of_action
- indications
- contraindications
- adverse_effects
- interactions: list of objects
  - target
  - severity
  - description
  - recommendation

### 2) CDC guidelines directory
Supported files:
- .txt
- .md
- .pdf (requires pdfplumber)

Topic is inferred from filename.

### 3) PubMed targeted ingestion
No local files needed.
Pipeline pulls abstracts via NCBI E-utilities for a predefined topic list.
PUBMED_EMAIL is required to enable this step.

### 4) MedlinePlus CSV
Required columns:
- drug_name
- summary

Optional columns:
- uses
- warnings
- side_effects
- source_url

## Chunking schema
Default chunk policy:
- chunk_chars: 1200
- overlap_chars: 220

CDC guideline chunks use larger windows:
- chunk_chars: 1400
- overlap_chars: 260

Drug interaction chunks use tighter windows:
- chunk_chars: 900
- overlap_chars: 120

## Metadata schema (stored per chunk)
Common fields:
- type
- category
- title
- source
- source_priority
- chunk_index
- chunk_count
- char_count
- ingested_at
- schema_version

Source-specific fields (when available):
- medication
- condition
- drug1
- drug2
- severity
- source_org
- file_name
- dataset
- evidence_level
- pmid
- journal
- year
- source_url

## Retrieval strategy
Query pipeline in Chroma:
1. Semantic retrieval (broad candidates)
2. Intent detection from query
3. Weighted reranking by:
- semantic similarity
- source_priority bonus (DrugBank > CDC > PubMed > MedlinePlus)
- intent-to-doc-type match
- evidence_level bonus (guideline/reference)

## Run
Use Phase 5 runner:
- python phase5_build_medical_kb.py

Menu options include:
- Build/update with priority pipeline
- Rebuild from scratch
- Query tests
- Interactive mode
- Summary with source/type breakdown
