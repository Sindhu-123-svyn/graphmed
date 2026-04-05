# GraphMed Phase 1 to Phase 9 Implementation Report

This document is a single reference for what was implemented in each phase, what each phase produces, and how to run each phase end-to-end in this repository.

## 0. Prerequisites (Run Once)

### Environment
- OS: Windows (PowerShell examples shown)
- Python: 3.11+ recommended
- Virtual environment: `venv`

### Setup
```powershell
# from project root
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional (LangGraph-specific stacks if needed in your setup):
```powershell
pip install -r requirements_langgraph.txt
```

### API Keys
Configure required keys in `.env` (examples already used in this project):
- `GROQ_API_KEY`
- `OPEN_ROUTER_API_KEY`
- `GOOGLE_API_KEY` (optional)
- `PUBMED_EMAIL` (for KB retrieval paths)

## 1. Phase 1 - Synthetic Data Generation and Normalization

### Brief Explanation
Phase 1 creates synthetic longitudinal patient timelines and validates/normalizes raw patient JSON files before downstream processing.

### What Was Accomplished
- Generated synthetic patients with 3-5 visits over realistic time progression.
- Added Phase 1 normalization pipeline to standardize BP fields (`BP` to `BP_systolic`/`BP_diastolic`).
- Added validation and reproducibility report generation.
- Outputs stored under `data/patients` and reports under `data/reports`.

### Key Scripts
- `phase1_generate_data.py`
- `phase1_normalize_patients.py`
- `phase1_validate.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1
python phase1_generate_data.py
python phase1_normalize_patients.py --input-dir data/patients --report-path data/reports/patient_normalization_report.json
python phase1_validate.py
```

### Main Outputs
- `data/patients/*.json`
- `data/reports/patient_normalization_report.json`

## 2. Phase 2 - Clinical NLP Extraction

### Brief Explanation
Phase 2 processes raw patient notes and extracts structured entities per visit (conditions, medications, symptoms, labs, relationships).

### What Was Accomplished
- Implemented rule-based and optional LLM-enhanced extraction.
- Added full-batch and test-batch processing options.
- Added reprocessing utility for improved extraction updates.

### Key Scripts
- `phase2_process_patients.py`
- `phase2_reprocess.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1
python phase2_process_patients.py
# choose option 2 (fast, rule-based) or option 3 (LLM-assisted)

# optional full reprocess utility
python phase2_reprocess.py
```

### Main Outputs
- `data/patients_processed/*.json`

## 3. Phase 3 - Knowledge Graph Construction and Validation

### Brief Explanation
Phase 3 builds patient-specific temporal knowledge graphs from processed visit entities and validates graph quality.

### What Was Accomplished
- Built graph construction pipeline with node/edge typing and temporal metadata.
- Added graph summary and visualization options.
- Added dedicated graph validation report script.

### Key Scripts
- `phase3_build_graphs.py`
- `phase3_validate_graphs.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1
python phase3_build_graphs.py
# choose option 1 for all patients or test options

python phase3_validate_graphs.py --graphs-dir data/graphs --report-path data/reports/phase3_graph_validation_report.json
```

### Main Outputs
- `data/graphs/*_graph.json`
- `data/reports/phase3_graph_validation_report.json`

## 4. Phase 4 - Vector Memory Store

### Brief Explanation
Phase 4 builds semantic memory stores from patient visit narratives for retrieval-augmented longitudinal reasoning.

### What Was Accomplished
- Implemented patient-level and global semantic retrieval over visit narratives.
- Added retrieval tests and memory summary tooling.
- Added telemetry-noise suppression patterns for Chroma runtime stability.

### Key Scripts
- `phase4_build_memory.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1
python phase4_build_memory.py
# choose option 1 for all patients or option 2 for test subset
```

### Main Outputs
- `data/chroma_db/` (patient memory vector store)

## 5. Phase 5 - Medical Knowledge Base

### Brief Explanation
Phase 5 creates/updates the external medical KB used for grounded clinical responses and lookups.

### What Was Accomplished
- Implemented medical KB build pipeline with source-priority strategy.
- Added rebuild path and interactive query mode.
- Added test queries for drug interaction, disease info, labs, and guideline content.

### Key Scripts
- `phase5_build_medical_kb.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1
python phase5_build_medical_kb.py
# option 1 = build/update, option 2 = rebuild from scratch
```

### Main Outputs
- `data/medical_kb/`

## 6. Phase 6 - Clinical Agent Runtime

### Brief Explanation
Phase 6 runs the GraphMed clinical reasoning agent (interactive and demo flows) over patient context.

### What Was Accomplished
- Implemented interactive session-based QA mode.
- Added patient switching and compact reasoning trace views.
- Added demo workflow for repeatable quick validation.

### Key Scripts
- `phase6_run_agent.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1
python phase6_run_agent.py
# option 1 = interactive mode, option 2 = demo mode
```

### Main Outputs
- Runtime answers in console
- Agent trace context during sessions

## 7. Phase 7 - Conflict Classifier Data and Training

### Brief Explanation
Phase 7 generates contradiction-classification data and trains a LoRA-based conflict classifier.

### What Was Accomplished
- Built synthetic pair generation pipeline for CONSISTENT vs CONFLICT examples.
- Added stratified train/val/test generation.
- Added LoRA fine-tuning pipeline, evaluation metrics, and metadata export.

### Key Scripts
- `phase7_generate_training_data.py`
- `phase7_train_conflict_classifier.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1
python phase7_generate_training_data.py
python phase7_train_conflict_classifier.py
# choose option 2 for train, option 3 for quick test, option 4 for full pipeline
```

### Main Outputs
- `data/conflict_data/train.json`
- `data/conflict_data/val.json`
- `data/conflict_data/test.json`
- `models/conflict_classifier/`
- `models/conflict_classifier/training_metadata.json`

## 8. Phase 8 - Graph Evolution and Conflict Resolution

### Brief Explanation
Phase 8 validates graph evolution behavior over time, including ADD/UPDATE/CONFLICT operations and trend tracking.

### What Was Accomplished
- Added operation-level tests for entity evolution.
- Added targeted conflict-detection and lab-trend tests.
- Added real-patient evolution test path.

### Key Scripts
- `phase8_test_evolution.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1
python phase8_test_evolution.py
# option 5 runs all tests
```

### Main Outputs
- Updated graph artifacts under `data/graphs/`
- Evolution test diagnostics in console

## 9. Phase 9 - Evaluation Framework

### Brief Explanation
Phase 9 evaluates GraphMed against baseline systems across factual QA, longitudinal consistency, and contradiction detection.

### What Was Accomplished
- Implemented three experiments:
  - E1: factual accuracy (50 QA pairs)
  - E2: longitudinal consistency (10 patients x 5 questions)
  - E3: contradiction detection (30 injected contradictions)
- Added JSON/CSV artifact export for reproducibility.
- Added runtime mode tagging (`full_stack` vs `fallback`) and provider mode tracking.
- Added OpenRouter provider preference path for higher rate-limit headroom.
- Added E1 answer normalization and deterministic evaluation behavior.
- Added GraphMed score-improvement mechanisms (query routing, grounding-first actions, structured output contract).

### Key Scripts
- `phase9_run_evaluation.py`
- `src/evaluation.py`

### How to Run
```powershell
.\venv\Scripts\Activate.ps1

# Recommended provider selection for evaluation
$env:PHASE9_LLM_PROVIDER='openrouter'

# Run evaluation menu
python phase9_run_evaluation.py
# option 1 = all experiments
# option 2 = E1 only
# option 3 = E2 only
# option 4 = E3 only
```

Optional non-interactive E1 run:
```powershell
$env:PHASE9_LLM_PROVIDER='openrouter'; "2" | python phase9_run_evaluation.py
```

### Main Outputs
- `evaluation/results/phase9/experiment1_factual_accuracy.json`
- `evaluation/results/phase9/experiment1_factual_accuracy_rows.csv`
- `evaluation/results/phase9/experiment1_manual_review_20.json`
- `evaluation/results/phase9/experiment2_longitudinal_consistency.json`
- `evaluation/results/phase9/experiment2_longitudinal_consistency_rows.csv`
- `evaluation/results/phase9/experiment3_conflict_detection.json`
- `evaluation/results/phase9/experiment3_conflict_detection_rows.csv`
- `evaluation/results/phase9/phase9_summary.json`
- `evaluation/results/phase9/phase9_results_table.csv`

## Recommended End-to-End Run Order

Use this order for a clean setup:
1. Phase 1: generate + normalize + validate raw patients.
2. Phase 2: run extraction to create processed patient files.
3. Phase 3: build and validate graphs.
4. Phase 4: build patient memory stores.
5. Phase 5: build/update medical KB.
6. Phase 6: run agent for qualitative checks.
7. Phase 7: generate and train conflict classifier.
8. Phase 8: run evolution and conflict tests.
9. Phase 9: run full evaluation and archive artifacts.

## Quick Troubleshooting Notes

- If you hit provider rate limits, set `PHASE9_LLM_PROVIDER=openrouter`.
- If Torch/ONNX runtime errors appear, evaluation may use fallback modes; check `mode` and `runtime_mode` fields in Phase 9 outputs.
- If Chroma telemetry warnings appear, Phase 4 and baseline code already suppresses telemetry noise in this repo.

## Document Scope

This report reflects the current implementation and scripts present in this repository as of the latest Phase 9 integration state.