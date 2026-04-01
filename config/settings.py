"""Configuration settings for GraphMed."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PATIENT_DIR = DATA_DIR / "patients"
MIMIC_DIR = DATA_DIR / "mimic_raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MEDICAL_KB_DIR = DATA_DIR / "medical_kb"
MODELS_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"

# Create directories if they don't exist
for dir_path in [PATIENT_DIR, MIMIC_DIR, SYNTHETIC_DIR, 
                 MEDICAL_KB_DIR, MODELS_DIR, EVAL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys (from .env file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Model configurations
LLM_MODEL = "llama-3.1-8b-instant"  # Groq model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CONFLICT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased"

# Vector store settings
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "patient_memories"

# Agent settings
MAX_AGENT_ITERATIONS = 10
AGENT_TEMPERATURE = 0.1

# Graph settings
CONFIDENCE_DECAY_RATE = 0.05  # 5% per month
MIN_CONFIDENCE_THRESHOLD = 0.2

# Evaluation
RANDOM_SEED = 42
TEST_PATIENTS_COUNT = 20

# Print configuration status
print(f"✓ Base directory: {BASE_DIR}")
print(f"✓ Data directory: {DATA_DIR}")
print(f"✓ Groq API key loaded: {GROQ_API_KEY is not None}")