"""
Fix ChromaDB telemetry warnings by patching the telemetry client
Run this once before using ChromaDB
"""

import chromadb
from chromadb.config import Settings
import warnings
import logging

# Suppress all ChromaDB telemetry
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.ERROR)
logging.getLogger("chromadb.auth").setLevel(logging.ERROR)
logging.getLogger("chromadb.rate_limiting").setLevel(logging.ERROR)
logging.getLogger("chromadb.quota").setLevel(logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("✅ ChromaDB warning suppression configured")

# Test that ChromaDB still works
try:
    client = chromadb.PersistentClient(
        path="data/chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    print("✅ ChromaDB client initialized successfully")
except Exception as e:
    print(f"Error: {e}")