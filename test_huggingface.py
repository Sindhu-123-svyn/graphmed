"""Test HuggingFace API connection."""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()

def test_huggingface_connection():
    """Test HuggingFace connection."""
    
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY not found in .env file")
        return False
    
    try:
        # Login
        login(token=api_key, add_to_git_credential=False)
        
        # Test API
        api = HfApi()
        models = api.list_models(filter="biomedical", limit=5)
        
        print(f"✅ HuggingFace API test successful!")
        print(f"Found {len(list(models))} biomedical models")
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace API test failed: {e}")
        return False

if __name__ == "__main__":
    test_huggingface_connection()