"""Simple test using requests library."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_groq_simple():
    """Test Groq API with direct HTTP request."""
    
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ No API key found")
        return False
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": "Say 'GraphMed ready!'"}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"✅ Groq API working!")
            print(f"Response: {message}")
            return True
        else:
            print(f"❌ API error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Groq API (Direct HTTP)")
    print("="*50 + "\n")
    test_groq_simple()