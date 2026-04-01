"""Test Groq API connection."""

import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def test_groq_connection():
    """Test if Groq API is working."""
    
    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file")
        return False
    
    try:
        # Initialize client
        client = Groq(api_key=api_key)
        
        # Make a simple test call
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say: GraphMed setup successful!"}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        # Print response
        result = response.choices[0].message.content
        print(f"✅ Groq API test successful!")
        print(f"Response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False

if __name__ == "__main__":
    test_groq_connection()