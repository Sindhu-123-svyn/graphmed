"""Quick verification of essential packages."""

import sys

def check_essential():
    """Check only essential packages."""
    
    print("Checking essential packages...\n")
    
    essentials = {
        'groq': 'Groq API',
        'langchain_groq': 'LangChain Groq',
        'pandas': 'Data processing',
        'numpy': 'Numerical operations',
        'dotenv': 'Environment variables'
    }
    
    all_ok = True
    for package, desc in essentials.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {desc} ({package})")
        except ImportError:
            print(f"❌ {desc} ({package}) - NOT INSTALLED")
            all_ok = False
    
    # Check spaCy separately
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print(f"✅ spaCy model (en_core_web_sm)")
    except:
        print(f"❌ spaCy model missing")
        all_ok = False
    
    return all_ok

if __name__ == "__main__":
    if check_essential():
        print("\n✅ Essential packages OK! You can proceed.")
    else:
        print("\n⚠️  Missing essential packages. Run install_all.bat")