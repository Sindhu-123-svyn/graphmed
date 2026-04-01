"""Verify all setup components are working."""

import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python 3.10+ required (found {version.major}.{version.minor})")
        return False

def check_packages():
    """Check required packages."""
    required = [
        'langchain', 'langgraph', 'chromadb', 'sentence_transformers',
        'transformers', 'peft', 'torch', 'pandas', 'numpy',
        'networkx', 'streamlit', 'groq'
    ]
    
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - not installed")
            missing.append(pkg)
    
    return len(missing) == 0

def check_directories():
    """Check project directories."""
    required_dirs = [
        'data/patients', 'data/mimic_raw', 'data/synthetic',
        'models', 'evaluation/results', 'src'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - missing")
            all_exist = False
    
    return all_exist

def check_spacy_models():
    """Check spaCy models."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy en_core_web_sm")
        
        # Check SciSpacy
        nlp_sci = spacy.load("en_core_sci_md")
        print("✅ SciSpacy en_core_sci_md")
        return True
    except Exception as e:
        print(f"❌ SpaCy models: {e}")
        return False

def main():
    """Run all checks."""
    print("\n" + "="*50)
    print("GraphMed Setup Verification")
    print("="*50 + "\n")
    
    checks = [
        ("Python Version", check_python_version()),
        ("Required Packages", check_packages()),
        ("Project Directories", check_directories()),
        ("SpaCy Models", check_spacy_models()),
    ]
    
    print("\n" + "-"*50)
    print("Summary")
    print("-"*50)
    
    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All setup checks passed! Ready to start Phase 1.")
    else:
        print("\n⚠️  Some checks failed. Review the output above.")
    
    return all_passed

if __name__ == "__main__":
    main()