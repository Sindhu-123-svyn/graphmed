@echo off
echo Installing GraphMed Dependencies...
echo.

echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Installing core packages...
pip install chromadb==0.4.22
pip install sentence-transformers==2.2.2
pip install transformers==4.40.0
pip install peft==0.10.0
pip install torch==2.1.0
pip install pandas==2.2.0
pip install networkx==3.2.1
pip install streamlit==1.31.0

echo.
echo Installing LangChain packages...
pip install langchain==0.2.0
pip install langgraph==0.0.20
pip install langchain-groq==0.1.0

echo.
echo Installing data science packages...
pip install scikit-learn==1.4.0
pip install matplotlib==3.8.2
pip install seaborn==0.13.2

echo.
echo Installing evaluation packages...
pip install evaluate==0.4.1
pip install bert-score==0.3.13
pip install rouge-score==0.1.2

echo.
echo Installing utilities...
pip install python-dotenv==1.0.0
pip install tqdm==4.66.1
pip install pyyaml==6.0.1
pip install requests==2.31.0
pip install pyvis==0.3.2

echo.
echo Installing API clients...
pip install groq==0.5.0
pip install huggingface-hub==0.22.0
pip install google-generativeai==0.5.0
pip install together==1.2.0
pip install cohere==5.5.0

echo.
echo Installing NLP packages...
pip install nltk==3.8.1
pip install spacy==3.7.2
pip install scispacy==0.5.4

echo.
echo Downloading spaCy models...
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

echo.
echo Installation complete!
pause