<!-- Readme.md -->
# Alice in Wonderland RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Ollama that allows users to ask questions about "Alice in Wonderland" and get contextually relevant answers based on the book's content.

## Features

- **Interactive Chat Interface**: Clean Streamlit-based chat UI
- **RAG Pipeline**: Uses retrieval-augmented generation for accurate, context-based responses
- **Local LLM**: Powered by Ollama's Llama3 model for privacy and offline usage
- **Context Display**: Shows retrieved document chunks for transparency
- **Chat History**: Maintains conversation history during the session

## Prerequisites

### System Requirements
- Windows 10/11 with WSL (Windows Subsystem for Linux)
- Python 3.8 or higher
- At least 8GB RAM (recommended for running Ollama models)

### WSL Setup
If you don't have WSL installed:
```powershell
# Run in PowerShell as Administrator
wsl --install
### Ollama Installation

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull the required models:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```


## Installation

### 1. Navigate to Project Directory

Open PowerShell and navigate to your project:

```powershell
PS E:\Programming_Stuff\DeepLearnng\Rag\Rag_md>
```

### 2. Switch to WSL

```powershell
wsl
```

You should now see your WSL prompt:
```bash
(base) shibinsha@GENIE-X3gen:/mnt/e/Programming_Stuff/DeepLearnng/Rag/Rag_md$

```plaintext

### 3. Create Virtual Environment (if not exists)
\`\`\`bash
python -m venv env
```

### 4. Activate Virtual Environment

```bash
source env/bin/activate

```plaintext

You should see the environment activated:
\`\`\`bash
(env) (base) shibinsha@GENIE-X3gen:/mnt/e/Programming_Stuff/DeepLearnng/Rag/Rag_md$
```

### 5. Install Dependencies

```bash
pip install streamlit langchain langchain-community langchain-ollama langchain-text-splitters

```plaintext

### 6. Prepare Data Directory
Create the required directory structure and add your data file:
\`\`\`bash
mkdir -p data/books
```

Place your `alice_in_wonderland.md` file in the `data/books/` directory.

## Running the Application

### Important: Use the Correct Command

❌ **DON'T run with python directly:**
```bash
python query_data.py

```plaintext
This will show warnings and won't work properly:
```

WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext!
Warning: to view this Streamlit app on a browser, run it with the following command:
streamlit run query_data.py [ARGUMENTS]
Session state does not function when running a script without `streamlit run`

```plaintext

✅ **DO run with streamlit:**
\`\`\`bash
streamlit run query_data.py
```

### Expected Output

When you run the correct command, you should see:
```bash
(env) (base) shibinsha@GENIE-X3gen:/mnt/e/Programming_Stuff/DeepLearnng/Rag/Rag_md$ streamlit run query_data.py

You can now view your Streamlit app in your browser.

Local URL: [http://localhost:8501](http://localhost:8501)
Network URL: [http://172.26.120.221:8501](http://172.26.120.221:8501)

gio: [http://localhost:8501](http://localhost:8501): Operation not supported

```plaintext

### Accessing the Application

Since the browser won't open automatically in WSL, manually open your browser and navigate to:
- **http://localhost:8501**

## Complete Workflow

Here's the complete step-by-step workflow you should follow each time:

```powershell
# 1. Open PowerShell and navigate to project
PS E:\Programming_Stuff\DeepLearnng\Rag\Rag_md> wsl

# 2. You're now in WSL
(base) shibinsha@GENIE-X3gen:/mnt/e/Programming_Stuff/DeepLearnng/Rag/Rag_md$ source env/bin/activate

# 3. Virtual environment is activated
(env) (base) shibinsha@GENIE-X3gen:/mnt/e/Programming_Stuff/DeepLearnng/Rag/Rag_md$ streamlit run query_data.py

# 4. Open browser manually to http://localhost:8501
```

## Project Structure

```plaintext
Rag_md/
├── env/                          # Virtual environment
├── data/
│   └── books/
│       └── alice_in_wonderland.md # Your markdown data file
├── query_data.py                 # Main Streamlit application
└── README.md                     # This documentation
```

## Expected Warnings (Can Be Ignored)

### 1. LangChain Deprecation Warning

```plaintext
LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0.
```

**Status**: Can be ignored - the application still works perfectly.

**Optional Fix**: Update the import in `query_data.py`:

```python
# Replace this line:
from langchain_community.llms import Ollama

# With:
from langchain_ollama import OllamaLLM

# And change:
llm = Ollama(model="llama3")

# To:
llm = OllamaLLM(model="llama3")
```

### 2. Browser Opening Warning

```plaintext
gio: http://localhost:8501: Operation not supported
```

**Status**: Normal in WSL - just open the browser manually.

## Troubleshooting

### Common Issues

#### 1. "Data file not found" Error

**Problem**:

```plaintext
Error: Data file not found at ./data/books/alice_in_wonderland.md
```

**Solution**: Ensure the file exists at the correct path:
```bash
ls -la data/books/alice_in_wonderland.md

```plaintext

#### 2. Ollama Connection Issues
**Problem**: Application can't connect to Ollama
**Solution**: 
\`\`\`bash
# Check if Ollama is running
ollama serve

# In another terminal, verify models are available
ollama list
```

#### 3. Virtual Environment Issues

**Problem**: Commands not found or wrong Python version
**Solution**: Make sure you're in the activated environment:
```bash

# Check if environment is activated (should show (env) in prompt)

which python

# Should show: /mnt/e/Programming_Stuff/DeepLearnng/Rag/Rag_md/env/bin/python

```plaintext

#### 4. Port Already in Use
**Problem**: 
```

OSError: [Errno 98] Address already in use

```plaintext
**Solution**: 
\`\`\`bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use a different port
streamlit run query_data.py --server.port 8502
```

## Configuration

### Customizing RAG Parameters

Edit these values in `query_data.py`:

```python
# Text chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Adjust based on your content
    chunk_overlap=200     # Overlap between chunks
)

# Retrieval
retriever = vector_store.as_retriever(k=5)  # Number of chunks to retrieve

# Models
llm = Ollama(model="llama3")                 # Change LLM model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")  # Change embedding model
```

## Performance Tips

1. **First Run**: Initial startup may take longer as it processes the document
2. **Memory**: Ensure sufficient RAM for Ollama models (8GB+ recommended)
3. **Caching**: The app uses `@st.cache_resource` to avoid reprocessing on each run


## Dependencies

Create a `requirements.txt` file with:

```plaintext
streamlit
langchain
langchain-community
langchain-ollama
langchain-text-splitters
```

Install all at once:
```bash
pip install -r requirements.txt

```plaintext

## Quick Reference

### Daily Usage Commands
\`\`\`bash
# Navigate and start
cd /mnt/e/Programming_Stuff/DeepLearnng/Rag/Rag_md
source env/bin/activate
streamlit run query_data.py

# Open browser to: http://localhost:8501
```

### Stopping the Application

- Press `Ctrl+C` in the terminal
- Or close the terminal window


## Support

If you encounter issues:

1. Ensure you're using `streamlit run` not `python`
2. Check that WSL and virtual environment are properly activated
3. Verify Ollama is running and models are installed
4. Manually open browser to localhost:8501


## License

[Add your license information here]

```plaintext

This README.md file is now ready to be saved in your project directory. It provides comprehensive documentation that matches your exact workflow and addresses all the specific issues you encounter when running the application.

<Actions>
<Action name="Create requirements.txt" description="Generate a requirements.txt file with exact package versions" />
<Action name="Create startup script" description="Create a batch/shell script to automate the WSL and environment setup" />
<Action name="Add sample data" description="Include a sample alice_in_wonderland.md file for testing" />
<Action name="Fix deprecation warnings" description="Update query_data.py to use the newer LangChain imports" />
<Action name="Add configuration file" description="Create a config.yaml for easier customization of RAG parameters" />
</Actions>
```