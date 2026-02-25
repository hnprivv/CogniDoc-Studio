# CogniDoc Studio 📄  
**AI-Powered Document Intelligence & PDF Production**

CogniDoc Studio is a Streamlit-based application that transforms raw, unstructured content into clean, intelligent, publication-ready PDFs. It combines classic document processing with modern AI techniques—analysis, formatting, and retrieval—into a single, cohesive workflow.

This is not a “click and convert” toy. It’s a document intelligence pipeline with a PDF as the final deliverable.

---

## What CogniDoc Studio Does

At a high level, CogniDoc Studio takes messy human input and turns it into a structured, analyzable, and professionally formatted document.

**Input → Intelligence → Output**

---

## Core Features

### 1. Multi-Source Content Ingestion
- Manual text input  
- File uploads: TXT, DOCX, PDF  
- Live speech-to-text transcription  

All inputs are unified into a single editable workspace.

### 2. Document Intelligence
- Readability scoring (Flesch–Kincaid)
- Estimated reading time
- Word & character counts
- Top themes / keywords
- Sentiment analysis

### 3. AI-Powered Formatting
- Local LLM (llama3.2 via Ollama)
- Markdown structuring with headings and bullets
- Chunked processing for speed and stability

### 4. Chat With Your Document (RAG)
- Vector embeddings + Chroma
- Answers strictly grounded in document content
- Explicit refusal when information is missing

### 5. Controlled PDF Generation
- Hierarchical headings
- Quotes and bullet lists
- Font selection and sizing
- Deterministic rendering via FPDF

---

## Tech Stack

- Streamlit  
- LangChain  
- Ollama (llama3.2)  
- HuggingFace Embeddings  
- Chroma  
- TextBlob  
- FPDF  
- PyPDF  
- python-docx  

---

## Project Structure

```
.
├── home.py
├── converter.py
├── utils.py
├── assets/
│   └── styles.css
```

---

## 🏃 Quick Start & Environment Setup

### Prerequisites
1. Install [Ollama](https://ollama.com/).
2. Pull the model: `ollama pull llama3.2`.
3. Have Python 3.10+ ready.

### 1. Setting Up the Virtual Environment (.venv)
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

**On Windows:**
```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

```

**On macOS/Linux:**

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

```

### 2. Installation & Launch

Once your `.venv` is active:

```bash
# Clone the repository
git clone [https://github.com/hnprivv/CogniDoc-Studio.git](https://github.com/hnprivv/CogniDoc-Studio.git)
cd CogniDoc-Studio

# Install dependencies
pip install -r requirements.txt

# Launch the studio
streamlit run home.py

```

## Design Philosophy

- Local-first AI
- Human-in-the-loop control
- Deterministic outputs
- Truthful, document-grounded AI

---

_Input chaos → structured intelligence → polished PDF._
