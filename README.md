# RAG Document QA System

An intelligent cloud-hosted application that enables contextual question answering over uploaded PDF documents using Retrieval-Augmented Generation (RAG). Built with Python and LangChain, and powered by AstraDB vector search and OpenAI models.

---
## 📌 Features

- 📄 PDF Upload & Processing – Automatically chunk and embed documents
- 🧠 Retrieval-Augmented Generation (RAG) – Context-aware AI responses
- 🔍 Metadata-Based Filtering – Document-level retrieval isolation
- 💬 Conversational QA – Maintain chat history
- ☁ Cloud Deployment – Hosted on Hugging Face Spaces
- 🐳 Docker Support – Containerized application
- 📊 Structured Logging – Centralized error tracking
- 🔐 Secure API Handling – Environment-based configuration

---

## 🛠 Tech Stack

**Backend:** Python  
**Framework:** LangChain  
**Vector Database:** AstraDB  
**LLM & Embeddings:** OpenAI API  
**UI Layer:** Gradio  
**Containerization:** Docker  
**Logging:** Python Logging Module  

---

## 🏗 System Architecture

### High-Level Flow

User  
↓  
Gradio Web Interface (Hugging Face Space)  
↓  
Backend Processing Layer  
↓  
OpenAI Embeddings + LLM  
↓  
AstraDB Vector Store  

---

## 🚀 Getting Started (Local Setup)

### Prerequisites

- Python 3.10+
- OpenAI API Key
- AstraDB Account
- Internet connection

---

### Clone Repository

```bash
git clone https://github.com/dhruv8315/document-qa.git
cd document-qa
```

---

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Configure Environment Variables
```bash
cd document-qa/src
```

Create a `.env` file:

```
OPENAI_API=your_openai_key
ASTRA_DB_API_ENDPOINT=your_endpoint
ASTRA_DB_APPLICATION_TOKEN=your_token
```

---

### Run Application

```bash
python app.py
```

Access locally at:

```
http://localhost:7860
```

---

### API Keys Setup

#### 1️⃣ OpenAI API Key
1. Visit https://platform.openai.com/
2. Create or log into your account
3. Go to **API Keys**
4. Generate a new secret key

#### 2️⃣ AstraDB Credentials
1. Create an account at https://www.datastax.com/astra
2. Create a new database
3. Navigate to **Database Settings**
4. Copy:
   - API Endpoint
   - Application Token


#### Set Locally

Add to your `.env` file:

## 📱 Usage

1. **Launch the Webapp** on http://localhost:7860
2. **Select PDF document** which you like to analyze
3. **Upload the document**
4. **Ask any questions** you like to know about the document
5. **Tap "Clear"** to upload a new document

## 🐳 Docker Support

### Build Image

```bash
docker build -t document-qa .
```

### Run Container

```bash
docker run -p 7860:7860 --env-file src/.env document-qa
```

## 🔐 Security Considerations

- API keys stored as environment variables
- `.env` excluded via `.gitignore`
- Logs excluded from version control
- No hardcoded credentials in source code
- Metadata filtering prevents cross-document leakage

## 📊 Logging & Error Handling

- Centralized logger configuration
- File + console logging
- Structured try/except exception handling
- Debug-friendly error tracing

---