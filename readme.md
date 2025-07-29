# LLM Document QA WebApp

A full-stack, dockerized document Question-Answering solution using **FastAPI** (backend, `api.py`) and **Streamlit** (frontend, `app.py`) powered by LangChain and the latest LLMs. This tool enables users to upload PDF documents and interact conversationally to query over their contents, with support for mul

Uploading video model streamlit.mp4…



Uploading Untitled video - Made with Clipchamp (1).mp4…



Uploading 20250729-0715-01.4091036.mp4…

tiple vectorstores and LLM providers.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start: Dockerized Deployment](#quick-start-dockerized-deployment)
- [FastAPI Backend (`api.py`)](#fastapi-backend-apipy)
  - [Endpoints](#endpoints)
  - [Usage](#usage)
- [Streamlit Frontend (`app.py`)](#streamlit-frontend-apppy)
- [Environment Variables & API Keys](#environment-variables--api-keys)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)
- [License](#license)

## Project Structure

```
.
├── api.py                # FastAPI backend (REST API)
├── app.py                # Streamlit frontend (UI)
├── requirements.txt      # Python dependencies for both
├── docker-compose.yml    # Orchestrates backend & frontend
├── Dockerfile.api
├── Dockerfile.app
```

## Features

- **FastAPI REST API** for document upload, chunking, vectorstore loading, and LLM-powered answers
- **Streamlit UI** for uploading PDFs, selecting vector DB/LLM, & chatting
- **LLM Providers:** OpenAI, Ollama, Groq, Gemini
- **Vectorstores:** FAISS, ChromaDB, Weaviate
- **Retrieval-Augmented Generation:** Contextual answers with sources from your PDFs
- **Completely Dockerized:** No manual environment set-up needed

## Requirements

- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/)
- API keys for your preferred LLMs (see [API Keys](#environment-variables--api-keys))

## Quick Start: Dockerized Deployment

1. **Clone this repository:**
   ```sh
   git clone 
   cd 
   ```

2. **Build and run the project using Docker Compose:**
   ```sh
   docker-compose up --build
   ```

   This will:
   - Start the FastAPI backend at [http://localhost:8000](http://localhost:8000)
   - Start the Streamlit frontend at [http://localhost:8501](http://localhost:8501)

3. **Access the Applications:**
   - **Frontend UI:** [http://localhost:8501](http://localhost:8501)
   - **Backend Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## FastAPI Backend (`api.py`)

### Overview

- **Purpose:** Provides REST endpoints to process uploaded PDFs, embed text, store in vector databases, and return smart answers to user queries.
- **Main Endpoints:**
  - `/main`: Accepts PDF files & user prompts, returns LLM-generated answers with document context.
  - `/upload_docs`: Simple endpoint for just uploading and storing PDFs.
  - `/docs_metadata`: Returns document chunk/info metadata and can perform similarity search.

### Endpoints

| Endpoint         | Method | Description                                                                |
|------------------|--------|----------------------------------------------------------------------------|
| `/main`          | POST   | Accepts model, API key, files, vectorstore name, and user prompt. Returns answers and references based on document context. |
| `/upload_docs`   | POST   | Uploads and saves PDF files to a timestamped folder.                        |
| `/docs_metadata` | POST   | Returns file metadata, chunk info, and performs similarity search if a prompt is provided. |

#### Sample Request to `/main` (via `curl`):

```sh
curl -X POST "http://localhost:8000/main" \
  -F 'model=Open AI' \
  -F 'model_name=gpt-3.5-turbo' \
  -F 'api_key=YOUR_OPENAI_KEY' \
  -F 'files=@/path/to/yourfile.pdf' \
  -F 'vector_store=FAISS' \
  -F 'user_prompt=Summarize the main points'
```

#### Output
- JSON with answer and context_docs.

## Streamlit Frontend (`app.py`)

### Overview

- **Purpose:** User-facing web application for uploading documents, selecting LLM/vectorstore, entering queries, and seeing answers + matched document snippets.
- **How it Works:**
  - Upload PDFs through the UI
  - Choose LLM provider (OpenAI, Ollama, Groq, Gemini) and model, provide key if needed
  - Pick vectorstore (FAISS, ChromaDB, Weaviate)
  - Ask questions and get answers/citations interactively

### Key Features

- **Live Model List Fetching:** For Ollama, Groq, Gemini models—shows only available options based on your API key.
- **No Code Needed:** All interaction via a simple web UI.
- **Document Similarity View:** Shows chunks of PDF most relevant to your question.

### Local (Non-Docker) Development

1. **Set up a Python virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate   # or .\venv\Scripts\activate on Windows
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Start backend:**
   ```sh
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

4. **Start frontend:**
   ```sh
   streamlit run app.py
   ```

## Environment Variables & API Keys

Depending on which LLM or vectorstore you select, you may need to provide appropriate API keys in the web app (sidebar inputs) or as part of the API call:

- **OpenAI**: You need an OpenAI API key
- **Ollama**: Requires Ollama models running locally ([Ollama docs](https://ollama.com/))
- **Groq**: Groq API key
- **Gemini**: Gemini API key
- **Weaviate**: For Weaviate, ensure a local or remote instance is running

**Keep your API keys secure!**

## Troubleshooting

- Make sure ports **8000** (backend) and **8501** (frontend) are free.
- Rebuild containers after you change `requirements.txt`:
  ```sh
  docker-compose up --build
  ```
- Have LLM/vectorstore backends (like Ollama, Weaviate) running locally if you choose them.
- For errors with large PDFs, consider increasing memory limits or breaking the PDF down.

## Credits

- [LangChain](https://github.com/langchain-ai/langchain)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [Ollama](https://ollama.com/)
- [Groq](https://groq.com/)
- [Google Gemini](https://ai.google.dev/)

## License

MIT (or specify as appropriate)

**Enjoy querying your PDFs with the power of modern LLMs – all from your browser!**

