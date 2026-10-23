# Resume ↔ Job Description Matcher

An AI-powered Resume–Job Description Matcher built with Streamlit, FastAPI, Gemini, and Supabase. It compares a resume with a job description, evaluates matching skills and experience, identifies gaps, and suggests improvements.

## Features

- Upload a resume and provide a job description.
- Analyze resume relevance using AI.
- Display a match score and identify matched and missing requirements.
- View analysis results and visual summaries.
- Export analysis results as a PDF.
- Sign in and keep analysis history associated with the user's account through Supabase.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| LLM | Google Gemini |
| Embeddings | Sentence Transformers |
| Vector store | ChromaDB |
| Authentication and history | Supabase |
| Document parsing | PyPDF and python-docx |
| Containerization | Docker and Docker Compose |

## Project Structure

```text
resume-jd-matcher/
├── backend/
│   ├── main.py
│   ├── rag_engine.py
│   ├── document_parser.py
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   └── Dockerfile
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create the required local environment files and configure your credentials privately.

The frontend uses these environment variables:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `BACKEND_URL`

The backend uses `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

**Never commit `.env`, `GEMINI_API_KEY.env`, API keys, or Supabase credentials to GitHub.** Configure secrets through environment variables or your hosting provider's secret settings.

### 4. Start the backend

From the project root:

```powershell
uvicorn backend.main:app --reload --port 8000
```

If your local setup expects the backend directory as the working directory, use the corresponding command documented in the code instead.

### 5. Start the frontend

Open a second terminal, activate the same virtual environment, and run:

```powershell
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

## Run with Docker Compose

After Docker Desktop is installed and your local environment variables are configured, run:

```powershell
docker compose up --build
```

The configured ports are:

- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`
- Backend API documentation: `http://localhost:8000/docs`

## Deployment

The application is being prepared for separate deployment of its FastAPI backend and Streamlit frontend. Configure Gemini and Supabase credentials as secrets in the respective hosting services. Update the frontend's `BACKEND_URL` to point to the deployed backend.

## Security Notes

- Do not upload `.env` or `GEMINI_API_KEY.env`.
- Do not place API keys directly in source code.
- Use private environment variables for backend secrets.
- Restrict access to your GitHub repository as appropriate.

## Known Limitations

- The vector store is temporary, so analysis data stored in ChromaDB is not persistent across requests.
- Analysis quality depends on the uploaded documents and the LLM response.
- Public deployment requires suitable authentication, access controls, and API usage limits.

## Project Purpose

This project demonstrates an end-to-end AI application combining document parsing, semantic retrieval, LLM-based analysis, a REST API, a web interface, and deployment configuration.
