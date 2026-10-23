# Resume ↔ Job Description Matcher (RAG-based)

A Retrieval-Augmented Generation (RAG) application that compares a resume against
a job description, scores the fit, identifies gaps, and suggests concrete resume
improvements — using semantic search + an LLM, not just keyword matching.

---

## Problem Statement

Manually checking whether a resume matches a job description is slow and
error-prone: keyword-matching tools miss semantic equivalence (e.g. "built REST
APIs" vs. "backend development experience"), while purely manual review doesn't
scale across many applications. This tool retrieves the *most relevant* resume
evidence for each JD requirement using embeddings, then asks an LLM to reason
over that evidence and produce a structured, actionable analysis.

---

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌────────────────────┐
│  Streamlit  │─────▶│   FastAPI         │─────▶│  Document Parser    │
│  Frontend   │      │   Backend         │      │  (pdf/docx/txt)     │
└─────────────┘      │  (/analyze)       │      └────────────────────┘
                      │                  │
                      │        │         │
                      │        ▼         │
                      │  ┌──────────────┐│      ┌────────────────────┐
                      │  │ Chunk resume ││─────▶│  sentence-           │
                      │  │ into segments││      │  transformers        │
                      │  └──────────────┘│      │  (embeddings)        │
                      │        │         │      └────────────────────┘
                      │        ▼         │               │
                      │  ┌──────────────┐│               ▼
                      │  │  ChromaDB    ││◀──────────────┘
                      │  │  (vector DB) ││
                      │  └──────────────┘│
                      │        │         │
                      │        ▼         │
                      │  Extract JD      │
                      │  requirements    │──────▶ Claude (LLM)
                      │  via LLM         │
                      │        │         │
                      │        ▼         │
                      │  Retrieve top-k  │
                      │  resume chunks   │
                      │  per requirement │
                      │        │         │
                      │        ▼         │
                      │  Final scoring + │──────▶ Claude (LLM)
                      │  gap analysis    │
                      └──────────────────┘
```

### Pipeline steps

1. **Parse** — extract raw text from uploaded resume/JD (PDF, DOCX, or TXT).
2. **Chunk** — split the resume into overlapping ~400-word segments.
3. **Embed** — encode chunks with a local `sentence-transformers` model
   (`all-MiniLM-L6-v2`) — free, runs offline, no API cost.
4. **Store** — index embeddings in an ephemeral ChromaDB collection.
5. **Extract requirements** — ask Claude to turn the JD into a clean list of
   discrete requirements (skills, tools, qualifications).
6. **Retrieve** — for each requirement, semantically retrieve the most
   relevant resume chunks (this is the "R" in RAG).
7. **Generate** — feed the requirement-by-requirement evidence to Claude and
   ask for a match score, matched/missing requirements, and improvement
   suggestions (this is the "G" in RAG).

---

## Tech Stack

| Component        | Technology                          | Reason |
|-------------------|--------------------------------------|--------|
| LLM               | Claude (Anthropic API)              | Reasoning over retrieved evidence |
| Embeddings        | sentence-transformers (local)       | Free, no API cost, fast |
| Vector store      | ChromaDB (ephemeral, in-memory)      | Zero setup, no external DB needed |
| Backend           | FastAPI                              | Clean REST API, async support |
| Frontend          | Streamlit                            | Fast to build, good demo UX |
| Containerization  | Docker + docker-compose              | Reproducible, portfolio-ready |

---

## Project Structure

```
resume-jd-matcher/
├── backend/
│   ├── main.py              # FastAPI app + /analyze endpoint
│   ├── rag_engine.py         # Chunking, embedding, retrieval, LLM calls
│   ├── document_parser.py    # PDF/DOCX/TXT text extraction
│   └── Dockerfile
├── frontend/
│   ├── app.py                # Streamlit UI
│   └── Dockerfile
├── sample_data/               # (optional) put test resumes/JDs here
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Run

### Option A — Local (no Docker)

```bash
# 1. Clone/copy the project, then from the project root:
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 2. Set your Anthropic API key
cp .env.example .env
# edit .env and paste your key

export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d '=' -f2)

# 3. Run the backend (from backend/ directory)
cd backend
uvicorn main:app --reload --port 8000

# 4. In a second terminal, run the frontend (from frontend/ directory)
cd frontend
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

### Option B — Docker (recommended for the resume-ready version)

```bash
cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY

docker compose up --build
```

- Frontend: `http://localhost:8501`
- Backend docs (Swagger): `http://localhost:8000/docs`

---

## API Reference

### `POST /analyze`

**Form-data fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `resume_file` | file | Yes | PDF/DOCX/TXT resume |
| `jd_text` | string | One of jd_text/jd_file | Pasted job description |
| `jd_file` | file | One of jd_text/jd_file | Uploaded JD file |

**Response:**
```json
{
  "requirements_analyzed": 12,
  "analysis_markdown": "## Match Score\n78/100\n\n## Matched Requirements\n..."
}
```

### `GET /health`
Returns `{"status": "ok"}` — used for container health checks.

---

## Known Limitations & Future Improvements

- **Ephemeral vector store**: each request creates and destroys its own
  ChromaDB collection — fine for single-user demo use, not for concurrent
  multi-user production traffic. A persistent Chroma/Postgres+pgvector
  setup would be the next step.
- **Single JD at a time**: could be extended to compare one resume against
  multiple JDs and rank them.
- **No authentication**: fine for a portfolio project; would need auth +
  rate limiting for public deployment.
- **LLM cost**: two LLM calls per analysis (requirement extraction + final
  scoring). Could be merged into one call to reduce latency/cost if needed.

---

## Why This Project (for resume/interview framing)

This project was built to demonstrate an end-to-end RAG pipeline — not just
calling an LLM API, but combining chunking, embeddings, semantic retrieval,
and structured LLM reasoning, wrapped in a real API + UI + Docker deployment.
It also solves a real problem the author faced personally: evaluating resume
fit against job descriptions during their own job search.

## STREAMLIT APP LINK:-
https://resume-jd-matcher-nsusw6hbvcssmcjagavlci.streamlit.app/
