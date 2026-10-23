"""
main.py
-------
FastAPI backend for the Resume-JD Matcher.

Endpoints:
  POST /analyze  -> upload resume file + JD text (or file), get match analysis
  GET  /health   -> simple health check
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from document_parser import parse_document, clean_text
from rag_engine import analyze_match

app = FastAPI(title="Resume-JD Matcher API", version="1.0.0")

# Allow the Streamlit frontend (running on a different port/container) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    resume_file: UploadFile = File(...),
    jd_text: str = Form(None),
    jd_file: UploadFile = File(None),
):
    """
    Accepts a resume file (pdf/docx/txt) and a job description,
    either as pasted text (jd_text) or as an uploaded file (jd_file).
    Returns a match score, gap analysis, and improvement suggestions.
    """
    if not jd_text and not jd_file:
        raise HTTPException(
            status_code=400,
            detail="Provide either jd_text or jd_file.",
        )

    try:
        resume_bytes = await resume_file.read()
        resume_text = clean_text(parse_document(resume_file.filename, resume_bytes))

        if jd_file:
            jd_bytes = await jd_file.read()
            jd_text_final = clean_text(parse_document(jd_file.filename, jd_bytes))
        else:
            jd_text_final = clean_text(jd_text)

        if not resume_text.strip():
            raise HTTPException(status_code=422, detail="Resume appears to be empty or unreadable.")
        if not jd_text_final.strip():
            raise HTTPException(status_code=422, detail="Job description appears to be empty.")

        result = analyze_match(resume_text, jd_text_final)
        return result

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # Keep the message actionable for local development; quota/model errors
        # should be visible instead of appearing as an unexplained frontend timeout.
        message = str(e)
        if "429" in message or "quota" in message.lower() or "rate limit" in message.lower():
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota/rate limit reached. Check https://ai.dev/rate-limit and retry when quota is available."
            )
        raise HTTPException(status_code=500, detail=f"Internal error: {message}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
