"""
Core RAG logic for Resume-JD Matcher.
Gemini classifies each requirement; Python calculates the final score and status counts.
"""
import os
import re
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# API key file remains in the inner project root, one level above backend/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "GEMINI_API_KEY.env")
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(
        "Gemini API key not found. Add GEMINI_API_KEY to GEMINI_API_KEY.env in the project root."
    )

genai.configure(api_key=API_KEY)
# Use Gemini's hosted embedding model instead of loading a local PyTorch model.
# This reduces local RAM use on small hosting instances.
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"
_chroma_client = chromadb.EphemeralClient()


def generate_embeddings(texts: List[str], task_type: str) -> List[List[float]]:
    """Generate hosted Gemini embeddings for a list of text strings."""
    embeddings = []
    for text in texts:
        response = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type=task_type,
        )
        embedding = response.get("embedding")
        if not embedding:
            raise ValueError("Gemini did not return an embedding. Check API access and quota.")
        embeddings.append(embedding)
    return embeddings
LLM_MODEL = "gemini-3.5-flash-lite"
_gemini_model = genai.GenerativeModel(LLM_MODEL)


def call_llm(prompt: str, max_tokens: int = 1500) -> str:
    response = _gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response. Check API access and model availability.")
    return response.text


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError("Gemini returned an invalid structured response. Please retry.")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Gemini returned an unexpected response format. Please retry.")
    return value


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def build_collection(resume_chunks: List[str]):
    collection_name = f"resume_{uuid.uuid4().hex[:8]}"
    collection = _chroma_client.create_collection(name=collection_name)
    if resume_chunks:
        embeddings = generate_embeddings(resume_chunks, "retrieval_document")
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(resume_chunks))],
            embeddings=embeddings,
            documents=resume_chunks,
        )
    return collection, collection_name


def retrieve_relevant_chunks(collection, query: str, top_k: int = 4) -> List[str]:
    if collection.count() == 0:
        return []
    query_embedding = generate_embeddings([query], "retrieval_query")
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
    )
    documents = results.get("documents") or []
    return documents[0] if documents else []


def cleanup_collection(collection_name: str):
    try:
        _chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass


def extract_jd_requirements(jd_text: str) -> List[str]:
    prompt = f"""
Extract distinct, concrete requirements from this job description.
Include skills, tools, and meaningful responsibilities. Avoid duplicates and vague items.
Do not invent requirements. Return JSON only:
{{"requirements": ["requirement one", "requirement two"]}}

JOB DESCRIPTION:
{jd_text}
"""
    data = _parse_json(call_llm(prompt, max_tokens=1200))
    raw_requirements = data.get("requirements", [])
    if not isinstance(raw_requirements, list):
        raise ValueError("Gemini did not return a valid list of job requirements.")
    requirements = []
    seen = set()
    for item in raw_requirements:
        if not isinstance(item, str):
            continue
        item = re.sub(r"^[-*•\d.\s]+", "", item).strip()
        key = re.sub(r"\W+", " ", item.lower()).strip()
        if item and key and key not in seen:
            requirements.append(item)
            seen.add(key)
    return requirements


def analyze_match(resume_text: str, jd_text: str) -> Dict:
    if not resume_text or not resume_text.strip():
        raise ValueError("No readable text was extracted from the resume.")
    if not jd_text or not jd_text.strip():
        raise ValueError("Please provide a job description.")

    resume_chunks = chunk_text(resume_text)
    if not resume_chunks:
        raise ValueError("No readable text was extracted from the resume.")

    collection, collection_name = build_collection(resume_chunks)
    try:
        requirements = extract_jd_requirements(jd_text)
        if not requirements:
            raise ValueError("Could not extract requirements from the job description.")

        evidence_map = {
            requirement: retrieve_relevant_chunks(collection, requirement, top_k=4)
            for requirement in requirements
        }

        evidence_block = "\n\n".join(
            f"REQUIREMENT: {requirement}\nRESUME EVIDENCE:\n"
            + ("\n".join(f"- {chunk}" for chunk in chunks)
               if chunks else "No relevant resume chunk retrieved.")
            for requirement, chunks in evidence_map.items()
        )

        prompt = f"""
Evaluate each requirement against ONLY the resume evidence below.
A missing retrieved chunk means the skill is not demonstrated by the evidence supplied; it does not prove the candidate lacks the skill.
Do not infer skills from adjacent skills, course titles, or a GitHub link alone.
Use exactly one status:
MATCH = explicit, convincing evidence;
PARTIAL = related or incomplete evidence;
NOT_DEMONSTRATED = no adequate evidence in the supplied resume excerpts.
Return JSON only in this format:
{{
  "results": [
    {{"requirement": "exact requirement text", "status": "MATCH", "evidence": "brief quote or accurate paraphrase", "suggestion": "truthful improvement suggestion"}}
  ],
  "overall_suggestions": ["suggestion one", "suggestion two", "suggestion three"]
}}
Return exactly one result for every requirement listed below, preserve the requirement text exactly, and do not fabricate evidence.
For no evidence, write: "No clear evidence found in the retrieved resume text."
Do not calculate a score; Python will calculate it from these statuses.

JOB DESCRIPTION:
{jd_text}

REQUIREMENTS AND RETRIEVED RESUME EVIDENCE:
{evidence_block}
"""
        parsed = _parse_json(call_llm(prompt, max_tokens=3500))
        raw_results = parsed.get("results", [])
        by_requirement = {}
        for item in raw_results:
            if isinstance(item, dict) and isinstance(item.get("requirement"), str):
                by_requirement[item["requirement"].strip().casefold()] = item

        results = []
        for requirement in requirements:
            item = by_requirement.get(requirement.strip().casefold(), {})
            status = str(item.get("status", "NOT_DEMONSTRATED")).upper().replace("-", "_").replace(" ", "_")
            if status not in {"MATCH", "PARTIAL", "NOT_DEMONSTRATED"}:
                status = "NOT_DEMONSTRATED"
            evidence = str(item.get("evidence", "")).strip()
            suggestion = str(item.get("suggestion", "")).strip()
            if not evidence:
                evidence = "No clear evidence found in the retrieved resume text."
            results.append({
                "requirement": requirement,
                "status": status,
                "evidence": evidence,
                "suggestion": suggestion,
            })

        matched = sum(r["status"] == "MATCH" for r in results)
        partial = sum(r["status"] == "PARTIAL" for r in results)
        gaps = sum(r["status"] == "NOT_DEMONSTRATED" for r in results)
        total = len(results)
        score = round((matched + partial * 0.5) / total * 100) if total else 0

        lines = [
            "## Match Score",
            f"**{score}/100**",
            f"Calculated from {total} unique requirements: {matched} MATCH, {partial} PARTIAL, and {gaps} NOT_DEMONSTRATED. MATCH earns full points, PARTIAL earns half points, and NOT_DEMONSTRATED earns zero. All requirements have equal weight.",
            "",
            "## Matched Requirements",
        ]
        fully_matched = [r for r in results if r["status"] == "MATCH"]
        partly_matched = [r for r in results if r["status"] == "PARTIAL"]
        missing = [r for r in results if r["status"] == "NOT_DEMONSTRATED"]

        if fully_matched:
            for r in fully_matched:
                lines.append(f"- **{r['requirement']} (MATCH):** {r['evidence']}")
        else:
            lines.append("- No requirements were fully matched based on the retrieved evidence.")

        if partly_matched:
            lines += ["", "### Partially Matched Requirements"]
            for r in partly_matched:
                lines.append(f"- **{r['requirement']} (PARTIAL):** {r['evidence']}")

        lines += ["", "## Gaps"]
        if missing:
            for r in missing:
                lines.append(f"- **{r['requirement']} (NOT_DEMONSTRATED):** {r['evidence']}")
        else:
            lines.append("- No gaps in the retrieved resume evidence were identified.")

        lines += ["", "## Suggestions"]
        suggestions = parsed.get("overall_suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = []
        suggestions = [str(s).strip() for s in suggestions if str(s).strip()]
        suggestions += [
            r["suggestion"] for r in results
            if r["status"] != "MATCH" and r["suggestion"]
        ]
        unique_suggestions = []
        seen_suggestions = set()
        for suggestion in suggestions:
            key = suggestion.casefold()
            if key not in seen_suggestions:
                unique_suggestions.append(suggestion)
                seen_suggestions.add(key)
        if not unique_suggestions:
            unique_suggestions = [
                "Tailor project descriptions to the job requirements using accurate, specific evidence.",
                "Include measurable outcomes where you have verified results.",
                "Only add skills and tools you have genuinely used.",
            ]
        for suggestion in unique_suggestions[:5]:
            lines.append(f"- {suggestion}")

        return {
            "requirements_analyzed": total,
            "analysis_markdown": "\n".join(lines),
            "match_score": score,
            "status_counts": {
                "matched": matched,
                "partial": partial,
                "gaps": gaps,
            },
            "requirement_results": results,
        }
    finally:
        cleanup_collection(collection_name)
