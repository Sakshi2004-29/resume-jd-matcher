"""
Resume-JD Matcher RAG engine.
Gemini classifies evidence; Python calculates score and status counts consistently.
"""
import os
import re
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "GEMINI_API_KEY.env")
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Gemini API key not found. Add GEMINI_API_KEY to GEMINI_API_KEY.env in the project root.")

genai.configure(api_key=API_KEY)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.EphemeralClient()
LLM_MODEL = "gemini-3.5-flash-lite"
_gemini_model = genai.GenerativeModel(LLM_MODEL)


def call_llm(prompt: str, max_tokens: int = 1800) -> str:
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


def parse_json_response(raw: str) -> Dict[str, Any]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise ValueError("Gemini returned an invalid structured response. Please retry.")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Gemini returned an unexpected response format. Please retry.")
    return data


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def build_collection(resume_chunks: List[str]):
    name = f"resume_{uuid.uuid4().hex[:8]}"
    collection = _chroma_client.create_collection(name=name)
    if resume_chunks:
        embeddings = _embedder.encode(resume_chunks).tolist()
        collection.add(ids=[f"chunk_{i}" for i in range(len(resume_chunks))],
                       embeddings=embeddings, documents=resume_chunks)
    return collection, name


def retrieve_relevant_chunks(collection, query: str, top_k: int = 5) -> List[str]:
    if collection.count() == 0:
        return []
    embedding = _embedder.encode([query]).tolist()
    result = collection.query(query_embeddings=embedding, n_results=min(top_k, collection.count()))
    docs = result.get("documents") or []
    return docs[0] if docs else []


def cleanup_collection(name: str):
    try:
        _chroma_client.delete_collection(name=name)
    except Exception:
        pass


def _requirement_key(text: str) -> str:
    """Normalize requirements to help remove exact and near-duplicate wording."""
    text = text.casefold()
    text = re.sub(r"\b(use|using|experience with|knowledge of|familiarity with|ability to|proficiency in|required skills|fundamentals)\b", " ", text)
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _deduplicate_requirements(items: List[str]) -> List[str]:
    """Remove exact duplicates and obvious overlapping requirement phrasings."""
    output = []
    keys = []
    for item in items:
        item = re.sub(r"\s+", " ", item).strip(" -•\t")
        if not item:
            continue
        key = _requirement_key(item)
        if not key:
            continue
        duplicate = False
        for existing_key in keys:
            # Exact normalized match, or one short phrasing is effectively contained in another.
            if key == existing_key:
                duplicate = True
                break
            shorter, longer = sorted((key, existing_key), key=len)
            if len(shorter) >= 9 and shorter in longer and len(shorter) / max(1, len(longer)) >= 0.68:
                duplicate = True
                break
        if not duplicate:
            output.append(item)
            keys.append(key)
    return output


def extract_jd_requirements(jd_text: str) -> List[str]:
    prompt = f"""
Extract distinct, concrete job requirements from this job description.
Include important skills, tools, and responsibilities. Avoid duplicates and generic filler.
Do not invent requirements. Return JSON only: {{"requirements":["requirement one"]}}
JOB DESCRIPTION:
{jd_text}
"""
    data = parse_json_response(call_llm(prompt, 1200))
    raw = data.get("requirements", [])
    if not isinstance(raw, list):
        raise ValueError("Gemini did not return a valid list of job requirements.")
    cleaned = []
    for item in raw:
        if isinstance(item, str):
            item = re.sub(r"^[-*•\d.\s]+", "", item).strip()
            if item:
                cleaned.append(item)
    return _deduplicate_requirements(cleaned)


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

        evidence_map = {req: retrieve_relevant_chunks(collection, req, 5) for req in requirements}
        evidence_block = "\n\n".join(
            f"REQUIREMENT: {req}\nRESUME EXCERPTS:\n" +
            ("\n".join(f"- {chunk}" for chunk in chunks) if chunks else "- No relevant excerpt retrieved.")
            for req, chunks in evidence_map.items()
        )

        prompt = f"""
Assess each requirement using ONLY the resume excerpts provided. Do not invent experience.
Distinguish evidence types:
- MATCH: direct practical evidence in a project, internship, work experience, or detailed accomplishment.
- PARTIAL: the skill is only listed in a skills section/certificate, or evidence is related but incomplete.
- NOT_DEMONSTRATED: no clear evidence in the supplied resume excerpts.
A certificate may support training/familiarity, but must NOT by itself prove hands-on work with a cloud platform.
A skill listed in a skills section may be PARTIAL if the JD asks for demonstrated application. If the JD only asks for familiarity or basic knowledge, a clear skills listing may be MATCH.
Do not penalize a candidate for not demonstrating a skill beyond what the JD asks for.
Return JSON only:
{{
 "results":[
  {{"requirement":"exact requirement text","status":"MATCH","evidence":"short factual quote/paraphrase from excerpts","suggestion":"specific truthful improvement, or empty string"}}
 ],
 "overall_suggestions":["suggestion one","suggestion two","suggestion three"]
}}
Return exactly one result per requirement, using its exact text. Allowed statuses: MATCH, PARTIAL, NOT_DEMONSTRATED.
If evidence is absent, say "No clear evidence found in the retrieved resume text." Do not calculate a score.

JOB DESCRIPTION:
{jd_text}

REQUIREMENTS AND RESUME EXCERPTS:
{evidence_block}
"""
        data = parse_json_response(call_llm(prompt, 3500))
        raw_results = data.get("results", [])
        by_req = {}
        for item in raw_results if isinstance(raw_results, list) else []:
            if isinstance(item, dict) and isinstance(item.get("requirement"), str):
                by_req[item["requirement"].strip().casefold()] = item

        results = []
        for req in requirements:
            item = by_req.get(req.strip().casefold(), {})
            status = str(item.get("status", "NOT_DEMONSTRATED")).upper().replace("-", "_").replace(" ", "_")
            if status not in {"MATCH", "PARTIAL", "NOT_DEMONSTRATED"}:
                status = "NOT_DEMONSTRATED"
            evidence = re.sub(r"\s+", " ", str(item.get("evidence", "")).strip())
            suggestion = re.sub(r"\s+", " ", str(item.get("suggestion", "")).strip())
            if not evidence:
                evidence = "No clear evidence found in the retrieved resume text."
            results.append({"requirement": req, "status": status, "evidence": evidence, "suggestion": suggestion})

        matched = sum(r["status"] == "MATCH" for r in results)
        partial = sum(r["status"] == "PARTIAL" for r in results)
        gaps = sum(r["status"] == "NOT_DEMONSTRATED" for r in results)
        total = len(results)
        score = round((matched + 0.5 * partial) / total * 100) if total else 0

        lines = [
            "## Match Score", f"**{score}/100**",
            f"Calculated from {total} unique requirements: {matched} MATCH, {partial} PARTIAL, and {gaps} NOT_DEMONSTRATED. MATCH earns full points, PARTIAL earns half points, and NOT_DEMONSTRATED earns zero. All requirements have equal weight.",
            "", "## Matched Requirements"
        ]
        for r in results:
            if r["status"] == "MATCH":
                lines.append(f"- **{r['requirement']} (MATCH):** {r['evidence']}")
        if matched == 0:
            lines.append("- No requirements were fully matched based on the retrieved evidence.")
        partials = [r for r in results if r["status"] == "PARTIAL"]
        lines += ["", "### Partially Matched Requirements"]
        if partials:
            for r in partials:
                lines.append(f"- **{r['requirement']} (PARTIAL):** {r['evidence']}")
        else:
            lines.append("- None.")
        lines += ["", "## Gaps"]
        gaps_list = [r for r in results if r["status"] == "NOT_DEMONSTRATED"]
        if gaps_list:
            for r in gaps_list:
                lines.append(f"- **{r['requirement']} (NOT_DEMONSTRATED):** {r['evidence']}")
        else:
            lines.append("- No gaps in the retrieved resume evidence were identified.")

        lines += ["", "## Suggestions"]
        suggestions = data.get("overall_suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = []
        suggestions = [str(s).strip() for s in suggestions if str(s).strip()]
        suggestions.extend(r["suggestion"] for r in results if r["status"] != "MATCH" and r["suggestion"])
        unique = []
        seen = set()
        for suggestion in suggestions:
            key = suggestion.casefold()
            if key not in seen:
                unique.append(suggestion)
                seen.add(key)
        if not unique:
            unique = [
                "Tailor project descriptions to the job requirements using accurate, specific evidence.",
                "Include measurable outcomes only where you have verified results.",
                "Only add skills and tools you have genuinely used."
            ]
        lines.extend(f"- {s}" for s in unique[:5])

        return {
            "requirements_analyzed": total,
            "analysis_markdown": "\n".join(lines),
            "match_score": score,
            "status_counts": {"matched": matched, "partial": partial, "gaps": gaps},
            "requirement_results": results,
        }
    finally:
        cleanup_collection(collection_name)
