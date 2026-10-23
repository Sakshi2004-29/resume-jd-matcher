"""Streamlit frontend for Resume-JD Matcher with Supabase authentication and persistent history."""

import io
import os
import re
import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as PDFImage

# .env is in the project root, one level above frontend/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

st.set_page_config(page_title="Resume-JD Matcher", page_icon="📄", layout="wide")

st.markdown("""
<style>
.block-container {max-width:1180px;padding-top:1.8rem;padding-bottom:3rem}
.hero {padding:1.35rem 1.5rem;border-radius:16px;background:linear-gradient(120deg,#172554,#1e40af);color:white;margin-bottom:1rem}
.hero h1 {margin:0;font-size:2rem}.hero p {margin:.4rem 0 0;color:#dbeafe}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25);padding:12px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

if "supabase_session" not in st.session_state:
    st.session_state.supabase_session = None
if "user" not in st.session_state:
    st.session_state.user = None
if "latest_analysis" not in st.session_state:
    st.session_state.latest_analysis = None


def get_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def apply_auth_session(client):
    """Restore the Supabase access token for authenticated database requests."""
    session = st.session_state.get("supabase_session")
    if session:
        try:
            client.auth.set_session(session["access_token"], session["refresh_token"])
        except Exception:
            # A stale session can be cleared by logging out and signing in again.
            st.session_state.supabase_session = None
            st.session_state.user = None
            return False
    return bool(st.session_state.get("user"))


def safe_pdf_text(value):
    return html.escape(str(value or ""), quote=False).replace("\n", "<br/>")


def markdown_line_to_pdf(value):
    """Convert common Markdown bold markers to ReportLab bold tags safely."""
    escaped = safe_pdf_text(value)
    # Convert Markdown bold safely. Avoid variable-width look-behind, which
    # Python's regular-expression engine rejects.
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    # Remove remaining single-asterisk emphasis markers without regex look-behind.
    escaped = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", escaped)
    return escaped


def parse_sections(markdown_text):
    sections, title, lines = [], "Overview", []
    for raw in (markdown_text or "").splitlines():
        line = raw.strip()
        heading = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading:
            if lines:
                sections.append((title, lines))
            title, lines = heading.group(1).strip().strip("#* "), []
        elif line:
            lines.append(line)
    if lines:
        sections.append((title, lines))
    return sections


def extract_score(report):
    match = re.search(r"(?<!\d)(\d{1,3})\s*/\s*100", report or "")
    if not match:
        match = re.search(r"(?i)(?:match score|score)[^\d]{0,25}(\d{1,3})\s*%?", report or "")
    return min(100, max(0, int(match.group(1)))) if match else None


def count_statuses(markdown_text):
    counts = {"Matched": 0, "Partial": 0, "Gaps": 0}
    for line in (markdown_text or "").splitlines():
        line = line.strip()
        if not re.match(r"^[-*•]\s+", line):
            continue
        upper = line.upper()
        if "(NOT_DEMONSTRATED)" in upper or "(NOT DEMONSTRATED)" in upper:
            counts["Gaps"] += 1
        elif "(PARTIAL)" in upper:
            counts["Partial"] += 1
        elif "(MATCH)" in upper:
            counts["Matched"] += 1
    return counts


def build_chart_images(score, counts):
    """Return PNG chart bytes for the UI and PDF report."""
    chart_bytes = {}
    categories = ["Matched", "Partially matched", "Not matched"]
    values = [int(counts.get("Matched", 0)), int(counts.get("Partial", 0)), int(counts.get("Gaps", 0))]

    # Score chart: label each segment directly to avoid a legend/axis-label collision.
    fig, ax = plt.subplots(figsize=(6.4, 2.35))
    safe_score = max(0, min(100, int(score))) if score is not None else 0
    remaining = 100 - safe_score
    ax.barh(["Match score"], [safe_score], color="#2878B5")
    ax.barh(["Match score"], [remaining], left=[safe_score], color="#F28E2B")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score out of 100", labelpad=7)
    ax.set_title(f"Overall Resume–JD Match: {safe_score}/100" if score is not None else "Overall Resume–JD Match", pad=12)
    if safe_score >= 12:
        ax.text(safe_score / 2, 0, f"Score: {safe_score}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    if remaining >= 12:
        ax.text(safe_score + remaining / 2, 0, f"Remaining: {remaining}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.8)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=170, bbox_inches="tight"); plt.close(fig)
    chart_bytes["score"] = buf.getvalue()

    # Requirement status chart: horizontal bars keep long category names readable.
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.barh(categories[::-1], values[::-1])
    ax.set_title("Requirement Status Breakdown")
    ax.set_xlabel("Number of requirements")
    ax.set_xlim(0, max(values + [1]) * 1.25)
    ax.bar_label(bars, padding=4)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.4)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=180, bbox_inches="tight"); plt.close(fig)
    chart_bytes["statuses"] = buf.getvalue()
    return chart_bytes


def count_requirement_results(requirement_results):
    counts = {"Matched": 0, "Partial": 0, "Gaps": 0}
    for item in requirement_results or []:
        status = str(item.get("status", "")).upper()
        if status == "MATCH":
            counts["Matched"] += 1
        elif status == "PARTIAL":
            counts["Partial"] += 1
        elif status in {"NOT_DEMONSTRATED", "NOT DEMONSTRATED", "GAP", "NOT_MATCHED"}:
            counts["Gaps"] += 1
    return counts



def suggested_action(item):
    """Return a useful, status-aware fallback instead of repeating generic advice."""
    suggestion = str(item.get("suggestion") or "").strip()
    generic_phrases = {
        "keep evidence accurate and specific",
        "add specific evidence if accurate",
        "add truthful evidence if you have it",
        "provide more details",
    }
    if suggestion and suggestion.lower().rstrip(".") not in generic_phrases:
        return suggestion

    requirement = str(item.get("requirement") or "this requirement").strip()
    status = str(item.get("status") or "").upper()
    if status == "MATCH":
        return "No change required. Keep this evidence accurate and make sure it reflects your actual experience."
    if status == "PARTIAL":
        return f"Add a specific, truthful project or work example showing how you used {requirement.lower()}."
    return f"If you have this experience, add a truthful example showing {requirement.lower()}; otherwise, leave the gap clearly identified."


def make_pdf(title, timestamp, score, report, resume_name="", requirement_results=None):
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=.62*inch,
                            leftMargin=.62*inch, topMargin=.62*inch, bottomMargin=.65*inch,
                            title="Resume-Job Description Match Report")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=19, leading=23, alignment=TA_CENTER,
                              textColor=colors.HexColor("#17365D"), spaceAfter=5))
    styles.add(ParagraphStyle("Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=13, leading=16, textColor=colors.HexColor("#17365D"),
                              spaceBefore=11, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle("BodyReport", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=9, leading=12.5, spaceAfter=5, splitLongWords=True))
    styles.add(ParagraphStyle("Meta", parent=styles["BodyText"], fontSize=8.5, leading=11,
                              textColor=colors.HexColor("#475569")))
    story = [
        Paragraph("RESUME ↔ JOB DESCRIPTION", styles["ReportTitle"]),
        Paragraph("MATCH ANALYSIS REPORT", styles["Section"]),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceAfter=10),
    ]
    meta_rows = [
        [Paragraph("<b>Job title</b>", styles["Meta"]), Paragraph(safe_pdf_text(title or "Untitled role"), styles["BodyReport"])],
        [Paragraph("<b>Generated</b>", styles["Meta"]), Paragraph(safe_pdf_text(timestamp), styles["BodyReport"])],
        [Paragraph("<b>Resume file</b>", styles["Meta"]), Paragraph(safe_pdf_text(resume_name), styles["BodyReport"])],
    ]
    meta = Table(meta_rows, colWidths=[1.15*inch, 5.35*inch], hAlign="LEFT")
    meta.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#F3F6FA")),
                              ("BOX",(0,0),(-1,-1),.5,colors.HexColor("#D7DEE8")),
                              ("INNERGRID",(0,0),(-1,-1),.25,colors.HexColor("#E1E6ED")),
                              ("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),8),
                              ("RIGHTPADDING",(0,0),(-1,-1),8), ("TOPPADDING",(0,0),(-1,-1),6),
                              ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story += [meta, Spacer(1, 12)]
    score_text = f"{score}/100" if score is not None else "N/A"
    story += [Paragraph(f"<b>Overall match score:</b> {safe_pdf_text(score_text)}", styles["BodyReport"]),
              Spacer(1, 8)]
    requirement_results = requirement_results or []
    counts_for_pdf = count_requirement_results(requirement_results)
    charts_for_pdf = build_chart_images(score, counts_for_pdf)
    story.append(Paragraph("Visual summary", styles["Section"]))
    for chart_key in ("score", "statuses"):
        story.append(PDFImage(io.BytesIO(charts_for_pdf[chart_key]), width=6.1*inch, height=2.65*inch if chart_key == "score" else 3.15*inch))
        story.append(Spacer(1, 5))
    groups = [
        ("Matched skills / requirements", "MATCH"),
        ("Partially matched skills", "PARTIAL"),
        ("Gaps — not demonstrated in resume", "NOT_DEMONSTRATED"),
    ]
    for heading, status in groups:
        story.append(Paragraph(safe_pdf_text(heading), styles["Section"]))
        items = [r for r in requirement_results if r.get("status") == status]
        if not items:
            story.append(Paragraph("None identified.", styles["BodyReport"]))
            continue
        rows = [[Paragraph("<b>Requirement</b>", styles["BodyReport"]),
                 Paragraph("<b>Evidence in resume</b>", styles["BodyReport"]),
                 Paragraph("<b>Suggested action</b>", styles["BodyReport"])]]
        for item in items:
            rows.append([Paragraph(safe_pdf_text(item.get("requirement")), styles["BodyReport"]),
                         Paragraph(safe_pdf_text(item.get("evidence")), styles["BodyReport"]),
                         Paragraph(safe_pdf_text(suggested_action(item)), styles["BodyReport"])])
        table = Table(rows, colWidths=[1.65*inch, 2.55*inch, 2.3*inch], repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCE7F5")),
                                   ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#CBD5E1")),
                                   ("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),5),
                                   ("RIGHTPADDING",(0,0),(-1,-1),5), ("TOPPADDING",(0,0),(-1,-1),5),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.extend([table, Spacer(1, 6)])
    story.append(Paragraph("Full analysis", styles["Section"]))
    for section_title, lines in parse_sections(report):
        story.append(Paragraph(markdown_line_to_pdf(section_title), styles["Section"]))
        for line in lines:
            cleaned = re.sub(r"^[-*•]\s+|^\d+[.)]\s+", "", line).strip()
            story.append(Paragraph("• " + markdown_line_to_pdf(cleaned), styles["BodyReport"]))
    def footer(canvas, page_doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
        canvas.line(.62*inch, .48*inch, letter[0]-.62*inch, .48*inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(.62*inch, .3*inch, "Resume-JD Matcher • Evidence-based estimate")
        canvas.drawRightString(letter[0]-.62*inch, .3*inch, f"Page {page_doc.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    output.seek(0)
    return output.getvalue()


def load_history(client, user_id):
    response = (client.table("analysis_history").select("*")
                .eq("user_id", user_id).order("created_at", desc=True).limit(50).execute())
    return response.data or []


def save_history(client, user_id, record):
    payload = {
        "user_id": user_id,
        "job_title": record["job_title"],
        "resume_name": record["resume_name"],
        "match_score": record["score"],
        "requirements_analyzed": record["requirements_analyzed"],
        "analysis_markdown": record["analysis_markdown"],
        "requirement_results": record.get("requirement_results", []),
    }
    client.table("analysis_history").insert(payload).execute()


client = get_client()
if client is None:
    st.error("Supabase configuration is missing. Check SUPABASE_URL and SUPABASE_KEY in the project-root .env file.")
    st.stop()

# Sidebar authentication and persistent history
with st.sidebar:
    st.header("Account")
    if not st.session_state.user:
        auth_mode = st.radio("Choose action", ["Sign in", "Create account"])
        with st.form("auth_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", help="Use at least 6 characters.")
            auth_submit = st.form_submit_button(auth_mode, use_container_width=True)
        if auth_submit:
            if not email.strip() or not password:
                st.error("Enter both email and password.")
            else:
                try:
                    if auth_mode == "Create account":
                        result = client.auth.sign_up({"email": email.strip(), "password": password})
                        if result.session:
                            st.session_state.supabase_session = {
                                "access_token": result.session.access_token,
                                "refresh_token": result.session.refresh_token,
                            }
                            st.session_state.user = {"id": result.user.id, "email": result.user.email}
                            st.success("Account created and signed in.")
                            st.rerun()
                        else:
                            st.success("Account created. Check your email to confirm it, then sign in.")
                    else:
                        result = client.auth.sign_in_with_password({"email": email.strip(), "password": password})
                        st.session_state.supabase_session = {
                            "access_token": result.session.access_token,
                            "refresh_token": result.session.refresh_token,
                        }
                        st.session_state.user = {"id": result.user.id, "email": result.user.email}
                        st.rerun()
                except Exception as exc:
                    st.error(f"Authentication failed: {exc}")
    else:
        st.success(f"Signed in as {st.session_state.user.get('email', 'user')}")
        if st.button("Log out", use_container_width=True):
            try:
                client.auth.sign_out()
            except Exception:
                pass
            st.session_state.user = None
            st.session_state.supabase_session = None
            st.session_state.latest_analysis = None
            st.rerun()

    if st.session_state.user:
        apply_auth_session(client)
        st.divider()
        st.header("Saved analysis history")
        try:
            history = load_history(client, st.session_state.user["id"])
            if not history:
                st.caption("Your saved analyses will appear here after your first analysis.")
            for row in history:
                label = f"{row.get('created_at', '')[:10]} · {row.get('job_title') or 'Job analysis'}"
                if st.button(label[:55], key=f"history_{row['id']}", use_container_width=True):
                    st.session_state.latest_analysis = {
                        "id": row["id"],
                        "job_title": row.get("job_title") or "Job match analysis",
                        "timestamp": (row.get("created_at") or "")[:16].replace("T", " "),
                        "score": row.get("match_score"),
                        "requirements_analyzed": row.get("requirements_analyzed") or 0,
                        "analysis_markdown": row.get("analysis_markdown") or "",
                        "resume_name": row.get("resume_name") or "",
                        "requirement_results": row.get("requirement_results") or [],
                    }
                    st.rerun()
                if st.button("Delete", key=f"delete_{row['id']}"):
                    client.table("analysis_history").delete().eq("id", row["id"]).eq("user_id", st.session_state.user["id"]).execute()
                    st.rerun()
        except Exception as exc:
            st.error(f"Could not load history: {exc}")

st.markdown("""
<div class="hero">
<h1>📄 Resume ↔ Job Description Matcher</h1>
<p>Evidence-based resume analysis powered by semantic retrieval and Gemini.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.user:
    st.info("Please sign in or create an account in the sidebar. Your analysis history will be saved to your account.")
    st.stop()

with st.form("analysis_form", clear_on_submit=False):
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("1 · Your resume")
        resume_file = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], key="resume_file")
    with right:
        st.subheader("2 · Job description")
        jd_mode = st.radio("Provide job description by", ["Paste text", "Upload file"], horizontal=True)
        jd_text, jd_file = "", None
        if jd_mode == "Paste text":
            jd_text = st.text_area("Paste the job description", height=220, placeholder="Paste the complete job description here…")
        else:
            jd_file = st.file_uploader("Upload JD (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], key="jd_file")
    job_title = st.text_input("Job title (optional)", placeholder="e.g., Data Analyst Intern")
    submitted = st.form_submit_button("🔍 Analyze match", type="primary", use_container_width=True)

if submitted:
    if not resume_file:
        st.error("Please upload your resume.")
    elif jd_mode == "Paste text" and not jd_text.strip():
        st.error("Please paste a job description.")
    elif jd_mode == "Upload file" and not jd_file:
        st.error("Please upload a job description file.")
    else:
        with st.spinner("Analyzing resume evidence and job requirements…"):
            try:
                files = {"resume_file": (resume_file.name, resume_file.getvalue())}
                data = {}
                if jd_mode == "Paste text":
                    data["jd_text"] = jd_text
                else:
                    files["jd_file"] = (jd_file.name, jd_file.getvalue())
                response = requests.post(f"{BACKEND_URL}/analyze", files=files, data=data, timeout=(10, 240))
                if not response.ok:
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    st.error(f"Backend returned HTTP {response.status_code}: {detail}")
                else:
                    result = response.json()
                    report = result.get("analysis_markdown", "No report was returned.")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    score = result.get("match_score", extract_score(report))
                    record = {
                        "job_title": job_title.strip() or "Job match analysis",
                        "timestamp": timestamp,
                        "score": score,
                        "requirements_analyzed": result.get("requirements_analyzed", 0),
                        "analysis_markdown": report,
                        "resume_name": resume_file.name,
                        "requirement_results": result.get("requirement_results", []),
                    }
                    try:
                        apply_auth_session(client)
                        save_history(client, st.session_state.user["id"], record)
                        st.success("Analysis completed and saved to your account history.")
                    except Exception as exc:
                        st.error(f"Analysis completed, but saving to Supabase failed: {exc}")
                    st.session_state.latest_analysis = record
            except requests.exceptions.Timeout:
                st.error("Analysis took too long. Check the backend terminal for Gemini quota/model errors.")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not reach the backend at {BACKEND_URL}. Start FastAPI and try again.")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

selected = st.session_state.get("latest_analysis")
if selected:
    st.divider()
    st.subheader("Analysis report")
    c1, c2, c3 = st.columns(3)
    c1.metric("Match score", f"{selected['score']}/100" if selected.get("score") is not None else "See report")
    c2.metric("Requirements analyzed", selected.get("requirements_analyzed", 0))
    c3.metric("Resume file", (selected.get("resume_name") or "")[:26])
    if selected.get("score") is not None:
        st.progress(max(0.0, min(1.0, selected["score"] / 100)), text=f"Calculated match score: {selected['score']} / 100")

    requirement_results = selected.get("requirement_results") or []
    counts = count_requirement_results(requirement_results)
    total_classified = sum(counts.values())

    st.markdown("### Visual match summary")
    chart_images = build_chart_images(selected.get("score"), counts)
    # Render charts in separate rows to prevent clipping/overlap on narrow screens.
    st.image(chart_images["score"], width=700)
    st.image(chart_images["statuses"], width=700)

    stat1, stat2, stat3 = st.columns(3)
    stat1.metric("Matched", counts["Matched"])
    stat2.metric("Partially matched", counts["Partial"])
    stat3.metric("Not matched / not demonstrated", counts["Gaps"])
    if total_classified == 0:
        st.warning("No structured requirement statuses were returned for this analysis. The score chart is shown, but status counts need requirement_results from the backend.")
    else:
        st.caption(f"Status counts cover {total_classified} classified requirements. These counts are separate from the overall match score.")

    if requirement_results:
        st.markdown("### Skills and requirements by status")
        matched_rows = [{"Skill / requirement": r.get("requirement", ""), "Evidence from resume": r.get("evidence", "")} for r in requirement_results if r.get("status") == "MATCH"]
        partial_rows = [{"Skill / requirement": r.get("requirement", ""), "Evidence from resume": r.get("evidence", ""), "What to improve": suggested_action(r)} for r in requirement_results if r.get("status") == "PARTIAL"]
        gap_rows = [{"Missing / not demonstrated": r.get("requirement", ""), "What the resume shows": r.get("evidence", ""), "Recommended action": suggested_action(r)} for r in requirement_results if r.get("status") == "NOT_DEMONSTRATED"]
        with st.expander(f"Matched ({len(matched_rows)})", expanded=True):
            if matched_rows:
                st.dataframe(pd.DataFrame(matched_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No fully matched requirements.")
        with st.expander(f"Partially matched ({len(partial_rows)})", expanded=True):
            if partial_rows:
                st.dataframe(pd.DataFrame(partial_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No partially matched requirements.")
        with st.expander(f"Gaps to work on ({len(gap_rows)})", expanded=True):
            if gap_rows:
                st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)
            else:
                st.success("No gaps identified in the retrieved resume evidence.")
    st.markdown("### Full analysis")
    st.markdown(selected.get("analysis_markdown", "No report available."))
    pdf = make_pdf(selected.get("job_title", ""), selected.get("timestamp", ""),
                   selected.get("score"), selected.get("analysis_markdown", ""),
                   selected.get("resume_name", ""), requirement_results)
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", selected.get("job_title", "")).strip("_") or "resume_match"
    st.download_button("⬇️ Download report as PDF", data=pdf,
                       file_name=f"{safe_title}_match_report.pdf",
                       mime="application/pdf", type="primary")
else:
    st.info("Upload a resume and job description, then select Analyze match to generate your report.")

st.divider()
st.caption("Built with FastAPI, Streamlit, ChromaDB, sentence-transformers, Supabase, and Gemini. Scores are evidence-based estimates, not hiring decisions.")
