import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import json

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)                               # expose rag/ package
sys.path.insert(0, os.path.join(ROOT, "src", "ml"))
from recommender import generate_recommendations
from agent.diagnosis import run_diagnosis_node, DiagnosisReport

# ── RAG retriever (Developer 3) ──────────────────────────────────────────────
try:
    from rag.retriever import retrieve_resources_for_gaps
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(ROOT, "src", "ml", "models")

# ── Label maps ───────────────────────────────────────────────────────────────
GRADE_MAP    = {0: "Grade 0", 1: "Grade 1", 2: "Grade 2",
                3: "Grade 3", 4: "Grade 4", 5: "Grade 5"}
CATEGORY_MAP = {0: "At-Risk",       1: "Below-Average", 2: "Average",
                3: "Above-Average", 4: "High-Performing", 5: "Exceptional"}

# severity → colour token used in st.markdown HTML
SEVERITY_COLOR = {
    "Critical": "#ef4444",   # red-500
    "Moderate": "#f97316",   # orange-500
    "Minor":    "#eab308",   # yellow-500
}
SEVERITY_BG = {
    "Critical": "#2d1117",
    "Moderate": "#2d1a0a",
    "Minor":    "#1f1c08",
}

STATUS_COLOR = {
    "At-Risk":         "#ef4444",
    "Below-Average":   "#f97316",
    "Average":         "#facc15",
    "Above-Average":   "#34d399",
    "High-Performing": "#22d3ee",
    "Exceptional":     "#a78bfa",
}
ALIGNMENT_COLOR = {
    "Misaligned":        "#ef4444",
    "Partially Aligned": "#f97316",
    "Aligned":           "#22c55e",
}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VectoSpace · Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Dark card shell ── */
.vs-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* ── Gap cards ── */
.gap-card {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
}
.gap-card h4 { margin: 0 0 .35rem 0; font-size: 1rem; font-weight: 600; }
.gap-card p  { margin: 0 0 .5rem 0; font-size: .875rem; opacity: .85; }
.gap-card ul { margin: 0; padding-left: 1.15rem; font-size: .85rem; }
.gap-card ul li { margin-bottom: .2rem; }

/* ── Pill badges ── */
.pill {
    display: inline-block;
    padding: .2rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .03em;
    margin-right: .4rem;
}

/* ── Metric grid ── */
.metric-box {
    background: #1f2937;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box .val { font-size: 1.8rem; font-weight: 700; }
.metric-box .lbl { font-size: .78rem; opacity: .6; margin-top: .15rem; }

/* ── Section divider ── */
.vs-divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 1.5rem 0;
}

/* ── Source badge ── */
.source-llm      { color: #a78bfa; }
.source-rule     { color: #60a5fa; }
.source-fallback { color: #f97316; }

/* ── Confidence bar ── */
.conf-track {
    background: #1f2937;
    border-radius: 999px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}

/* ── Resource cards (RAG) ── */
.resource-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: .75rem;
    transition: border-color .2s;
}
.resource-card:hover { border-color: #6366f1; }
.resource-card .rc-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: .5rem;
    margin-bottom: .45rem;
}
.resource-card .rc-title {
    font-size: .95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0;
}
.resource-card .rc-title a {
    color: #818cf8;
    text-decoration: none;
}
.resource-card .rc-title a:hover { text-decoration: underline; }
.resource-card .rc-summary {
    font-size: .83rem;
    color: #94a3b8;
    margin: .35rem 0 .5rem 0;
    line-height: 1.55;
}
.resource-card .rc-meta {
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
    align-items: center;
}
.rc-score-bar {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    font-size: .75rem;
    color: #64748b;
}
.rc-score-fill {
    display: inline-block;
    height: 5px;
    border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
}
.rc-gap-group {
    margin-bottom: 1.1rem;
}
.rc-gap-label {
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .05em;
    text-transform: uppercase;
    opacity: .5;
    margin-bottom: .4rem;
}
/* ── RAG status pill ── */
.rag-status {
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    font-size: .78rem;
    font-weight: 600;
    padding: .2rem .65rem;
    border-radius: 999px;
    border: 1px solid;
}
.rag-on  { color: #34d399; background: #05190f; border-color: #34d39955; }
.rag-off { color: #64748b; background: #1e293b; border-color: #33415555; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 VectoSpace")
    st.caption("Student Performance · Diagnosis + RAG Engine")
    st.markdown("---")

    # ── LLM toggle ────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Diagnosis Mode")
    use_llm = st.toggle(
        "Enable LLM Diagnosis",
        value=False,
        help="Requires GEMINI_API_KEY or OPENAI_API_KEY in environment. Falls back to rule-based when unavailable.",
    )
    if use_llm:
        st.info("🤖 LLM mode — Gemini / GPT-4o will enrich the diagnosis.", icon="✨")
    else:
        st.info("📐 Rule-based mode — fast, deterministic, no API key needed.", icon="⚡")

    st.markdown("---")

    # ── RAG Resource Retrieval toggle ─────────────────────────────────────────
    st.markdown("### 📚 Resource Retrieval (RAG)")
    if RAG_AVAILABLE:
        use_rag = st.toggle(
            "Enable RAG Resource Retrieval",
            value=True,
            help="Retrieves personalised learning resources for each detected gap using semantic search over a curated educational corpus.",
        )
        rag_use_llm_summary = st.toggle(
            "LLM-generated resource summaries",
            value=False,
            help="When enabled, uses Gemini/GPT-4o to write a tailored summary for each resource. Falls back to template if no API key is set.",
        )
        rag_top_k = st.slider(
            "Resources per gap",
            min_value=1, max_value=5, value=3,
            help="Number of resources retrieved per learning gap.",
        )
        if use_rag:
            st.markdown('<span class="rag-status rag-on">🟢 RAG Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="rag-status rag-off">⚪ RAG Disabled</span>', unsafe_allow_html=True)
    else:
        use_rag = False
        rag_use_llm_summary = False
        rag_top_k = 3
        st.warning("⚠️ RAG package not found. Run `pip install -r requirements.txt` to enable resource retrieval.", icon="📦")

    st.markdown("---")

    # ── Student goals input ───────────────────────────────────────────────────
    st.markdown("### 🎯 Student Learning Goals")
    st.caption("One goal per line. These are fed into the Diagnosis Node along with each student's data.")
    goals_raw = st.text_area(
        label="Goals (one per line)",
        value="Pass all subjects with a score ≥ 60\nAttend at least 80% of classes\nImprove study hours to 7+ per week",
        height=160,
        label_visibility="collapsed",
        key="goals_input",
    )
    student_goals = [g.strip() for g in goals_raw.splitlines() if g.strip()]
    st.markdown(f"`{len(student_goals)}` goal(s) configured")

    st.markdown("---")
    st.markdown("### 📂 About")
    st.markdown(
        "Upload a student CSV file to get **ML-powered grade predictions**, "
        "**learning gap diagnosis**, **RAG-powered resource retrieval**, "
        "and **personalised study recommendations**."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:1.5rem 0 1rem 0;">
  <h1 style="margin:0;font-size:2rem;font-weight:700;">
    🎓 Student Performance Predictor
    <span style="font-size:1rem;font-weight:400;opacity:.5;margin-left:.5rem;">+ Diagnosis · RAG Engine</span>
  </h1>
  <p style="opacity:.55;margin:.3rem 0 0 0;">
    Upload student data · Get grade predictions · Identify learning gaps · Retrieve learning resources · Generate personalised actions
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML model…")
def load_model():
    with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scale_cols.pkl"), "rb") as f:
        scale_cols = pickle.load(f)
    return model, scaler, scale_cols


def preprocess_raw_data(df, scaler, scale_cols):
    df = df.copy()
    if "student_id"  in df.columns: df.drop(columns=["student_id"],  inplace=True)
    if "final_grade" in df.columns: df.drop(columns=["final_grade"], inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    binary_map = {"yes": 1, "no": 0}
    for col in ["internet_access", "extra_activities"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    travel_map = {"<15 min": 0, "15-30 min": 1, "30-60 min": 2, ">60 min": 3}
    if "travel_time" in df.columns and df["travel_time"].dtype == "object":
        df["travel_time"] = df["travel_time"].map(travel_map).fillna(0).astype(int)

    edu_map = {"no formal": 0, "high school": 1, "diploma": 2,
               "graduate": 3, "post graduate": 4, "phd": 5}
    if "parent_education" in df.columns and df["parent_education"].dtype == "object":
        df["parent_education"] = df["parent_education"].map(edu_map).fillna(0).astype(int)

    nominal_cols = [c for c in ["gender", "school_type", "study_method"] if c in df.columns]
    if nominal_cols:
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)

    if scaler is not None and scale_cols:
        cols_to_scale = [c for c in scale_cols if c in df.columns]
        if cols_to_scale:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _status_pill(label: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}55;">{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# RAG RESOURCE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_ICON = {
    "Mathematics": "📐",
    "Science":     "🔬",
    "English":     "📖",
    "Attendance":  "🗓️",
    "Study Time":  "⏱️",
}

def _render_resources(resources: list[dict]):
    """Render RAG-retrieved resources grouped by learning gap."""
    if not resources:
        st.info("No resources retrieved — either no gaps were identified or RAG is disabled.", icon="📭")
        return

    # Group by gap area
    from collections import defaultdict
    grouped: dict = defaultdict(list)
    for r in resources:
        grouped[r.get("gap", "General")].append(r)

    for gap_area, items in grouped.items():
        sev      = items[0].get("severity", "Moderate")
        sev_col  = SEVERITY_COLOR.get(sev, "#94a3b8")
        area_icon = _DOMAIN_ICON.get(gap_area, "📚")

        st.markdown(
            f'<div class="rc-gap-label" style="color:{sev_col};">{area_icon} {gap_area} '
            f'<span style="opacity:.6;font-weight:400;">({sev})</span></div>',
            unsafe_allow_html=True,
        )

        for r in items:
            score     = float(r.get("score", 0))
            score_pct = min(int(score * 100), 100)   # raw cosine → %
            score_bar = (
                f'<span class="rc-score-bar">'  
                f'<span class="rc-score-fill" style="width:{score_pct}px;"></span>'
                f'{score:.3f}</span>'
            )

            # Severity pill
            sev_pill = (
                f'<span class="pill" style="background:{sev_col}22;color:{sev_col};'
                f'border:1px solid {sev_col}44;font-size:.7rem;">{sev}</span>'
            )

            title_html = (
                f'<a href="{r["url"]}" target="_blank" rel="noopener noreferrer">'
                f'{r["title"]}</a>'
            )

            st.markdown(
                f"""
                <div class="resource-card">
                  <div class="rc-header">
                    <p class="rc-title">{title_html}</p>
                    <div style="white-space:nowrap;">{sev_pill}</div>
                  </div>
                  <p class="rc-summary">{r['summary']}</p>
                  <div class="rc-meta">
                    {score_bar}
                    <span style="font-size:.73rem;color:#334155;">relevance score</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSIS + RESOURCES RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_diagnosis(report: DiagnosisReport, row_raw: pd.Series):
    """Render a full DiagnosisReport inside a rich Streamlit layout."""

    status_col  = STATUS_COLOR.get(report.overall_status, "#94a3b8")
    align_col   = ALIGNMENT_COLOR.get(report.goal_alignment, "#94a3b8")
    source_cls  = {"llm": "source-llm", "rule-based": "source-rule"}.get(report.source, "source-fallback")
    source_icon = {"llm": "✨ LLM-powered", "rule-based": "⚡ Rule-based", "llm-fallback": "⚠️ LLM fallback"}.get(report.source, report.source)

    # ── Header strip ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="vs-card">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;">
            <div>
              <span style="font-size:1.2rem;font-weight:700;">{report.student_name or "Student"}</span>
              <span style="opacity:.45;font-size:.85rem;margin-left:.5rem;">#{report.student_id or "—"}</span>
            </div>
            <div>
              {_status_pill(report.overall_status, status_col)}
              {_status_pill(report.predicted_grade, "#6366f1")}
              {_status_pill(report.goal_alignment, align_col)}
              <span class="pill {source_cls}" style="background:#ffffff0a;border:1px solid #ffffff15;">{source_icon}</span>
            </div>
          </div>
          <hr class="vs-divider" style="margin:.8rem 0;">
          <p style="font-size:.9rem;opacity:.75;margin:0;">{report.diagnosis_notes}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Confidence bar ────────────────────────────────────────────────────────
    pct = int(report.confidence_score * 100)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1rem;">
          <span style="font-size:.8rem;opacity:.55;white-space:nowrap;">Diagnosis confidence</span>
          <div class="conf-track" style="flex:1;">
            <div class="conf-fill" style="width:{pct}%;"></div>
          </div>
          <span style="font-size:.85rem;font-weight:600;">{pct}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_gaps, tab_strengths, tab_actions, tab_resources, tab_raw = st.tabs(
        ["🔴 Learning Gaps", "✅ Strengths", "🚀 Priority Actions", "📚 Resources", "📄 Raw Report"]
    )

    # ── Tab 1: Learning Gaps ──────────────────────────────────────────────────
    with tab_gaps:
        if not report.learning_gaps:
            st.success("🎉 No learning gaps identified — this student meets all assessed criteria!")
        else:
            for gap in sorted(report.learning_gaps,
                              key=lambda g: {"Critical": 0, "Moderate": 1, "Minor": 2}[g.severity]):
                sev_col = SEVERITY_COLOR.get(gap.severity, "#94a3b8")
                sev_bg  = SEVERITY_BG.get(gap.severity, "#111827")
                recs_html = "".join(f"<li>{r}</li>" for r in gap.recommendations)
                st.markdown(
                    f"""
                    <div class="gap-card"
                         style="background:{sev_bg};border-left-color:{sev_col};">
                      <h4 style="color:{sev_col};">
                        {gap.area}
                        <span class="pill" style="font-size:.7rem;background:{sev_col}22;
                              color:{sev_col};border:1px solid {sev_col}44;
                              margin-left:.5rem;">{gap.severity}</span>
                      </h4>
                      <p>📍 {gap.evidence}</p>
                      <ul>{recs_html}</ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Strengths ──────────────────────────────────────────────────────
    with tab_strengths:
        if not report.strengths:
            st.info("No explicit strengths recorded — check individual subject scores above.")
        else:
            for s in report.strengths:
                st.markdown(f"✅ &nbsp; {s}")

    # ── Tab 3: Priority Actions ───────────────────────────────────────────────
    with tab_actions:
        for i, action in enumerate(report.priority_actions, 1):
            st.markdown(
                f"""
                <div style="display:flex;gap:.75rem;align-items:flex-start;margin-bottom:.6rem;">
                  <span style="background:#6366f1;color:#fff;border-radius:50%;
                               width:1.5rem;height:1.5rem;display:flex;align-items:center;
                               justify-content:center;font-size:.75rem;font-weight:700;
                               flex-shrink:0;">{i}</span>
                  <span style="font-size:.88rem;padding-top:.15rem;">{action}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Tab 4: Resources (placeholder — filled by caller) ────────────────────
    # The actual content is injected by the caller via st.session_state
    # so that we can pass resources without re-running the Retrieval Node.
    with tab_resources:
        resources = st.session_state.get(f"_resources_{report.student_id}_{id(report)}", None)
        if resources is None:
            st.info("📡 Enable **RAG Resource Retrieval** in the sidebar and re-run diagnosis to see personalised resources here.", icon="📚")
        else:
            _render_resources(resources)

    # ── Tab 5: Raw JSON ───────────────────────────────────────────────────────
    with tab_raw:
        report_dict = report.to_dict()
        st.json(report_dict, expanded=False)
        st.download_button(
            label="⬇️ Download Diagnosis JSON",
            data=json.dumps(report_dict, indent=2, default=str),
            file_name=f"diagnosis_{report.student_id or 'student'}.json",
            mime="application/json",
            key=f"dl_json_{report.student_id}_{id(report)}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "📂 Upload Student Data (CSV)",
    type=["csv"],
    help="CSV should include columns like attendance_percentage, study_hours, math_score, science_score, english_score, internet_access, etc.",
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;opacity:.5;">
      <div style="font-size:3rem;">📋</div>
      <p style="font-size:1rem;margin:.5rem 0 0 0;">Upload a student CSV file using the button above to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load data & model ─────────────────────────────────────────────────────────
raw_df      = pd.read_csv(uploaded_file)
original_df = raw_df.copy()
model, scaler, scale_cols = load_model()

# ── Preprocess ────────────────────────────────────────────────────────────────
has_strings  = raw_df.select_dtypes(include="object").shape[1] > 0
processed_df = preprocess_raw_data(raw_df, scaler, scale_cols) if has_strings else raw_df.copy()
if "final_grade" in processed_df.columns:
    processed_df.drop(columns=["final_grade"], inplace=True)

input_df = processed_df.copy()
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

predictions  = model.predict(input_df)

results_df = original_df.copy()
results_df["Predicted Grade"]  = [GRADE_MAP.get(p, f"Grade {p}") for p in predictions]
results_df["Classification"]   = [CATEGORY_MAP.get(p, "Unknown") for p in predictions]
results_df["_pred_int"]        = predictions  # keep for diagnosis

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_data, tab_search = st.tabs(
    ["📊 Overview", "📄 Full Dataset", "🔍 Student Diagnosis"]
)

# ── Tab: Overview ─────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown("### 📊 Batch Summary")

    total      = len(results_df)
    grade_cts  = pd.Series(predictions).value_counts()
    top_grade  = GRADE_MAP.get(int(grade_cts.idxmax()), "N/A")
    top_cat    = CATEGORY_MAP.get(int(grade_cts.idxmax()), "N/A")
    at_risk_n  = int((predictions <= 1).sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Students",     total)
    with m2:
        st.metric("Most Common Grade",  top_grade)
    with m3:
        st.metric("Most Common Status", top_cat)
    with m4:
        st.metric("⚠️ At-Risk Students", at_risk_n,
                  delta=f"{at_risk_n/total*100:.1f}% of class",
                  delta_color="inverse")

    st.markdown("<hr class='vs-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Grade Distribution**")
        grade_dist = results_df["Predicted Grade"].value_counts().reset_index()
        grade_dist.columns = ["Grade", "Count"]
        st.bar_chart(grade_dist.set_index("Grade"))

    with col2:
        st.markdown("**Classification Distribution**")
        cat_dist = results_df["Classification"].value_counts().reset_index()
        cat_dist.columns = ["Status", "Count"]
        st.bar_chart(cat_dist.set_index("Status"))

    # ── Configured Goals ──────────────────────────────────────────────────────
    st.markdown("<hr class='vs-divider'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Configured Learning Goals")
    if student_goals:
        for g in student_goals:
            st.markdown(f"- {g}")
    else:
        st.warning("No goals configured. Add goals in the sidebar to power the Diagnosis Node.")

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<hr class='vs-divider'>", unsafe_allow_html=True)
    st.markdown("### 📥 Download Full Results")
    export_df = results_df.drop(columns=["_pred_int"], errors="ignore")
    st.download_button(
        label     = "⬇️ Download All Predictions as CSV",
        data      = export_df.to_csv(index=False),
        file_name = "predictions.csv",
        mime      = "text/csv",
    )

# ── Tab: Full Dataset ─────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### 📄 Uploaded Dataset with Predictions")
    display_df = results_df.drop(columns=["_pred_int"], errors="ignore")
    st.dataframe(display_df, use_container_width=True)

# ── Tab: Student Diagnosis ────────────────────────────────────────────────────
with tab_search:
    st.markdown("### 🔍 Student Search & Diagnosis")
    st.caption(
        "Search by student name, ID, or row number. "
        "The Diagnosis Node will analyse their performance against your configured goals."
    )

    if not student_goals:
        st.warning("⚠️ No learning goals are set. Please add goals in the sidebar before running diagnosis.")

    search_query = st.text_input(
        "Search student",
        placeholder="e.g. John, STU001, or row number 5",
        label_visibility="collapsed",
    )

    if not search_query.strip():
        st.info("👆 Enter a name, ID, or row number to look up a student.")
        st.stop()

    # ── Match logic ───────────────────────────────────────────────────────────
    query   = search_query.strip()
    matched = pd.DataFrame()

    if query.isdigit():
        row_num = int(query) - 1
        if 0 <= row_num < len(results_df):
            matched = results_df.iloc[[row_num]]

    if matched.empty:
        q_low = query.lower()
        for col in ["student_id", "student_name", "name", "id"]:
            if col in results_df.columns:
                mask    = results_df[col].astype(str).str.lower().str.contains(q_low, na=False)
                matched = pd.concat([matched, results_df[mask]])
        matched = matched.drop_duplicates()

    if matched.empty:
        st.warning(
            f"No student found for **\"{query}\"**. "
            f"Try a different name, ID, or a row number between 1 and {len(results_df)}."
        )
        st.stop()

    st.success(f"Found **{len(matched)}** student(s)")
    st.dataframe(
        matched.drop(columns=["_pred_int"], errors="ignore"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 🧠 Learning Diagnosis Report")

    if not student_goals:
        st.error(
            "No goals configured — please add at least one goal in the sidebar "
            "to enable the Diagnosis Node."
        )
        st.stop()

    for idx in matched.index:
        row      = results_df.loc[idx]
        pred_int = int(row["_pred_int"])

        # ── Build performance_data dict for the Diagnosis Node ────────────────
        perf_data = {}
        for field in [
            "student_id", "student_name",
            "attendance_percentage", "study_hours",
            "math_score", "science_score", "english_score",
            "internet_access", "extra_activities",
        ]:
            val = row.get(field)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                perf_data[field] = val

        perf_data["predicted_grade"]    = pred_int
        perf_data["predicted_category"] = pred_int  # use same index; category_map aligned

        sname = str(row.get("student_name", row.get("name", f"Student {idx + 1}")))
        sid   = str(row.get("student_id",   row.get("id", "—")))

        with st.expander(
            f"🎓 {sname} (#{sid}) — {row['Predicted Grade']} · {row['Classification']}",
            expanded=True,
        ):
            with st.spinner("Running Diagnosis Node…"):
                report = run_diagnosis_node(
                    student_goals   = student_goals,
                    performance_data= perf_data,
                    use_llm         = use_llm,
                )

            # ── RAG Retrieval ──────────────────────────────────────────────────
            resources: list = []
            if use_rag and RAG_AVAILABLE and report.learning_gaps:
                with st.spinner("📡 Retrieving personalised learning resources…"):
                    try:
                        gaps_dicts = [g.to_dict() for g in report.learning_gaps]
                        resources  = retrieve_resources_for_gaps(
                            learning_gaps = gaps_dicts,
                            use_llm       = rag_use_llm_summary,
                            top_k         = rag_top_k,
                        )
                    except Exception as rag_err:
                        st.warning(f"RAG retrieval encountered an error: {rag_err}", icon="⚠️")

            # Store resources in session state so _render_diagnosis can find them
            res_key = f"_resources_{report.student_id}_{id(report)}"
            st.session_state[res_key] = resources if (use_rag and RAG_AVAILABLE) else None

            _render_diagnosis(report, row)

            # ── RAG Resource count banner (outside tabs) ───────────────────────
            if use_rag and RAG_AVAILABLE:
                if resources:
                    st.success(
                        f"📚 **{len(resources)} learning resource(s)** retrieved for "
                        f"{len(report.learning_gaps)} identified gap(s). "
                        f"See the **Resources** tab above.",
                        icon="✅",
                    )
                elif report.learning_gaps:
                    st.info("RAG retrieval returned no results — try lowering the relevance threshold.", icon="📭")

            # ── Legacy recommendations (collapsible) ──────────────────────────
            with st.expander("💡 Quick Study Recommendations (rule-based)", expanded=False):
                legacy_data = {
                    "attendance_percentage": row.get("attendance_percentage", 100),
                    "study_hours":           row.get("study_hours", 10),
                    "math_score":            row.get("math_score", 100),
                    "science_score":         row.get("science_score", 100),
                    "english_score":         row.get("english_score", 100),
                    "internet_access":       row.get("internet_access", 1),
                }
                recs = generate_recommendations(legacy_data, f"Grade {pred_int}")
                for i, r in enumerate(recs, 1):
                    st.write(f"{i}. {r}")