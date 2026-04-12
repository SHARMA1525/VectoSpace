# 🎓 VectoSpace · Student Performance Predictor + Diagnosis Engine

## Overview

VectoSpace is a Streamlit-based web application that predicts student academic performance using a pre-trained Random Forest classifier, then runs a **Diagnosis Node** to identify learning gaps, evaluate goal alignment, and generate structured, evidence-based recommendations.

### Core Features

| Feature | Description |
|---|---|
| **CSV Upload** | Upload a batch of student records for bulk prediction |
| **Grade Prediction** | Classify students into Grade 0–5 using ML (Random Forest) |
| **Status Classification** | At-Risk → Below-Average → Average → Above-Average → High-Performing → Exceptional |
| **🆕 Diagnosis Engine** | LLM-powered (or rule-based) learning gap analysis against student goals |
| **🆕 Goal Alignment** | Determines if a student is Aligned / Partially Aligned / Misaligned with their goals |
| **🆕 Severity Scoring** | Gaps rated Critical / Moderate / Minor with evidence and recommendations |
| **🆕 Confidence Score** | Data completeness score penalised for missing key fields |
| **Data Visualisation** | Grade and Classification distribution charts |
| **Export Results** | Download predictions as CSV or individual diagnosis as JSON |

---

## Project Structure

```text
VectoSpace/
├── app.py                        # Main Streamlit application (UI + orchestration)
├── agent/                        # 🆕 Diagnosis Engine
│   ├── __init__.py               # Public API exports
│   ├── prompts.py                # LLM prompt templates, few-shot examples, guardrails
│   └── diagnosis.py              # Diagnosis Node: rule engine + LLM caller + schema validation
├── src/
│   └── ml/
│       ├── recommender.py        # Legacy rule-based recommendation engine
│       ├── retrain.py            # Model retraining script
│       └── models/               # Saved artefacts (random_forest.pkl, scaler.pkl, ...)
├── notebooks/                    # Jupyter notebooks for data analysis & experiments
├── datasets/                     # Raw datasets
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Hariksh/VectoSpace.git
cd VectoSpace

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib seaborn

# Optional — for LLM-powered Diagnosis:
pip install google-generativeai   # for Gemini
pip install openai                 # for GPT-4o
```

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage

1. **Upload** a CSV containing student records (see column guide below).
2. **Configure goals** in the sidebar — one goal per line (e.g. *"Score ≥ 65 in all subjects"*).
3. *(Optional)* **Enable LLM Diagnosis** in the sidebar — requires `GEMINI_API_KEY` or `OPENAI_API_KEY` in your environment.
4. View the **Overview** tab for batch metrics and charts.
5. Use **Student Diagnosis** to search by name, ID, or row number and see the full diagnosis report.
6. **Download** predictions as CSV or individual student reports as JSON.

### Expected CSV Columns

| Column | Type | Description |
|---|---|---|
| `student_id` | string | Unique student identifier (optional) |
| `student_name` | string | Student name (optional) |
| `attendance_percentage` | float | Attendance % (0–100) |
| `study_hours` | float | Weekly self-study hours |
| `math_score` | int | Mathematics score (0–100) |
| `science_score` | int | Science score (0–100) |
| `english_score` | int | English score (0–100) |
| `internet_access` | yes/no | Internet access at home |
| `extra_activities` | yes/no | Participation in extra-curricular activities |
| `travel_time` | string | `<15 min`, `15-30 min`, `30-60 min`, `>60 min` |
| `parent_education` | string | `no formal`, `high school`, `diploma`, `graduate`, `post graduate`, `phd` |
| `gender` | string | `male` / `female` / other |

---

## 🧠 Diagnosis Engine (agent/)

### Architecture

```
student_goals + performance_data
        │
        ▼
┌─────────────────────────┐
│   run_diagnosis_node()  │
│                         │
│  1. Rule-based engine   │  ← always runs (fast, deterministic)
│  2. Prompt builder      │  ← few-shot + guardrails
│  3. LLM caller          │  ← Gemini → OpenAI → skip
│  4. JSON validator      │  ← schema + allow-list checks
└─────────────────────────┘
        │
        ▼
   DiagnosisReport
```

### Prompt Strategy

**`agent/prompts.py`** implements a multi-layer prompt safety strategy:

| Strategy | Implementation |
|---|---|
| **Role priming** | System prompt defines model as an "expert educational diagnostician" |
| **Schema enforcement** | Exact JSON structure specified with types; model told to output JSON only |
| **Few-shot prompting** | 2 labelled examples: one At-Risk (4 gaps), one High-Performing (empty gaps) |
| **Negative anchoring** | High-Performing example teaches model to output `[]` gaps — not hallucinate problems |
| **Guardrails** | 4 rules: no fabrication, empty gaps when clean, `INSUFFICIENT_DATA` sentinel, 2–4 recs |
| **Uncertainty hedging** | `confidence_score` penalised for missing data fields |

### DiagnosisReport Schema

```json
{
  "student_id":        "STU007",
  "student_name":      "Dev Patel",
  "overall_status":    "At-Risk",
  "predicted_grade":   "Grade 1",
  "goal_alignment":    "Misaligned",
  "learning_gaps": [
    {
      "area":            "Mathematics",
      "severity":        "Critical",
      "evidence":        "Math score is 35, 25 points below goal of 60.",
      "recommendations": ["...", "...", "..."]
    }
  ],
  "strengths":        ["Science score of 68 meets or approaches goal."],
  "priority_actions": ["[Critical] Address Attendance: ..."],
  "confidence_score":  0.9,
  "diagnosis_notes":   "Dev Patel has 4 identified gaps ...",
  "source":            "rule-based"
}
```

---

## Machine Learning

The core prediction uses a **Random Forest classifier** located at `src/ml/models/random_forest.pkl`.  
Training features are listed in `src/ml/models/feature_names.pkl`.  
During inference, missing columns are filled with `0` to prevent feature-mismatch errors.

---

## LLM Configuration (Optional)

Set one of the following environment variables to enable LLM-powered diagnosis:

```bash
export GEMINI_API_KEY="your-gemini-api-key"   # Google Gemini (preferred)
# or
export OPENAI_API_KEY="your-openai-key"        # OpenAI GPT-4o
```

If neither key is set, the app falls back seamlessly to the deterministic rule-based engine — **no crash, no error**.
