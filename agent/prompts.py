"""
agent/prompts.py
────────────────────────────────────────────────────────────────
Prompt templates and builder functions for the Diagnosis Node.

Design decisions:
  • SYSTEM_PROMPT  – sets the model's role, output schema, and hard guardrails.
  • FEW_SHOT_EXAMPLES – two labelled input/output pairs that ground the model
    in the exact JSON schema expected by the final report.
  • build_diagnosis_prompt() – assembles the full conversation-style prompt
    (system + few-shot + live student data) ready to pass to the LLM.

Guardrail strategy
  ──────────────────
  1. Strict JSON schema enforcement:  the model is told to output ONLY valid
     JSON matching the DiagnosisReport schema — nothing else.
  2. Bounded recommendations:  each gap's recommendations list must have
     exactly 2–4 items; the model must refuse to fabricate subjects or
     scores that are not present in the input.
  3. Negative-example anchoring:  the few-shot set includes one strong
     student so the model learns it must output empty gap lists, not invent
     problems where none exist.
  4. Uncertainty hedge:  if a required field cannot be determined from the
     data, the model must use the sentinel string "INSUFFICIENT_DATA" rather
     than hallucinating a value.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert educational diagnostician and learning-gap analyst.
Your sole job is to analyse a student's performance data and stated learning
goals, then produce a structured JSON diagnosis report.

╔══════════════════════════════════════════════════════════════════════════╗
║  OUTPUT CONTRACT  — you MUST follow all rules below without exception   ║
╚══════════════════════════════════════════════════════════════════════════╝

1. RESPOND WITH VALID JSON ONLY.
   Do not emit any prose, markdown fences, or explanation outside the JSON.

2. SCHEMA (all keys required):
{
  "student_id"        : <string | null>,
  "student_name"      : <string | null>,
  "overall_status"    : <"At-Risk" | "Below-Average" | "Average" | "Above-Average" | "High-Performing" | "Exceptional">,
  "predicted_grade"   : <"Grade 0" | "Grade 1" | "Grade 2" | "Grade 3" | "Grade 4" | "Grade 5">,
  "goal_alignment"    : <"Aligned" | "Partially Aligned" | "Misaligned">,
  "learning_gaps"     : [
    {
      "area"            : <string>,
      "severity"        : <"Critical" | "Moderate" | "Minor">,
      "evidence"        : <string>,
      "recommendations" : [<string>, ...]   // 2–4 items, concrete and actionable
    }
  ],
  "strengths"         : [<string>],
  "priority_actions"  : [<string>],         // 3–5 top-level actions for the student
  "confidence_score"  : <float 0.0–1.0>,    // your confidence in this diagnosis
  "diagnosis_notes"   : <string>            // free-text summary ≤ 100 words
}

3. GUARDRAILS — violations make the output invalid:
   • NEVER fabricate a subject score not present in performance_data.
   • NEVER add a learning gap for a score ≥ 70 unless the student's goal
     explicitly requires mastery (≥ 90) of that subject.
   • If both scores and attendance are healthy and the student meets their
     goals, learning_gaps MUST be an empty list [].
   • If a field cannot be determined from the provided data, use the
     sentinel string "INSUFFICIENT_DATA".
   • Recommendations must reference specific numbers from the data
     (e.g. "Your math score is 42 — target 60 by practising …").
   • confidence_score reflects data completeness: penalise 0.1 for each
     missing key field (attendance, study_hours, subject scores).

4. TONE: professional, empathetic, evidence-based.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
# Each example is a (user_message, assistant_response) pair.
# Including a "strong student" example teaches the model to produce empty
# gap lists rather than inventing problems.

FEW_SHOT_EXAMPLES = [
    # ── Example 1: At-Risk student with clear gaps ────────────────────────
    {
        "role": "user",
        "content": """\
STUDENT GOALS:
- Pass all subjects with a score ≥ 60
- Improve overall grade to Grade 3

PERFORMANCE DATA:
{
  "student_id": "STU042",
  "student_name": "Arjun Mehta",
  "attendance_percentage": 61.0,
  "study_hours": 3.5,
  "math_score": 38,
  "science_score": 55,
  "english_score": 49,
  "internet_access": true,
  "extra_activities": false,
  "predicted_grade": 1,
  "predicted_category": "At-Risk"
}""",
    },
    {
        "role": "assistant",
        "content": """\
{
  "student_id": "STU042",
  "student_name": "Arjun Mehta",
  "overall_status": "At-Risk",
  "predicted_grade": "Grade 1",
  "goal_alignment": "Misaligned",
  "learning_gaps": [
    {
      "area": "Mathematics",
      "severity": "Critical",
      "evidence": "Math score is 38, which is 22 points below the student's goal of 60.",
      "recommendations": [
        "Dedicate 45 minutes daily to practicing fundamental algebra and arithmetic (current score: 38).",
        "Use free resources like Khan Academy to work through math modules sequentially.",
        "Request weekly 1-on-1 tutoring sessions focused on identified weak topics."
      ]
    },
    {
      "area": "English",
      "severity": "Moderate",
      "evidence": "English score is 49, which is 11 points below the student's goal of 60.",
      "recommendations": [
        "Read at least one article or chapter daily to build vocabulary and comprehension.",
        "Practice writing structured short paragraphs and request teacher feedback weekly."
      ]
    },
    {
      "area": "Attendance",
      "severity": "Critical",
      "evidence": "Attendance is 61%, which is below the 75% minimum threshold for adequate learning exposure.",
      "recommendations": [
        "Set a personal goal to attend every scheduled class for the next 4 weeks — target 80%.",
        "Identify specific reasons for absences and seek support (transport, health, scheduling)."
      ]
    },
    {
      "area": "Study Time",
      "severity": "Moderate",
      "evidence": "Only 3.5 study hours per week — insufficient for Grade 3 targets.",
      "recommendations": [
        "Increase self-study to at least 7 hours per week using a structured timetable.",
        "Use the Pomodoro technique: 25-minute focused sessions with 5-minute breaks."
      ]
    }
  ],
  "strengths": [
    "Science score of 55 is the strongest subject and approaching the 60 target.",
    "Has internet access, enabling use of online learning resources."
  ],
  "priority_actions": [
    "Immediately book an academic counselling session to create a recovery plan.",
    "Raise attendance to above 85% starting this week.",
    "Focus 60% of study time on Mathematics (score: 38) and English (score: 49).",
    "Schedule a weekly progress check-in with a teacher or mentor."
  ],
  "confidence_score": 0.92,
  "diagnosis_notes": "Arjun is significantly misaligned with his goals across attendance, study hours, and two of three scored subjects. Mathematics is the most critical gap. Immediate remediation and consistent attendance improvement are the top priorities. Science shows relative strength and can serve as a confidence anchor."
}""",
    },

    # ── Example 2: High-Performing student – no gaps ──────────────────────
    {
        "role": "user",
        "content": """\
STUDENT GOALS:
- Maintain Grade 4 or above
- Score above 75 in all subjects

PERFORMANCE DATA:
{
  "student_id": "STU109",
  "student_name": "Priya Sharma",
  "attendance_percentage": 91.0,
  "study_hours": 14,
  "math_score": 88,
  "science_score": 92,
  "english_score": 85,
  "internet_access": true,
  "extra_activities": true,
  "predicted_grade": 4,
  "predicted_category": "High-Performing"
}""",
    },
    {
        "role": "assistant",
        "content": """\
{
  "student_id": "STU109",
  "student_name": "Priya Sharma",
  "overall_status": "High-Performing",
  "predicted_grade": "Grade 4",
  "goal_alignment": "Aligned",
  "learning_gaps": [],
  "strengths": [
    "Attendance at 91% is well above the recommended threshold, ensuring consistent knowledge exposure.",
    "All subject scores (Math: 88, Science: 92, English: 85) exceed the stated goal of 75.",
    "14 study hours per week reflect strong academic discipline.",
    "Participation in extra activities supports well-rounded development."
  ],
  "priority_actions": [
    "Continue the current structured study routine — it is clearly effective.",
    "Consider peer tutoring or mentoring to reinforce your own understanding while helping others.",
    "Explore advanced or enrichment material in Science (score: 92) to prepare for higher-level studies."
  ],
  "confidence_score": 0.97,
  "diagnosis_notes": "Priya meets or exceeds all stated academic goals. No learning gaps are identified. Her strong attendance combined with disciplined study habits forms the foundation of her performance. The recommended focus is on stretch goals and sustained excellence rather than remediation."
}""",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_diagnosis_prompt(student_goals: list[str], performance_data: dict) -> list[dict]:
    """
    Constructs a full conversation-style prompt list for the LLM.

    Parameters
    ----------
    student_goals : list[str]
        A list of the student's stated learning goals.
        Example: ["Pass all subjects with ≥ 60", "Improve attendance to 80%"]

    performance_data : dict
        Raw performance record for a single student.  Expected keys (all
        optional but scored on confidence):
          student_id, student_name, attendance_percentage, study_hours,
          math_score, science_score, english_score, internet_access,
          extra_activities, predicted_grade, predicted_category

    Returns
    -------
    list[dict]
        A messages list in OpenAI / Gemini chat format:
        [
          {"role": "system",    "content": SYSTEM_PROMPT},
          {"role": "user",      "content": <few-shot example 1>},
          {"role": "assistant", "content": <few-shot response 1>},
          ...
          {"role": "user",      "content": <live student data>},
        ]
    """
    import json

    # Format the student goals block
    goals_block = "\n".join(f"- {g}" for g in student_goals) if student_goals else "- No specific goals provided."

    # Sanitise the performance data dict for serialisation
    safe_data = {k: v for k, v in performance_data.items() if v is not None}
    data_block = json.dumps(safe_data, indent=2, default=str)

    live_user_message = (
        f"STUDENT GOALS:\n{goals_block}\n\n"
        f"PERFORMANCE DATA:\n{data_block}"
    )

    # Assemble messages list
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject few-shot examples
    for msg in FEW_SHOT_EXAMPLES:
        messages.append(msg)

    # Append the live request
    messages.append({"role": "user", "content": live_user_message})

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys

    sample_goals = ["Score ≥ 65 in all subjects", "Reduce absences to under 10%"]
    sample_data = {
        "student_id": "STU007",
        "student_name": "Dev Patel",
        "attendance_percentage": 58.0,
        "study_hours": 4.0,
        "math_score": 44,
        "science_score": 70,
        "english_score": 55,
        "internet_access": True,
        "predicted_grade": 1,
        "predicted_category": "At-Risk",
    }

    messages = build_diagnosis_prompt(sample_goals, sample_data)
    print(f"Total messages in prompt chain: {len(messages)}")
    for i, m in enumerate(messages):
        print(f"\n[{i}] role={m['role']}")
        print(m["content"][:300] + ("…" if len(m["content"]) > 300 else ""))
