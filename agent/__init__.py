"""
agent/
──────
The VectoSpace Agent package.

Modules
-------
prompts   – LLM prompt templates, few-shot examples, and prompt builder.
diagnosis – Diagnosis Node: gap identification and structured report generation.
"""
from agent.diagnosis import run_diagnosis_node, batch_diagnose, DiagnosisReport, LearningGap
from agent.prompts   import build_diagnosis_prompt, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

__all__ = [
    "run_diagnosis_node",
    "batch_diagnose",
    "DiagnosisReport",
    "LearningGap",
    "build_diagnosis_prompt",
    "SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
]
