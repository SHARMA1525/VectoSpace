"""
rag/retriever.py
────────────────────────────────────────────────────────────────────────────
Retrieval Node — Developer 3 (RAG Engineer)

What this module does
─────────────────────
1. Accepts a learning-gap diagnosis (from Dev 2's Diagnosis Node) — either
   as a DiagnosisReport object or as the raw AgentState dict.
2. Translates each identified LearningGap into one or more semantic search
   queries (e.g.  "Mathematics Critical gap – score below 40").
3. Queries the vectorstore (built in vectorstore_setup.py) to retrieve the
   top-k most relevant educational resources for every gap.
4. Deduplicates results across gaps and ranks them by relevance score.
5. Uses an LLM (Gemini → OpenAI → template fallback) to generate a concise
   1–2 sentence summary for each retrieved resource, tailored to the
   specific gap it addresses.
6. Returns a fully-populated list of resource dicts — each with keys:
      title, url, summary, gap, severity, score
   — ready to be stored in AgentState["resources"].
7. Also exposes  run_retrieval_node(state)  which follows the LangGraph
   node signature and returns  {"resources": [...]} .

LLM Backend (same priority chain as diagnosis.py)
─────────────────────────────────────────────────
  a) Google Gemini (GEMINI_API_KEY)
  b) OpenAI GPT-4o (OPENAI_API_KEY)
  c) Template-based fallback summary when no API key is present.

Environment variables
─────────────────────
  GEMINI_API_KEY    – enables Google Gemini for summary generation.
  OPENAI_API_KEY    – enables OpenAI GPT-4o for summary generation.
  RAG_TOP_K         – number of resources to retrieve per gap   (default: 3)
  RAG_MIN_SCORE     – minimum cosine similarity threshold        (default: 0.10)
  RAG_MAX_RESOURCES – hard cap on total resources returned       (default: 10)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RAG_TOP_K         = int(os.getenv("RAG_TOP_K",         "3"))
RAG_MIN_SCORE     = float(os.getenv("RAG_MIN_SCORE",   "0.10"))
RAG_MAX_RESOURCES = int(os.getenv("RAG_MAX_RESOURCES", "10"))

# ─────────────────────────────────────────────────────────────────────────────
# LAZY VECTORSTORE  (built once per process, cached in module scope)
# ─────────────────────────────────────────────────────────────────────────────

_VECTORSTORE: dict | None = None


def _get_vectorstore() -> dict:
    """Return the module-level vectorstore, building it on first call."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        from rag.vectorstore_setup import build_vectorstore
        logger.info("Initialising vectorstore…")
        _VECTORSTORE = build_vectorstore()
    return _VECTORSTORE


# ─────────────────────────────────────────────────────────────────────────────
# QUERY BUILDER  — converts LearningGap info into search queries
# ─────────────────────────────────────────────────────────────────────────────

def _build_queries(area: str, severity: str, evidence: str) -> list[str]:
    """
    Generate 1–3 complementary search queries for a single learning gap.
    Multiple queries improve recall across differently-worded documents.
    """
    queries = [
        f"{area} {severity} learning gap study resources",
        f"{area} educational materials for students struggling with {area.lower()}",
    ]
    # Add an evidence-anchored query for richer context matching
    short_evidence = evidence[:120] if len(evidence) > 120 else evidence
    if short_evidence:
        queries.append(f"{area} resources: {short_evidence}")
    return queries


# ─────────────────────────────────────────────────────────────────────────────
# LLM SUMMARY GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def _generate_summary_with_llm(
    resource_title: str,
    resource_url:   str,
    resource_desc:  str,
    gap_area:       str,
    gap_severity:   str,
    gap_evidence:   str,
) -> str | None:
    """
    Ask the LLM to produce a 1–2 sentence, student-facing summary explaining
    *why* this specific resource helps address the identified gap.
    Returns None on failure so the caller can use a template fallback.
    """
    prompt = (
        f"You are an educational advisor writing concise resource summaries for students.\n\n"
        f"LEARNING GAP:\n"
        f"  Area     : {gap_area}\n"
        f"  Severity : {gap_severity}\n"
        f"  Evidence : {gap_evidence}\n\n"
        f"RESOURCE:\n"
        f"  Title   : {resource_title}\n"
        f"  URL     : {resource_url}\n"
        f"  Description: {resource_desc}\n\n"
        f"Write exactly 1–2 sentences explaining how this resource directly helps this "
        f"student address their {gap_area} gap. Be specific, practical, and encouraging. "
        f"Do NOT start with 'This resource' or 'This website'. Respond with the summary only."
    )

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=gemini_key)
            model    = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            text     = response.text.strip()
            if text:
                logger.debug("Gemini summary generated for '%s'.", resource_title)
                return text
        except Exception as exc:
            logger.warning("Gemini summary generation failed: %s", exc)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI  # type: ignore
            client   = OpenAI(api_key=openai_key)
            resp     = client.chat.completions.create(
                model       = "gpt-4o",
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.4,
                max_tokens  = 120,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                logger.debug("OpenAI summary generated for '%s'.", resource_title)
                return text
        except Exception as exc:
            logger.warning("OpenAI summary generation failed: %s", exc)

    return None   # signal to caller to use fallback template


def _make_summary(
    resource_title: str,
    resource_url:   str,
    resource_desc:  str,
    gap_area:       str,
    gap_severity:   str,
    gap_evidence:   str,
    use_llm:        bool = True,
) -> str:
    """
    Return a student-facing summary for a resource.
    Tries LLM first; uses a structured template on failure.
    """
    if use_llm:
        llm_summary = _generate_summary_with_llm(
            resource_title, resource_url, resource_desc,
            gap_area, gap_severity, gap_evidence,
        )
        if llm_summary:
            return llm_summary

    # ── Template fallback ─────────────────────────────────────────────────────
    severity_adverb = {
        "Critical": "urgently",
        "Moderate": "significantly",
        "Minor":    "helpfully",
    }.get(gap_severity, "directly")

    return (
        f"**{resource_title}** {severity_adverb} addresses your {gap_area} gap: "
        f"{resource_desc[:180].rstrip('.')}. "
        f"Use this resource to practise and improve your {gap_area.lower()} skills."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE RETRIEVAL LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_resources_for_gaps(
    learning_gaps: list[dict[str, Any]],
    use_llm:       bool = True,
    top_k:         int  = RAG_TOP_K,
    min_score:     float = RAG_MIN_SCORE,
    max_resources: int  = RAG_MAX_RESOURCES,
) -> list[dict[str, str]]:
    """
    Retrieve and summarise educational resources for a list of learning gaps.

    Parameters
    ----------
    learning_gaps : list[dict]
        Each dict should have at minimum:
            area        : str  (e.g. "Mathematics")
            severity    : str  ("Critical" | "Moderate" | "Minor")
            evidence    : str  (diagnosis evidence text)
            recommendations : list[str]  (optional, used to enrich queries)
    use_llm : bool
        Whether to call an LLM for richer resource summaries.
    top_k : int
        Number of candidate resources to retrieve per gap per query.
    min_score : float
        Minimum cosine similarity threshold.
    max_resources : int
        Hard cap on the total number of resources returned.

    Returns
    -------
    list[dict[str, str]]
        Each dict has: title, url, summary, gap, severity, score.
        Sorted by descending relevance score, deduplicated by URL.
    """
    if not learning_gaps:
        logger.info("No learning gaps provided — skipping retrieval.")
        return []

    store = _get_vectorstore()

    from rag.vectorstore_setup import search_vectorstore

    seen_urls: set[str]    = set()
    all_results: list[dict] = []

    # ── Process each gap ──────────────────────────────────────────────────────
    for gap in learning_gaps:
        area     = gap.get("area",     "General")
        severity = gap.get("severity", "Moderate")
        evidence = gap.get("evidence", "")

        logger.info("Retrieving resources for gap: %s (%s)", area, severity)

        queries = _build_queries(area, severity, evidence)

        gap_hits: dict[str, dict] = {}   # url → best-scoring hit

        for query in queries:
            hits = search_vectorstore(
                query     = query,
                store     = store,
                top_k     = top_k,
                min_score = min_score,
            )
            for hit in hits:
                url = hit["url"]
                # Keep only the highest-scoring occurrence per URL per gap
                if url not in gap_hits or hit["score"] > gap_hits[url]["score"]:
                    gap_hits[url] = hit

        # Sort this gap's hits by score and take top_k unique
        sorted_hits = sorted(gap_hits.values(), key=lambda h: h["score"], reverse=True)[:top_k]

        for hit in sorted_hits:
            url = hit["url"]
            if url in seen_urls:
                continue     # already included from a previous gap
            seen_urls.add(url)

            # Generate tailored summary
            summary = _make_summary(
                resource_title = hit["title"],
                resource_url   = url,
                resource_desc  = hit["summary"],
                gap_area       = area,
                gap_severity   = severity,
                gap_evidence   = evidence,
                use_llm        = use_llm,
            )

            all_results.append({
                "title":    hit["title"],
                "url":      url,
                "summary":  summary,
                "gap":      area,
                "severity": severity,
                "score":    str(round(hit["score"], 4)),
            })

            if len(all_results) >= max_resources:
                logger.info("Reached max_resources cap (%d). Stopping retrieval.", max_resources)
                break

        if len(all_results) >= max_resources:
            break

    logger.info("Retrieval complete. %d resources found for %d gaps.", len(all_results), len(learning_gaps))
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH NODE  — AgentState interface
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph Retrieval Node.

    Reads  state["learning_gaps"]  which is the serialised learning-gap list
    produced by the Diagnosis Node (Dev 2).

    Accepts three formats for  learning_gaps:
      • A list[dict]  with keys area/severity/evidence (native DiagnosisReport
        format after .to_dict()).
      • A JSON string encoding that list.
      • A plain English description string (legacy / mock format) — parsed
        with a simple heuristic to extract subject names.

    Returns
    -------
    dict
        {"resources": list[dict]}  — ready to merge into AgentState.
    """
    raw_gaps = state.get("learning_gaps")
    use_llm  = state.get("use_llm", True)

    # ── Normalise learning_gaps to list[dict] ─────────────────────────────────
    gaps: list[dict] = []

    if isinstance(raw_gaps, list):
        # Already the right type
        for item in raw_gaps:
            if isinstance(item, dict):
                gaps.append(item)
            elif hasattr(item, "to_dict"):
                gaps.append(item.to_dict())

    elif isinstance(raw_gaps, str):
        # Try JSON decode first
        import json, re
        try:
            decoded = json.loads(raw_gaps)
            if isinstance(decoded, list):
                gaps = decoded
        except (json.JSONDecodeError, ValueError):
            # Fall back: extract subject keywords from plain text
            subject_keywords = {
                "Mathematics": ["math", "mathematics", "algebra", "calculus"],
                "Science":     ["science", "physics", "chemistry", "biology"],
                "English":     ["english", "language", "writing", "grammar"],
                "Attendance":  ["attendance", "absent", "missing classes"],
                "Study Time":  ["study hours", "study time", "productivity"],
            }
            text_lower = raw_gaps.lower()
            for area, keywords in subject_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    gaps.append({
                        "area":     area,
                        "severity": "Moderate",
                        "evidence": raw_gaps[:200],
                    })

    elif hasattr(raw_gaps, "learning_gaps"):
        # DiagnosisReport object passed directly
        for g in raw_gaps.learning_gaps:
            gaps.append(g.to_dict() if hasattr(g, "to_dict") else g)

    if not gaps:
        logger.info("Retrieval Node: no parseable learning gaps in state. Returning empty resources.")
        return {"resources": []}

    logger.info("Retrieval Node: processing %d gap(s).", len(gaps))

    resources = retrieve_resources_for_gaps(
        learning_gaps = gaps,
        use_llm       = bool(use_llm),
    )

    return {"resources": resources}


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)

    # Simulate gaps produced by Dev 2's Diagnosis Node
    mock_gaps = [
        {
            "area":     "Mathematics",
            "severity": "Critical",
            "evidence": "Math score is 35, which is 25 points below the student's goal of 60.",
            "recommendations": [
                "Dedicate 45 minutes daily to algebra and arithmetic.",
                "Use Khan Academy for structured practice modules.",
            ],
        },
        {
            "area":     "Attendance",
            "severity": "Critical",
            "evidence": "Attendance is 58%, which is critically low (threshold: 60%).",
            "recommendations": [
                "Aim to attend 85% of classes over the next 4 weeks.",
            ],
        },
        {
            "area":     "Study Time",
            "severity": "Moderate",
            "evidence": "Only 4.0 study hours/week — below the minimum of 5 hours.",
            "recommendations": [
                "Increase weekly study time from 4 hrs to at least 7 hrs.",
            ],
        },
    ]

    print("=" * 60)
    print("TEST 1: Direct gap retrieval (no LLM)")
    print("=" * 60)
    resources = retrieve_resources_for_gaps(mock_gaps, use_llm=False)
    for r in resources:
        print(f"\n[{r['gap']} / {r['severity']}]  score={r['score']}")
        print(f"  Title   : {r['title']}")
        print(f"  URL     : {r['url']}")
        print(f"  Summary : {r['summary'][:160]}…" if len(r['summary']) > 160 else f"  Summary : {r['summary']}")

    print("\n" + "=" * 60)
    print("TEST 2: LangGraph node with JSON-encoded gaps")
    print("=" * 60)
    import json
    mock_state = {
        "learning_gaps": json.dumps(mock_gaps),
        "use_llm":       False,
    }
    result = run_retrieval_node(mock_state)
    print(f"\nrun_retrieval_node returned {len(result['resources'])} resources.")
    pprint.pprint(result["resources"], width=100)

    print("\n" + "=" * 60)
    print("TEST 3: LangGraph node with plain-text legacy gap string")
    print("=" * 60)
    mock_state_legacy = {
        "learning_gaps": "Mocked learning gaps based on performance data. mathematics attendance.",
        "use_llm":       False,
    }
    result_legacy = run_retrieval_node(mock_state_legacy)
    print(f"\nrun_retrieval_node (legacy) returned {len(result_legacy['resources'])} resources.")
    for r in result_legacy["resources"]:
        print(f"  [{r['gap']}] {r['title']}")

    print("\nAll retrieval tests passed ✓")
