"""
rag/vectorstore_setup.py
────────────────────────────────────────────────────────────────────────────
VectorStore Setup — Developer 3 (RAG Engineer)

Responsibilities
────────────────
1. Maintain a curated corpus of educational resource documents (title, URL,
   subject domain, short description) that covers the subjects tracked by
   VectoSpace: Mathematics, Science, English, Attendance, Study Time, etc.
2. Convert each document into a numeric embedding using one of:
      a) SentenceTransformers (local, no API key required) — preferred.
      b) A simple TF-IDF-based fallback for environments where
         sentence-transformers is not installed.
3. Build a FAISS index (or a plain NumPy dot-product store as a fallback)
   over those embeddings so that retriever.py can run fast k-NN queries.
4. Persist the built index to disk in  rag/data/  so subsequent runs load
   instantly without re-embedding on every startup.

Environment variables (all optional)
─────────────────────────────────────
  EMBEDDING_MODEL   – SentenceTransformer model name
                      (default: "all-MiniLM-L6-v2")
  VECTORSTORE_PATH  – directory where the index is saved
                      (default: rag/data)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent          # project root
_DEFAULT_STORE = _ROOT / "rag" / "data"
VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", str(_DEFAULT_STORE)))
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_INDEX_FILE = VECTORSTORE_PATH / "faiss_index.pkl"
_DOCS_FILE  = VECTORSTORE_PATH / "documents.json"

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS  — curated educational resources
#
# Each document is a dict with the following keys:
#   title   : human-readable resource name
#   url     : canonical link to the resource
#   domain  : one or more subject areas this resource covers
#   summary : 1-2 sentence description used as the embedding text *and*
#             returned to the student as the resource synopsis
# ─────────────────────────────────────────────────────────────────────────────

RESOURCE_CORPUS: list[dict[str, Any]] = [

    # ── Mathematics ───────────────────────────────────────────────────────────
    {
        "title":   "Khan Academy – Mathematics",
        "url":     "https://www.khanacademy.org/math",
        "domain":  ["Mathematics", "math"],
        "summary": (
            "Free, self-paced video lessons and practice exercises covering "
            "arithmetic, algebra, geometry, trigonometry, and calculus. "
            "Ideal for building foundational mathematics skills from scratch."
        ),
    },
    {
        "title":   "Paul's Online Math Notes",
        "url":     "https://tutorial.math.lamar.edu/",
        "domain":  ["Mathematics", "math", "calculus", "algebra"],
        "summary": (
            "Comprehensive, exam-friendly notes and worked examples for "
            "Algebra, Calculus I–III, Differential Equations, and Linear "
            "Algebra. Especially useful for university-level mathematics."
        ),
    },
    {
        "title":   "Brilliant – Math Courses",
        "url":     "https://brilliant.org/courses/math/",
        "domain":  ["Mathematics", "math", "problem solving"],
        "summary": (
            "Interactive problem-solving courses in mathematics and logic. "
            "Brilliant's guided approach teaches reasoning rather than rote "
            "memorisation, reinforcing deep conceptual understanding."
        ),
    },
    {
        "title":   "MIT OpenCourseWare – Mathematics",
        "url":     "https://ocw.mit.edu/courses/mathematics/",
        "domain":  ["Mathematics", "math", "linear algebra", "calculus"],
        "summary": (
            "Free MIT lecture notes, assignments, and exams for undergraduate "
            "mathematics courses including Single-Variable Calculus, Linear "
            "Algebra, and Probability & Statistics."
        ),
    },
    {
        "title":   "Wolfram MathWorld",
        "url":     "https://mathworld.wolfram.com/",
        "domain":  ["Mathematics", "math", "reference"],
        "summary": (
            "The web's most extensive mathematics encyclopedia, covering "
            "definitions, theorems, and examples across all branches of "
            "mathematics. Best used as a quick reference while studying."
        ),
    },

    # ── Science ───────────────────────────────────────────────────────────────
    {
        "title":   "Khan Academy – Science",
        "url":     "https://www.khanacademy.org/science",
        "domain":  ["Science", "physics", "chemistry", "biology"],
        "summary": (
            "Free video lessons and practice quizzes covering biology, "
            "chemistry, physics, and Earth sciences at secondary and "
            "introductory university levels."
        ),
    },
    {
        "title":   "CK-12 – Science Flexbooks",
        "url":     "https://www.ck12.org/student/",
        "domain":  ["Science", "physics", "chemistry", "biology"],
        "summary": (
            "Open-source, customisable digital textbooks aligned to school "
            "curricula. Includes simulations, videos, and adaptive practice "
            "for physics, chemistry, biology, and earth science."
        ),
    },
    {
        "title":   "PhET Interactive Science Simulations",
        "url":     "https://phet.colorado.edu/",
        "domain":  ["Science", "physics", "chemistry"],
        "summary": (
            "Free, research-based science simulations from the University of "
            "Colorado. Students can experiment with physics, chemistry, and "
            "earth science concepts interactively without any lab equipment."
        ),
    },
    {
        "title":   "NCBI – Biology and Life Sciences Resources",
        "url":     "https://www.ncbi.nlm.nih.gov/education/",
        "domain":  ["Science", "biology", "life sciences"],
        "summary": (
            "Educational resources from the National Center for Biotechnology "
            "Information, including tutorials on genetics, genomics, and "
            "molecular biology for students and educators."
        ),
    },

    # ── English / Language Arts ───────────────────────────────────────────────
    {
        "title":   "British Council – LearnEnglish",
        "url":     "https://learnenglish.britishcouncil.org/",
        "domain":  ["English", "language", "grammar", "writing"],
        "summary": (
            "Free English learning resources from the British Council: "
            "grammar explanations, listening exercises, reading passages, "
            "and writing guides suitable for all proficiency levels."
        ),
    },
    {
        "title":   "Purdue OWL – Writing and Grammar",
        "url":     "https://owl.purdue.edu/owl/general_writing/",
        "domain":  ["English", "writing", "grammar", "academic writing"],
        "summary": (
            "The Purdue Online Writing Lab is the definitive reference for "
            "academic writing, grammar rules, citation formats (APA, MLA), "
            "and essay structure. Invaluable for assignments and exams."
        ),
    },
    {
        "title":   "BBC Learning English",
        "url":     "https://www.bbc.co.uk/learningenglish/",
        "domain":  ["English", "language", "vocabulary", "listening"],
        "summary": (
            "Daily grammar tips, vocabulary lessons, and real-world English "
            "from the BBC. Audio and video content makes it ideal for "
            "improving both written and spoken English simultaneously."
        ),
    },
    {
        "title":   "Project Gutenberg – Free Classic Literature",
        "url":     "https://www.gutenberg.org/",
        "domain":  ["English", "literature", "reading"],
        "summary": (
            "Over 70,000 free eBooks of classic literature to improve reading "
            "comprehension and analytical skills. Reading a chapter daily is "
            "one of the most effective ways to strengthen English."
        ),
    },

    # ── Attendance / Time Management ──────────────────────────────────────────
    {
        "title":   "Coursera – Learning How to Learn",
        "url":     "https://www.coursera.org/learn/learning-how-to-learn",
        "domain":  ["Attendance", "Study Time", "habits", "productivity"],
        "summary": (
            "A science-backed MOOC by Barbara Oakley and Terrence Sejnowski "
            "covering memory, procrastination, and effective study techniques. "
            "Critical for students struggling with attendance or productivity."
        ),
    },
    {
        "title":   "Todoist – Student Planning Guide",
        "url":     "https://todoist.com/productivity-methods/student-study-plan",
        "domain":  ["Study Time", "Attendance", "planning", "organisation"],
        "summary": (
            "A practical guide to building a structured weekly study timetable, "
            "breaking large goals into daily tasks, and maintaining academic "
            "momentum throughout the semester using proven productivity methods."
        ),
    },
    {
        "title":   "Forest App – Focus & Study Timer",
        "url":     "https://www.forestapp.cc/",
        "domain":  ["Study Time", "focus", "productivity"],
        "summary": (
            "A gamified Pomodoro-style timer app that helps students stay "
            "focused by growing a virtual tree during study sessions. "
            "Reduces phone distraction and builds sustained study habits."
        ),
    },
    {
        "title":   "Cal Newport – Study Strategies (Blog)",
        "url":     "https://calnewport.com/blog/",
        "domain":  ["Study Time", "Study habits", "deep work"],
        "summary": (
            "Cal Newport's research-driven blog on deliberate practice, "
            "deep work, and effective study strategies for students. "
            "Highly recommended for improving study quality over quantity."
        ),
    },

    # ── General Academic Skills ───────────────────────────────────────────────
    {
        "title":   "Anki – Spaced Repetition Flashcards",
        "url":     "https://apps.ankiweb.net/",
        "domain":  ["Mathematics", "Science", "English", "general", "memory"],
        "summary": (
            "Free flashcard software using spaced-repetition algorithms to "
            "optimise long-term memorisation. Community decks are available "
            "for almost every school subject, from vocabulary to equations."
        ),
    },
    {
        "title":   "OpenStax – Free Peer-reviewed Textbooks",
        "url":     "https://openstax.org/subjects",
        "domain":  ["Mathematics", "Science", "English", "general"],
        "summary": (
            "High-quality, peer-reviewed, openly-licensed textbooks in maths, "
            "sciences, social sciences, and humanities. Fully free to read "
            "online — a direct replacement for expensive school textbooks."
        ),
    },
    {
        "title":   "Quizlet – Flashcards & Study Sets",
        "url":     "https://quizlet.com/",
        "domain":  ["Mathematics", "Science", "English", "general", "revision"],
        "summary": (
            "Millions of student-created study sets for every subject. "
            "Multiple learning modes (flashcards, matching, tests) make "
            "Quizlet effective for both vocabulary building and exam revision."
        ),
    },
    {
        "title":   "edX – University Online Courses",
        "url":     "https://www.edx.org/",
        "domain":  ["Mathematics", "Science", "English", "general"],
        "summary": (
            "Online courses from top universities (MIT, Harvard, etc.) on "
            "mathematics, science, writing, and more. Many courses are free "
            "to audit, making university-level content accessible to all."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING BACKEND  (SentenceTransformers → TF-IDF fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_sentence_transformers():
    """Return the SentenceTransformer class or None if not installed."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except ImportError:
        return None


def _build_tfidf_embedder(texts: list[str]):
    """
    Minimal TF-IDF vectoriser used as a fallback when sentence-transformers
    is not available.  Returns (matrix, vectoriser).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(texts).toarray()
    return mat, vec


# ─────────────────────────────────────────────────────────────────────────────
# FAISS / NUMPY INDEX
# ─────────────────────────────────────────────────────────────────────────────

def _build_faiss_index(embeddings):
    """
    Build a FAISS IndexFlatIP (inner-product / cosine after normalisation).
    Falls back to storing the raw numpy matrix when faiss-cpu is not available.
    """
    try:
        import faiss  # type: ignore
        import numpy as np

        mat = np.array(embeddings, dtype="float32")
        # L2-normalise so inner-product == cosine similarity
        faiss.normalize_L2(mat)
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        logger.info("Built FAISS IndexFlatIP with %d vectors (dim=%d).", *mat.shape)
        return ("faiss", index)

    except ImportError:
        import numpy as np

        mat = np.array(embeddings, dtype="float32")
        # Manual L2 normalisation
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        mat   = mat / norms
        logger.info("faiss-cpu not found — using NumPy dot-product store (%d vectors).", len(mat))
        return ("numpy", mat)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_vectorstore(force: bool = False) -> dict:
    """
    Build (or load from disk) the vectorstore.

    Parameters
    ----------
    force : bool
        If True, rebuild even if a cached index exists on disk.

    Returns
    -------
    dict with keys:
        "index_type" : "faiss" | "numpy" | "tfidf"
        "index"      : the FAISS index object or numpy matrix
        "documents"  : list[dict]  — the original RESOURCE_CORPUS entries
        "embedder"   : callable(texts) → embeddings, or (vectoriser, matrix)
        "embed_type" : "sentence_transformers" | "tfidf"
    """
    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    # ── Load from cache if available ──────────────────────────────────────────
    if not force and _INDEX_FILE.exists() and _DOCS_FILE.exists():
        logger.info("Loading vectorstore from cache: %s", VECTORSTORE_PATH)
        try:
            with open(_INDEX_FILE, "rb") as fh:
                store = pickle.load(fh)
            with open(_DOCS_FILE, "r", encoding="utf-8") as fh:
                store["documents"] = json.load(fh)
            logger.info("Vectorstore loaded (%d docs).", len(store["documents"]))
            return store
        except Exception as exc:
            logger.warning("Cache load failed (%s) — rebuilding.", exc)

    # ── Build embeddings ──────────────────────────────────────────────────────
    logger.info("Building vectorstore from corpus (%d docs)…", len(RESOURCE_CORPUS))

    # Compose embedding text: title + domain keywords + summary
    embed_texts = [
        f"{doc['title']}. {' '.join(doc['domain'])}. {doc['summary']}"
        for doc in RESOURCE_CORPUS
    ]

    SentenceTransformer = _try_import_sentence_transformers()

    if SentenceTransformer is not None:
        logger.info("Using SentenceTransformer: %s", EMBEDDING_MODEL)
        model      = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(embed_texts, show_progress_bar=False, normalize_embeddings=True)
        embed_type = "sentence_transformers"
        embedder   = model
    else:
        logger.info("sentence-transformers not found — using TF-IDF embedder.")
        embeddings, embedder = _build_tfidf_embedder(embed_texts)
        embed_type = "tfidf"

    # ── Build index ───────────────────────────────────────────────────────────
    index_type, index = _build_faiss_index(embeddings)

    store = {
        "index_type": index_type if embed_type == "sentence_transformers" else "tfidf",
        "index":      index,
        "documents":  RESOURCE_CORPUS,
        "embedder":   embedder,
        "embed_type": embed_type,
    }

    # ── Persist to disk ───────────────────────────────────────────────────────
    try:
        # Don't pickle the numpy matrix version of faiss — just store it all
        with open(_INDEX_FILE, "wb") as fh:
            pickle.dump({
                "index_type": store["index_type"],
                "index":      store["index"],
                "embedder":   store["embedder"],
                "embed_type": store["embed_type"],
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(_DOCS_FILE, "w", encoding="utf-8") as fh:
            # Documents are plain dicts — safe to JSON-serialise
            json.dump(RESOURCE_CORPUS, fh, indent=2, ensure_ascii=False)
        logger.info("Vectorstore persisted to: %s", VECTORSTORE_PATH)
    except Exception as exc:
        logger.warning("Could not persist vectorstore: %s", exc)

    return store


def search_vectorstore(
    query:       str,
    store:       dict,
    top_k:       int  = 5,
    min_score:   float = 0.10,
) -> list[dict]:
    """
    Retrieve the top-k most relevant documents for *query* from *store*.

    Parameters
    ----------
    query      : free-text query string (e.g. a learning gap description).
    store      : dict returned by build_vectorstore().
    top_k      : number of candidates to return.
    min_score  : minimum cosine similarity threshold (0–1). Documents below
                 this score are filtered out.

    Returns
    -------
    list[dict]
        Each element has keys: title, url, domain, summary, score.
        Sorted descending by score.
    """
    import numpy as np

    documents  = store["documents"]
    embedder   = store["embedder"]
    index_type = store["index_type"]
    embed_type = store["embed_type"]

    # ── Embed the query ───────────────────────────────────────────────────────
    if embed_type == "sentence_transformers":
        q_vec = embedder.encode([query], normalize_embeddings=True)
    else:
        # TF-IDF path: embedder is (vectoriser, corpus_matrix) tuple
        vectoriser = embedder
        q_vec = vectoriser.transform([query]).toarray()
        norm  = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10
        q_vec = q_vec / norm

    q_vec = np.array(q_vec, dtype="float32")

    # ── Search index ──────────────────────────────────────────────────────────
    if index_type == "faiss":
        scores, indices = store["index"].search(q_vec, min(top_k, len(documents)))
        scores  = scores[0].tolist()
        indices = indices[0].tolist()
    else:
        # numpy or tfidf dot-product
        corpus_mat = store["index"]
        raw_scores = corpus_mat.dot(q_vec[0])
        top_idx    = np.argsort(raw_scores)[::-1][:top_k]
        indices    = top_idx.tolist()
        scores     = raw_scores[top_idx].tolist()

    # ── Filter and format ─────────────────────────────────────────────────────
    results = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(documents):
            continue
        if float(score) < min_score:
            continue
        doc = dict(documents[idx])
        doc["score"] = round(float(score), 4)
        results.append(doc)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)

    store = build_vectorstore(force=True)
    print(f"\nVectorstore ready — {len(store['documents'])} documents indexed.")
    print(f"Embed backend : {store['embed_type']}")
    print(f"Index type    : {store['index_type']}\n")

    test_queries = [
        "Mathematics score is critically low, needs algebra and calculus help",
        "Student has poor English writing and grammar skills",
        "Low attendance and needs help with time management and study habits",
        "Science score is below average, needs physics and chemistry resources",
    ]

    for q in test_queries:
        print(f"QUERY : {q}")
        hits = search_vectorstore(q, store, top_k=3)
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['title']}  →  {h['url']}")
        print()
