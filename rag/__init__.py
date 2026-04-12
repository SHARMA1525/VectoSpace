"""
rag/
────
The VectoSpace RAG (Retrieval-Augmented Generation) package.

Developer 3 — Knowledge Retrieval & RAG Engineer

Modules
-------
vectorstore_setup  – Corpus, embedding, and FAISS index construction.
retriever          – Retrieval Node: gap → query → resources → summaries.

Public API
----------
    from rag.retriever          import run_retrieval_node, retrieve_resources_for_gaps
    from rag.vectorstore_setup  import build_vectorstore, search_vectorstore, RESOURCE_CORPUS

Quickstart
----------
    from rag.retriever import run_retrieval_node

    # Plug into the LangGraph AgentState after the Diagnosis Node
    result = run_retrieval_node({
        "learning_gaps": diagnosis_report.to_dict()["learning_gaps"],
        "use_llm":       True,
    })
    # result["resources"] → list[dict] with title, url, summary, gap, severity, score
"""

from rag.retriever         import run_retrieval_node, retrieve_resources_for_gaps
from rag.vectorstore_setup import build_vectorstore, search_vectorstore, RESOURCE_CORPUS

__all__ = [
    "run_retrieval_node",
    "retrieve_resources_for_gaps",
    "build_vectorstore",
    "search_vectorstore",
    "RESOURCE_CORPUS",
]
