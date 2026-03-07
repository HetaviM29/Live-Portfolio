from __future__ import annotations

from typing import List

from .vectorstore import PortfolioVectorStore, RetrievedDocument
from .loader import load_portfolio_documents


class PortfolioRetriever:
    """Thin wrapper that handles similarity ranking heuristics."""

    def __init__(self, vector_store: PortfolioVectorStore, min_score: float = 0.05) -> None:
        self._vector_store = vector_store
        self._min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        sections: list[str] | None = None,
    ) -> List[RetrievedDocument]:
        # Pull a wider candidate set first, then filter/rank.
        candidate_k = max(top_k * 4, 20)
        results = self._vector_store.similarity_search(query, top_k=candidate_k)

        if sections:
            section_set = {section.strip().lower() for section in sections}
            results = [r for r in results if r.document.section.lower() in section_set]

        filtered = [result for result in results if result.score >= self._min_score]
        return filtered[:top_k]


# Singleton instance for simple search function
_vector_store: PortfolioVectorStore | None = None
_retriever: PortfolioRetriever | None = None


def _get_retriever() -> PortfolioRetriever:
    global _vector_store, _retriever
    if _retriever is None:
        documents = load_portfolio_documents()
        _vector_store = PortfolioVectorStore(documents)
        _retriever = PortfolioRetriever(_vector_store, min_score=0.05)
    return _retriever


def search(query: str, top_k: int = 5, sections: list[str] | None = None) -> List[str]:
    """Simple search function that returns content strings for the LLM context."""
    retriever = _get_retriever()
    results = retriever.retrieve(query, top_k=top_k, sections=sections)
    return [item.document.content for item in results]


def search_with_metadata(
    query: str,
    top_k: int = 5,
    sections: list[str] | None = None,
) -> List[RetrievedDocument]:
    """Search function that returns full document metadata for sources."""
    retriever = _get_retriever()
    return retriever.retrieve(query, top_k=top_k, sections=sections)
