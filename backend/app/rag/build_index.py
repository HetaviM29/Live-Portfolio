"""Utility script to rebuild the in-memory index for local testing."""

from __future__ import annotations

from pprint import pprint

from .loader import load_portfolio_documents
from .vectorstore import PortfolioVectorStore


def rebuild_index() -> PortfolioVectorStore:
	documents = load_portfolio_documents()
	store = PortfolioVectorStore(documents)
	return store


if __name__ == "__main__":
	store = rebuild_index()
	sample = store.similarity_search("Tell me about Hetavi's projects", top_k=2)
	pprint(sample)
