from __future__ import annotations

from dataclasses import dataclass
from typing import List

import faiss

from .embeddings import EmbeddingModel
from .loader import PortfolioDocument


@dataclass(slots=True)
class RetrievedDocument:
	document: PortfolioDocument
	score: float


class PortfolioVectorStore:
	"""In-memory FAISS vector index backed by dense embeddings."""

	def __init__(self, documents: List[PortfolioDocument]) -> None:
		if not documents:
			raise ValueError("Cannot initialize PortfolioVectorStore with zero documents.")

		self._documents = documents
		self._embedder = EmbeddingModel()
		corpus = [doc.content for doc in documents]
		embeddings = self._embedder.encode(corpus)
		if embeddings.size == 0:
			raise ValueError("Could not build embeddings for the provided documents.")

		dimension = embeddings.shape[1]
		self._index = faiss.IndexFlatIP(dimension)
		faiss.normalize_L2(embeddings)
		self._index.add(embeddings)

	def similarity_search(self, query: str, top_k: int = 4) -> List[RetrievedDocument]:
		k = top_k or 4
		query_vector = self._embedder.encode([query])
		faiss.normalize_L2(query_vector)

		scores, indices = self._index.search(query_vector, min(k, len(self._documents)))

		results: List[RetrievedDocument] = []
		for score, idx in zip(scores[0], indices[0]):
			if idx < 0:
				continue
			results.append(
				RetrievedDocument(
					document=self._documents[idx],
					score=float(score),
				)
			)

		return results
