from __future__ import annotations

import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
	"""Dense embedding model backed by sentence-transformers."""

	def __init__(self) -> None:
		model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
		self._model = SentenceTransformer(model_name)

	def encode(self, texts: List[str]) -> np.ndarray:
		"""Return float32 dense vectors for all texts."""
		if not texts:
			return np.empty((0, 0), dtype=np.float32)

		embeddings = self._model.encode(
			texts,
			convert_to_numpy=True,
			show_progress_bar=False,
		)
		return np.array(embeddings, dtype=np.float32)
