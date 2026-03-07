from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
	"""Metadata for each supporting portfolio document."""

	id: str = Field(..., description="Unique identifier for the document")
	title: str = Field(..., description="Human readable label for the document")
	section: str = Field(..., description="High-level grouping such as projects or skills")
	score: float = Field(..., ge=0.0, description="Similarity score between 0 and 1")


class ChatRequest(BaseModel):
	"""Incoming message from the frontend chat client."""

	message: str = Field(..., min_length=1, description="User's natural language prompt")
	session_id: str | None = Field(
		default=None,
		min_length=1,
		description="Optional chat session id used to preserve conversational context",
	)


class ChatResponse(BaseModel):
	"""Structured response returned to the UI."""

	answer: str
	sources: list[Source]
