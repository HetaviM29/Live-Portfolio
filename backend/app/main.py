from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.models.chat_model import ChatRequest
from app.routes.chat import chat_service, router as chat_router


def _get_allowed_origins() -> List[str]:
	raw = os.getenv("ALLOWED_ORIGINS", "*")
	origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
	if not origins:
		return ["*"]
	if "*" in origins:
		return ["*"]

	# Keep insertion order while removing duplicates.
	return list(dict.fromkeys(origins))


app = FastAPI(title="Hetavi Portfolio API", version="1.0.0", docs_url="/docs")

allowed_origins = _get_allowed_origins()
allow_credentials = "*" not in allowed_origins

app.add_middleware(
	CORSMiddleware,
	allow_origins=allowed_origins,
	allow_credentials=allow_credentials,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
	return {
		"name": "Hetavi Portfolio API",
		"status": "ok",
		"health": "/health",
		"docs": "/docs",
	}


@app.get("/health")
async def health_check() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/query")
async def query_stream_endpoint(payload: ChatRequest):
	"""Alias endpoint used by deployed frontend; streams SSE tokens."""
	return StreamingResponse(
		chat_service.stream_answer(payload),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
	)


app.include_router(chat_router)


def _resolve_port(default: int = 8000) -> int:
	raw_port = os.getenv("PORT")
	if not raw_port:
		return default

	try:
		return int(raw_port)
	except ValueError:
		return default


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(
		"app.main:app",
		host=os.getenv("HOST", "0.0.0.0"),
		port=_resolve_port(),
	)
