from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat import router as chat_router


def _get_allowed_origins() -> List[str]:
	raw = os.getenv("ALLOWED_ORIGINS", "*")
	origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
	return origins or ["*"]


app = FastAPI(title="Hetavi Portfolio API", version="1.0.0", docs_url="/docs")

app.add_middleware(
	CORSMiddleware,
	allow_origins=_get_allowed_origins(),
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
	return {"status": "ok"}


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
