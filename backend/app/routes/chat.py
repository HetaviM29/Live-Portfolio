from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.chat_model import ChatRequest, ChatResponse
from app.services.llm_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = ChatService()


@router.post("", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
	"""Accept a chat prompt and return a grounded response."""

	return chat_service.answer(payload)


@router.post("/stream")
async def chat_stream_endpoint(payload: ChatRequest):
	"""Stream the response as Server-Sent Events."""
	return StreamingResponse(
		chat_service.stream_answer(payload),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
	)
