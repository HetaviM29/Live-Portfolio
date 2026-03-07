from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(slots=True)
class PortfolioDocument:
	"""Plain representation of a portfolio knowledge chunk."""

	id: str
	title: str
	content: str
	section: str


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _clean_text(text: str) -> str:
	"""Normalize markdown text while keeping useful structure."""
	cleaned = text.replace("\r\n", "\n")
	cleaned = re.sub(r"\n\s*---\s*\n", "\n", cleaned)
	cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
	return cleaned.strip()


def _section_from_filename(file_path: Path) -> str:
	return file_path.stem.replace("_", " ").strip().lower()


def _split_by_h2(text: str, fallback_title: str) -> List[tuple[str, str]]:
	"""Split markdown into sections by '## ' headers.

	If no H2 headers exist, return one section for the whole file.
	"""
	lines = text.splitlines()
	sections: List[tuple[str, str]] = []
	current_title = fallback_title
	current_lines: List[str] = []

	for line in lines:
		stripped = line.strip()
		if stripped.startswith("## "):
			if current_lines:
				body = _clean_text("\n".join(current_lines))
				if body:
					sections.append((current_title, body))
				current_lines = []
			current_title = stripped[3:].strip() or fallback_title
		else:
			current_lines.append(line)

	if current_lines:
		body = _clean_text("\n".join(current_lines))
		if body:
			sections.append((current_title, body))

	return sections if sections else [(fallback_title, _clean_text(text))]


def _chunk_qa_file(text: str, section: str) -> List[PortfolioDocument]:
	"""Parse Q/A markdown into query-friendly chunks."""
	chunks: List[PortfolioDocument] = []
	blocks = re.split(r"(?=^Q:\s)", text, flags=re.MULTILINE)

	for idx, block in enumerate(blocks):
		content = _clean_text(block)
		if not content:
			continue

		q_match = re.search(r"^Q:\s*(.+)$", content, flags=re.MULTILINE)
		title = q_match.group(1).strip() if q_match else f"Q&A {idx + 1}"
		chunks.append(
			PortfolioDocument(
				id=f"{section}-qa-{idx}",
				title=title,
				section=section,
				content=f"Section: {section}\nTitle: {title}\n{content}",
			)
		)

	return chunks


def load_portfolio_documents() -> List[PortfolioDocument]:
	"""Read all markdown portfolio files and create retrieval chunks."""

	if not DATA_DIR.exists():
		raise FileNotFoundError(f"Portfolio data directory not found at {DATA_DIR}")

	md_files = sorted(DATA_DIR.glob("*.md"))
	if not md_files:
		raise FileNotFoundError(f"No markdown files found in {DATA_DIR}")

	documents: List[PortfolioDocument] = []
	for md_file in md_files:
		raw_text = md_file.read_text(encoding="utf-8")
		section = _section_from_filename(md_file)
		fallback_title = md_file.stem.replace("_", " ").title()

		if section == "qa examples":
			documents.extend(_chunk_qa_file(raw_text, section))
			continue

		sections = _split_by_h2(raw_text, fallback_title)
		for idx, (title, body) in enumerate(sections):
			if len(body) < 24:
				continue

			documents.append(
				PortfolioDocument(
					id=f"{section}-{idx}",
					title=title,
					section=section,
					content=f"Section: {section}\nTitle: {title}\n{body}",
				)
			)

	return documents
