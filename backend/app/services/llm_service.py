from __future__ import annotations

import json
import os
import re
from pathlib import Path
from threading import RLock
from typing import Generator, List

import requests
from dotenv import load_dotenv

from app.models.chat_model import ChatRequest, ChatResponse, Source
from app.rag.loader import PortfolioDocument, load_portfolio_documents
from app.rag.retriever import search, search_with_metadata

# Load .env from backend folder or root folder
backend_env = Path(__file__).resolve().parent.parent.parent / ".env"
root_env = Path(__file__).resolve().parent.parent.parent.parent / ".env"

if backend_env.exists():
    load_dotenv(backend_env, override=True)
elif root_env.exists():
    load_dotenv(root_env, override=True)
else:
    load_dotenv(override=True)


def _detect_sections(question: str) -> tuple[list[str] | None, int]:
    """Map question intent to relevant sections and retrieval depth."""
    q = question.lower()

    # Check project first — "worked on projects" should route to projects, not experience
    if _is_project_question(question):
        return ["projects"], 8
    if any(token in q for token in ["hire", "why should we hire", "fit for", "right candidate"]):
        return ["about me", "skills", "experience", "projects"], 8
    if any(token in q for token in ["intern", "internship", "experience", "work", "job"]):
        sections = ["experience"]
        if "education" in q:
            sections.append("education")
        return sections, 8
    if any(token in q for token in ["skill", "tech", "technology", "stack", "language"]):
        return ["skills", "qa examples", "about me"], 7
    if any(token in q for token in ["education", "college", "school", "degree"]):
        return ["education", "about me"], 6
    if any(token in q for token in ["achievement", "award", "certification"]):
        return ["achievements"], 6
    if any(token in q for token in ["about", "who", "background", "introduce", "introduction"]):
        return ["about me", "qa examples"], 6

    return None, 5


def _extract_user_name(question: str) -> str | None:
    """Extract a likely user name from conversational intros."""
    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{0,40})",
        r"\bi am\s+([A-Za-z][A-Za-z\s'-]{0,40})",
        r"\bi'm\s+([A-Za-z][A-Za-z\s'-]{0,40})",
    ]
    blocked = {"hetavi", "assistant", "there", "here", "fine"}

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        raw_name = re.sub(r"[^A-Za-z\s'-]", "", match.group(1)).strip()
        if not raw_name:
            continue
        parts = [part for part in raw_name.split() if part]
        if not parts:
            continue
        candidate = " ".join(parts[:2])
        if candidate.lower() in blocked:
            continue
        return " ".join(part.capitalize() for part in candidate.split())

    return None


def _is_conversational_opening(question: str) -> bool:
    """Detect greeting or self-introduction messages."""
    q = question.strip().lower()
    if not q:
        return False

    portfolio_terms = [
        "project",
        "skills",
        "experience",
        "education",
        "achievement",
        "hire",
        "technology",
        "tech",
        "contact",
        "internship",
        "about hetavi",
        "tell me about",
    ]
    if any(term in q for term in portfolio_terms):
        return False

    has_self_intro = bool(re.search(r"\b(my name is|i am|i'm)\s+[a-z]", q))
    if has_self_intro:
        return True

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return any(q == greeting or q.startswith(f"{greeting} ") for greeting in greetings)


def _build_opening_reply(question: str, user_name: str | None) -> str | None:
    """Return a short onboarding-style reply for conversational openers."""
    if not _is_conversational_opening(question):
        return None

    if user_name:
        return (
            f"Nice to meet you, {user_name}. What would you like to know about Hetavi "
            "today: projects, skills, experience, education, or achievements?"
        )

    return (
        "Hi. How can I help you with information about Hetavi? "
        "You can ask about her projects, skills, experience, education, or achievements."
    )


def _is_project_question(question: str) -> bool:
    """Detect project-focused questions even when phrased as technology/system queries."""
    q = question.lower()
    project_terms = [
        "project",
        "projects",
        "phishing",
        "student management",
        "electronic health",
        "health record",
        "living portfolio",
        "ai portfolio",
        "narrative based rpg",
        "physiotherapy",
        "real review",
        "avl dictionary",
        "document intelligence",
        "chatbot portfolio",
    ]
    if any(token in q for token in project_terms):
        return True

    system_words = ["system", "application", "portal", "chatbot"]
    project_ops = ["technology", "technologies", "tech", "used", "built", "involved", "challenge"]
    return any(token in q for token in system_words) and any(token in q for token in project_ops)


def _question_keywords(question: str) -> list[str]:
    """Extract useful keywords for lightweight project matching."""
    q = question.lower()
    stop_words = {
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "tell",
        "about",
        "her",
        "his",
        "their",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "did",
        "does",
        "do",
        "in",
        "on",
        "of",
        "to",
        "for",
        "with",
        "and",
        "me",
        "she",
        "project",
        "projects",
        "technology",
        "technologies",
        "tech",
        "used",
        "use",
        "involved",
        "system",
        "systems",
        "application",
        "applications",
        "challenge",
        "challenges",
    }

    keywords: list[str] = []
    for phrase in [
        "machine learning",
        "computer vision",
        "document intelligence",
        "student management",
        "health record",
        "real review",
        "living portfolio",
        "ai portfolio",
        "text classification",
        "rag",
        "retrieval augmented generation",
    ]:
        if phrase in q:
            keywords.append(phrase)

    for token in re.findall(r"[a-z0-9+/#-]+", q):
        if token in stop_words:
            continue
        if len(token) < 3 and token not in {"ai", "ml", "cv"}:
            continue
        keywords.append(token)

    unique: list[str] = []
    seen: set[str] = set()
    for item in keywords:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _project_match_score(doc: PortfolioDocument, question: str, keywords: list[str]) -> int:
    """Score how well a project document matches the question."""
    q = question.lower()
    body = _strip_chunk_metadata(doc.content)
    title = doc.title.lower()
    searchable = f"{title}\n{body.lower()}"

    score = 0
    if title in q:
        score += 8

    for keyword in keywords:
        if keyword in searchable:
            score += 2 if keyword in title else 1

    alias_map = {
        "living portfolio": ["rag", "retrieval augmented generation", "document intelligence", "chatbot"],
        "ai-resistant phishing": ["phishing", "deception", "machine learning", "text classification"],
        "student management": ["nrutyashree", "dance academy", "student", "portal"],
        "electronic health": ["ehr", "health", "medical", "patient"],
        "narrative based rpg": ["rpg", "story", "game", "branching"],
        "physiotherapy": ["posture", "exercise", "pose", "mediapipe"],
        "real review": ["review", "authenticity", "nlp"],
        "avl dictionary": ["avl", "dictionary", "tree", "c++"],
    }
    for title_key, aliases in alias_map.items():
        if title_key in title and any(alias in q for alias in aliases):
            score += 3

    return score


def _top_project_matches(projects: list[PortfolioDocument], question: str) -> list[PortfolioDocument]:
    keywords = _question_keywords(question)
    ranked: list[tuple[int, PortfolioDocument]] = []
    min_score = 2
    for doc in projects:
        score = _project_match_score(doc, question, keywords)
        if score >= min_score:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked]


def _normalize_pronouns(text: str) -> str:
    """Enforce she/her phrasing in generated response text."""
    if not text:
        return text

    replacements = {
        r"\bhe's\b": "she's",
        r"\bHe's\b": "She's",
        r"\bhe is\b": "she is",
        r"\bHe is\b": "She is",
        r"\bhe\b": "she",
        r"\bHe\b": "She",
        r"\bhim\b": "her",
        r"\bHim\b": "Her",
        r"\bhis\b": "her",
        r"\bHis\b": "Her",
    }
    normalized = text
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _first_meaningful_line(content: str) -> str:
    cleaned = re.sub(r"^Section:\s*.+$", "", content, flags=re.MULTILINE)
    cleaned = re.sub(r"^Title:\s*.+$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    for line in cleaned.splitlines():
        stripped = line.strip().lstrip("-*")
        if stripped and not stripped.startswith("#"):
            return stripped
    return cleaned[:140].strip()


def _extract_field(content: str, field: str) -> str:
    match = re.search(rf"{field}:\s*(.+)", content, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_bullets(text: str, limit: int = 4) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
            if len(bullets) >= limit:
                break
    return bullets


def _truncate_text(text: str, max_len: int = 120) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned

    # Prefer ending at a sentence boundary before max_len.
    sentence_end = -1
    for token in [". ", "! ", "? ", ".", "!", "?"]:
        idx = cleaned.rfind(token, 0, max_len + 1)
        if idx > sentence_end:
            sentence_end = idx
    if sentence_end >= 0:
        sentence = cleaned[: sentence_end + 1].strip()
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    # Fall back to word boundary and close with punctuation (no broken words).
    word_end = cleaned.rfind(" ", 0, max_len + 1)
    if word_end > 0:
        snippet = cleaned[:word_end].rstrip()
        if snippet and snippet[-1] not in ".!?":
            snippet += "."
        return snippet

    return cleaned


def _best_summary_line(content: str) -> str:
    """Extract a meaningful sentence while skipping label-only markdown lines."""
    cleaned = _strip_chunk_metadata(content)
    skip_exact = {
        "achievement:",
        "description:",
        "role:",
        "responsibilities included:",
        "marks:",
    }

    for line in cleaned.splitlines():
        stripped = line.strip().lstrip("-* ")
        if not stripped or stripped.startswith("#"):
            continue
        lower = stripped.lower()
        if lower in skip_exact:
            continue
        if lower.endswith(":") and len(lower.split()) <= 4:
            continue
        return stripped

    return _first_meaningful_line(content)


def _unique_docs(docs: list[PortfolioDocument]) -> list[PortfolioDocument]:
    seen: set[str] = set()
    unique: list[PortfolioDocument] = []
    for doc in docs:
        if doc.id in seen:
            continue
        seen.add(doc.id)
        unique.append(doc)
    return unique


def _docs_by_section() -> dict[str, list[PortfolioDocument]]:
    docs = load_portfolio_documents()
    grouped: dict[str, list[PortfolioDocument]] = {}
    for doc in docs:
        grouped.setdefault(doc.section, []).append(doc)
    return grouped


def _detect_intent(question: str) -> str | None:
    q = question.lower()
    if any(token in q for token in ["contact", "reach", "email", "phone", "linkedin", "connect"]):
        return "contact"
    if any(token in q for token in ["hire", "why should we hire", "fit for", "right candidate"]):
        return "hire"
    if _is_project_question(question):
        return "projects"
    if any(token in q for token in ["intern", "internship", "experience", "work", "job"]):
        return "experience"
    if any(token in q for token in ["skill", "tech", "technology", "stack", "language"]):
        return "skills"
    if any(token in q for token in ["education", "college", "school", "degree", "cgpa"]):
        return "education"
    if any(token in q for token in ["achievement", "award", "certification", "hackathon"]):
        return "achievements"
    if any(token in q for token in ["about", "who", "background", "introduce", "introduction"]):
        return "about"
    return None


def _is_experience_question(question: str) -> bool:
    q = question.lower()
    return any(
        token in q
        for token in [
            "intern",
            "internship",
            "experience",
            "work experience",
            "work history",
            "job",
        ]
    )


def _extract_markdown_section(content: str, heading: str) -> str:
    """Extract body text under a markdown H3 heading until section delimiter."""
    match = re.search(
        rf"###\s*{re.escape(heading)}\s*\n+(.+?)(?:\n---\s*\n|\Z)",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _clean_duration(duration: str) -> str:
    if not duration:
        return duration
    return (
        duration.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
        .strip()
    )


def _build_experience_answer(docs: list[PortfolioDocument]) -> tuple[str, list[PortfolioDocument]] | None:
    internships = [
        doc for doc in docs if doc.section == "experience" and "intern" in doc.title.lower()
    ]
    if not internships:
        return None

    lines = ["She has completed these internships:", ""]
    for doc in internships:
        org = _extract_field(doc.content, "Organization") or "Organization not specified"
        duration = _clean_duration(_extract_field(doc.content, "Duration")) or "Duration not specified"
        body = _strip_chunk_metadata(doc.content)
        description = _extract_markdown_section(body, "Description")
        summary = _first_meaningful_line(description) if description else _first_meaningful_line(doc.content)
        lines.append(f"- **{doc.title}** at {org} ({duration}): {_truncate_text(summary, 180)}")

    return _normalize_pronouns("\n".join(lines)), internships


def _build_projects_answer(question: str, docs: list[PortfolioDocument]) -> tuple[str, list[PortfolioDocument]] | None:
    projects = [doc for doc in docs if doc.section == "projects"]
    if not projects:
        return None

    q = question.lower()
    matched = _top_project_matches(projects, question)
    is_which_project = "which project" in q or q.startswith("which ")
    asks_tech = any(token in q for token in ["tech", "technology", "technologies", "stack", "used"])
    asks_challenges = any(token in q for token in ["challenge", "difficult", "problem", "issue"])
    asks_outcomes = any(token in q for token in ["outcome", "result", "impact", "achieve"])
    asks_single_details = any(phrase in q for phrase in ["tell me about", "explain", "describe"]) and "projects" not in q

    if asks_challenges:
        selected = matched[:4] if matched else projects[:4]
        lines = ["These are the key project challenges she faced:", ""]
        for doc in selected:
            body = _strip_chunk_metadata(doc.content)
            challenges = _extract_markdown_section(body, "Challenges")
            challenge_points = _extract_bullets(challenges, limit=2)
            if challenge_points:
                detail = "; ".join(_truncate_text(point, 90) for point in challenge_points)
            else:
                detail = _truncate_text(_first_meaningful_line(challenges or doc.content), 130)
            lines.append(f"- **{doc.title}**: {detail}")
        return _normalize_pronouns("\n".join(lines)), selected

    if asks_outcomes:
        selected = matched[:4] if matched else projects[:4]
        lines = ["These are the main outcomes from her projects:", ""]
        for doc in selected:
            body = _strip_chunk_metadata(doc.content)
            outcomes = _extract_markdown_section(body, "Outcomes")
            outcome_points = _extract_bullets(outcomes, limit=2)
            if outcome_points:
                detail = "; ".join(_truncate_text(point, 95) for point in outcome_points)
            else:
                detail = _truncate_text(_first_meaningful_line(outcomes or doc.content), 130)
            lines.append(f"- **{doc.title}**: {detail}")
        return _normalize_pronouns("\n".join(lines)), selected

    if is_which_project:
        selected = matched[:3]
        if not selected:
            return (
                "I could not find a clear project match for that question in the portfolio data.",
                [],
            )
        heading = "The project that matches this best is:" if len(selected) == 1 else "The projects that match this are:"
        lines = [heading, ""]
        for doc in selected:
            body = _strip_chunk_metadata(doc.content)
            tech_stack = _extract_markdown_section(body, "Tech Stack")
            techs = _extract_bullets(tech_stack, limit=3)
            detail = f" Key tech: {', '.join(techs)}" if techs else ""
            lines.append(f"- **{doc.title}**.{detail}")
        return _normalize_pronouns("\n".join(lines)), selected

    if asks_single_details and matched:
        selected = [matched[0]]
    elif asks_tech and matched:
        selected = matched[:2]
    elif matched:
        selected = matched[:3]
    else:
        selected = projects[:3]

    if len(selected) == 1:
        doc = selected[0]
        body = _strip_chunk_metadata(doc.content)
        description = _extract_markdown_section(body, "Description")
        tech_stack = _extract_markdown_section(body, "Tech Stack")
        summary = _first_meaningful_line(description) if description else _first_meaningful_line(doc.content)
        techs = _extract_bullets(tech_stack, limit=4)
        lines = [f"Here is a focused summary of **{doc.title}**:", ""]
        lines.append(f"- {_truncate_text(summary, 180)}")
        if techs:
            lines.append(f"- Key tech: {', '.join(techs)}")
        return _normalize_pronouns("\n".join(lines)), selected

    lines = ["Here are the most relevant projects for your question:", ""]
    for doc in selected:
        body = _strip_chunk_metadata(doc.content)
        description = _extract_markdown_section(body, "Description")
        tech_stack = _extract_markdown_section(body, "Tech Stack")
        summary = _first_meaningful_line(description) if description else _first_meaningful_line(doc.content)
        techs = _extract_bullets(tech_stack, limit=3)
        if techs:
            lines.append(
                f"- **{doc.title}**: {_truncate_text(summary, 170)} Key tech: {', '.join(techs)}"
            )
        else:
            lines.append(f"- **{doc.title}**: {_truncate_text(summary, 180)}")

    return _normalize_pronouns("\n".join(lines)), selected


def _contains_term(text: str, term: str) -> bool:
    """Case-insensitive term match that also works for terms like C++ and HTML/CSS."""
    return bool(re.search(rf"(?<!\\w){re.escape(term)}(?!\\w)", text, flags=re.IGNORECASE))


def _build_skills_answer(docs: list[PortfolioDocument]) -> tuple[str, list[PortfolioDocument]] | None:
    skills_docs = [doc for doc in docs if doc.section == "skills"]
    if not skills_docs:
        return None

    combined = "\n".join(_strip_chunk_metadata(doc.content) for doc in skills_docs)
    catalog = {
        "Languages": ["C++", "Python", "Java", "JavaScript", "SQL", "HTML/CSS"],
        "AI/ML": ["Machine Learning", "Computer Vision", "RAG", "Embeddings", "Prompt Engineering"],
        "Database and Developer Tools": [
            "MongoDB",
            "MySQL",
            "PostgreSQL",
            "Git",
            "VSCode",
            "Docker",
            "AWS",
        ],
        "Frameworks and Libraries": [
            "React.js",
            "Node.js",
            "Tailwind CSS",
            "REST API",
            "REST APIs",
            "React Native",
            "FastAPI",
            "Express.js",
        ],
        "Relevant Coursework": [
            "Database Management",
            "Software Engineering",
            "Data Structures and Algorithms",
            "Operating Systems",
            "Computer Networks",
        ],
    }

    lines = ["Her technical strengths include:", ""]
    for heading, terms in catalog.items():
        present: list[str] = []
        for term in terms:
            if _contains_term(combined, term):
                present.append(term)
        if present:
            lines.append(f"- **{heading}**: {', '.join(present[:6])}")

    if len(lines) <= 2:
        # Fallback for unusual formatting in skills markdown
        lines.extend([f"- {doc.title}" for doc in skills_docs[:5]])

    return _normalize_pronouns("\n".join(lines)), skills_docs[:5]


def _build_education_answer(docs: list[PortfolioDocument]) -> tuple[str, list[PortfolioDocument]] | None:
    education_docs = [doc for doc in docs if doc.section == "education"]
    if not education_docs:
        return None

    lines = ["Her education background is:", ""]
    for doc in education_docs[:4]:
        content = _strip_chunk_metadata(doc.content)
        institution = _extract_field(content, "Institution")
        degree = _extract_field(content, "Degree")
        field = _extract_field(content, "Field")
        status = _extract_field(content, "Status")

        percentage_10 = _extract_field(content, "10th Percentage")
        percentage_12 = _extract_field(content, "12th Percentage")

        if doc.title.lower().startswith("college"):
            detail = f"{degree} in {field}".strip()
            extra = f" ({status})" if status else ""
            lines.append(f"- **{doc.title}**: {institution} - {detail}{extra}")
            continue

        if percentage_12:
            lines.append(f"- **{doc.title}**: {institution} - 12th Percentage: {percentage_12}")
            continue

        if percentage_10:
            lines.append(f"- **{doc.title}**: {institution} - 10th Percentage: {percentage_10}")
            continue

        lines.append(f"- **{doc.title}**: {_truncate_text(_first_meaningful_line(content), 110)}")

    return _normalize_pronouns("\n".join(lines)), education_docs[:4]


def _build_achievements_answer(docs: list[PortfolioDocument]) -> tuple[str, list[PortfolioDocument]] | None:
    achievements_docs = [doc for doc in docs if doc.section == "achievements"]
    if not achievements_docs:
        return None

    selected = achievements_docs[:5]
    lines = ["Here are her notable achievements:", ""]
    for doc in selected:
        summary = _best_summary_line(doc.content)
        lines.append(f"- **{doc.title}**: {_truncate_text(summary, 120)}")

    return _normalize_pronouns("\n".join(lines)), selected


def _build_about_answer(docs: list[PortfolioDocument]) -> tuple[str, list[PortfolioDocument]] | None:
    about_docs = [doc for doc in docs if doc.section == "about me"]
    if not about_docs:
        return None

    basic = next((doc for doc in about_docs if "basic information" in doc.title.lower()), None)
    intro = next((doc for doc in about_docs if "short introduction" in doc.title.lower()), None)
    interests = next((doc for doc in about_docs if "areas of interest" in doc.title.lower()), None)

    name = _extract_field(basic.content, "Name") if basic else "Hetavi Modi"
    degree = _extract_field(basic.content, "Degree") if basic else ""
    location = _extract_field(basic.content, "Location") if basic else ""
    cgpa = _extract_field(basic.content, "CGPA") if basic else ""
    intro_line = _first_meaningful_line(intro.content) if intro else ""
    interest_items = _extract_bullets(_strip_chunk_metadata(interests.content), limit=3) if interests else []

    lines = [
        f"{name} is a {degree} student based in {location}. She has a CGPA of {cgpa} and a strong practical focus on building real-world systems.",
        "",
    ]

    if intro_line:
        lines.append(f"- {_truncate_text(intro_line, 140)}")
    if interest_items:
        lines.append(f"- Key interests: {', '.join(interest_items)}")

    used = [doc for doc in [basic, intro, interests] if doc is not None]
    return _normalize_pronouns("\n".join(lines)), used


def _build_contact_answer(docs: list[PortfolioDocument]) -> tuple[str, list[PortfolioDocument]]:
    about_docs = [doc for doc in docs if doc.section == "about me"]
    basic = next((doc for doc in about_docs if "basic information" in doc.title.lower()), None)
    location = _extract_field(basic.content, "Location") if basic else "Pune, Maharashtra, India"

    lines = [
        "I do not have direct public contact fields in the current knowledge base, but here is the best available info:",
        "",
        f"- Location: {location}",
        "- You can connect with her through her professional portfolio mode and public profiles.",
    ]
    used = [basic] if basic else []
    return _normalize_pronouns("\n".join(lines)), used


def _build_hire_answer(docs: list[PortfolioDocument]) -> tuple[str, list[PortfolioDocument]] | None:
    """Create a focused "why hire" response using portfolio facts."""
    about_docs = [doc for doc in docs if doc.section == "about me"]
    skills_docs = [doc for doc in docs if doc.section == "skills"]
    experience_docs = [doc for doc in docs if doc.section == "experience" and "intern" in doc.title.lower()]
    project_docs = [doc for doc in docs if doc.section == "projects"]

    basic = next((doc for doc in about_docs if "basic information" in doc.title.lower()), None)
    strengths_doc = next((doc for doc in about_docs if "strengths" in doc.title.lower()), None)
    interests_doc = next((doc for doc in about_docs if "areas of interest" in doc.title.lower()), None)

    name = _extract_field(basic.content, "Name") if basic else "Hetavi Modi"
    degree = _extract_field(basic.content, "Degree") if basic else "Computer Engineering"
    cgpa = _extract_field(basic.content, "CGPA") if basic else ""

    interests = _extract_bullets(_strip_chunk_metadata(interests_doc.content), limit=5) if interests_doc else []
    strengths = _extract_bullets(_strip_chunk_metadata(strengths_doc.content), limit=3) if strengths_doc else []

    internship_bits: list[str] = []
    for doc in experience_docs[:2]:
        org = _extract_field(doc.content, "Organization")
        if org:
            internship_bits.append(f"{doc.title} at {org}")
        else:
            internship_bits.append(doc.title)

    project_titles = [doc.title for doc in project_docs[:3]]
    skills_blob = "\n".join(_strip_chunk_metadata(doc.content) for doc in skills_docs)
    stack = [
        term
        for term in ["Python", "Java", "JavaScript", "FastAPI", "Node.js", "React.js", "RAG", "Machine Learning"]
        if _contains_term(skills_blob, term)
    ]

    lines = [
        f"{name} is a strong hire because she combines solid technical depth with hands-on delivery across AI and software engineering.",
        "",
    ]

    if cgpa:
        lines.append(f"- **Strong fundamentals:** {degree} with a CGPA of {cgpa}.")
    if stack:
        lines.append(f"- **Relevant stack:** {', '.join(stack[:6])}.")
    if internship_bits:
        lines.append(f"- **Practical exposure:** {', '.join(internship_bits)}.")
    if project_titles:
        lines.append(f"- **Execution track record:** Built projects like {', '.join(project_titles)}.")
    if interests:
        lines.append(f"- **Growth direction:** Focused on {', '.join(interests[:4])}.")
    if strengths:
        lines.append(f"- **Work style:** {', '.join(strengths)}.")

    used_docs = [doc for doc in [basic, strengths_doc, interests_doc] if doc is not None]
    used_docs.extend(experience_docs[:2])
    used_docs.extend(project_docs[:2])
    return _normalize_pronouns("\n".join(lines)), _unique_docs(used_docs)


def _build_structured_answer(question: str) -> tuple[str | None, list[PortfolioDocument]]:
    """Deterministic, concise answers for most common question types."""
    intent = _detect_intent(question)
    if not intent:
        return None, []

    docs = load_portfolio_documents()

    if intent == "experience":
        result = _build_experience_answer(docs)
        return result if result else (None, [])
    if intent == "projects":
        result = _build_projects_answer(question, docs)
        return result if result else (None, [])
    if intent == "skills":
        result = _build_skills_answer(docs)
        return result if result else (None, [])
    if intent == "education":
        result = _build_education_answer(docs)
        return result if result else (None, [])
    if intent == "achievements":
        result = _build_achievements_answer(docs)
        return result if result else (None, [])
    if intent == "about":
        result = _build_about_answer(docs)
        return result if result else (None, [])
    if intent == "hire":
        result = _build_hire_answer(docs)
        return result if result else (None, [])
    if intent == "contact":
        return _build_contact_answer(docs)

    return None, []


def _stream_text_chunks(text: str, words_per_chunk: int = 3) -> Generator[str, None, None]:
    """Yield small text chunks to preserve typing-style streaming in UI."""
    words = re.findall(r"\S+\s*", text)
    for idx in range(0, len(words), words_per_chunk):
        yield "".join(words[idx : idx + words_per_chunk])


def _source_payload_from_docs(docs: list[PortfolioDocument]) -> list[dict]:
    payload: list[dict] = []
    for doc in _unique_docs(docs)[:5]:
        payload.append(
            {
                "id": doc.id,
                "title": doc.title,
                "section": doc.section,
                "score": 1.0,
            }
        )
    return payload


def _source_models_from_docs(docs: list[PortfolioDocument]) -> list[Source]:
    return [
        Source(id=item["id"], title=item["title"], section=item["section"], score=item["score"])
        for item in _source_payload_from_docs(docs)
    ]


def _infer_owner_name(chunks: list[str]) -> str:
    """Infer portfolio owner name from retrieved context if available."""
    for chunk in chunks:
        match = re.search(r"Name:\s*([A-Za-z ]{2,60})", chunk)
        if match:
            return match.group(1).strip()
    return "the portfolio owner"

def _resolve_provider() -> tuple[str, list[str], str | None]:
    """Resolve provider URL/model from environment.

    Supports both:
    - Groq keys in GROQ_API_KEY (usually gsk_...)
    - xAI Grok keys in GROQ_API_KEY or XAI_API_KEY (usually xai-...)
    """
    key = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")
    if not key:
        return "", [], None

    if key.startswith("xai-"):
        configured = os.getenv("XAI_MODEL")
        models = [configured] if configured else [
            "grok-3-mini",
            "grok-3-mini-fast",
            "grok-2-latest",
        ]
        return "https://api.x.ai/v1/chat/completions", [m for m in models if m], key

    configured = os.getenv("GROQ_MODEL")
    models = [configured] if configured else [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    ]
    return "https://api.groq.com/openai/v1/chat/completions", [m for m in models if m], key


def _build_system_prompt(owner_name: str, context: str, user_name: str | None = None) -> str:
    """Build the system prompt for the LLM."""
    return f"""You are {owner_name}'s personal AI portfolio assistant. You speak warmly and confidently about her.

STRICT RULES:
1. Pronoun policy: ALWAYS use she/her pronouns. NEVER use he/him.
2. Answer ONLY using the provided context below. Do NOT invent information.
3. NEVER start with meta-phrases like "Based on my portfolio", "According to the context", "Here's what I found", "About Hetavi:". Start directly with the answer content.
4. If the context does not contain the answer, say so honestly.
5. NEVER dump all the raw context into the response. Summarize and present naturally.

RESPONSE STYLE:
- Conversational and natural, like a knowledgeable friend describing {owner_name}.
- STRICT LENGTH: 60-150 words maximum. Be concise and impactful. Do NOT exceed this.
- If the user message is a greeting/self-introduction, reply in 1-2 short sentences and ask what they want to know about Hetavi.
- For list-type answers (skills, projects, internships), use a brief 1-sentence intro followed by markdown bullet points.
- Each bullet point on its own line, starting with "- ".
- Max 5 bullet points. Each bullet should be 1-2 lines max.
- Do NOT list every technology or responsibility. Pick the 2-3 most important for each item.
- If user name is available, you may address them by first name once naturally.

QUESTION-SPECIFIC GUIDELINES:
- Internships/Experience: List ALL internships with **Role** at Organization (Duration) and 1-line summary. Keep each entry to 1-2 lines.
- Projects: Name, one-line description, and key technologies. Max 4-5 projects.
- Skills: Group skills into categories (e.g., "AI/ML: Python, TensorFlow, scikit-learn"). Max 4-5 categories.
- About/Who: Warm summary of who she is, her field, CGPA, and standout qualities.
- Education: Institution, degree, and scores.

User Name (if known): {user_name or "Unknown"}

Context:
{context}
"""


def ask_llm(
    question: str,
    history_messages: list[dict[str, str]] | None = None,
    user_name: str | None = None,
) -> str:
    """Call provider API with RAG context to answer user questions."""

    url, model_names, api_key = _resolve_provider()
    if not api_key:
        return None

    section_filter, top_k = _detect_sections(question)
    context_chunks = search(question, top_k=top_k, sections=section_filter)
    owner_name = _infer_owner_name(context_chunks)

    context = "\n\n".join(context_chunks)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    system_prompt = _build_system_prompt(owner_name, context, user_name=user_name)
    for model_name in model_names:
        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            messages.extend(history_messages[-6:])
        messages.append({"role": "user", "content": question})

        data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 400,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return _normalize_pronouns(result["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"LLM request failed for model '{model_name}': {e}")

    return None


def stream_llm(
    question: str,
    history_messages: list[dict[str, str]] | None = None,
    user_name: str | None = None,
) -> Generator[str, None, None]:
    """Stream tokens from the LLM response."""
    url, model_names, api_key = _resolve_provider()
    if not api_key:
        return

    section_filter, top_k = _detect_sections(question)
    context_chunks = search(question, top_k=top_k, sections=section_filter)
    owner_name = _infer_owner_name(context_chunks)
    context = "\n\n".join(context_chunks)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = _build_system_prompt(owner_name, context, user_name=user_name)

    for model_name in model_names:
        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            messages.extend(history_messages[-6:])
        messages.append({"role": "user", "content": question})

        data = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "max_tokens": 400,
        }

        try:
            resp = requests.post(
                url, headers=headers, json=data, stream=True, timeout=60
            )
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    token = chunk["choices"][0].get("delta", {}).get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            return  # successfully consumed stream
        except Exception as e:
            print(f"Streaming failed for model '{model_name}': {e}")
            continue


def _strip_chunk_metadata(content: str) -> str:
    """Remove loader metadata headers from a chunk body."""
    cleaned = re.sub(r"^Section:\s*.+$", "", content, flags=re.MULTILINE)
    cleaned = re.sub(r"^Title:\s*.+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _fallback_answer(question: str, retrieved_items: list) -> str:
    """Create concise intent-aware answers directly from retrieved chunks."""
    if not retrieved_items:
        return (
            "I could not find specific information about that yet. "
            "You can ask about projects, skills, internships, education, or achievements."
        )

    q = question.lower()

    if any(token in q for token in ["intern", "internship", "experience", "work"]):
        internship_docs = [
            item for item in retrieved_items
            if "intern" in item.document.title.lower() or "intern" in item.document.content.lower()
        ]
        if internship_docs:
            lines = ["Here is a concise summary of her internship experience:", ""]
            for item in internship_docs[:4]:
                org = _extract_field(item.document.content, "Organization")
                duration = _extract_field(item.document.content, "Duration")
                summary = _first_meaningful_line(item.document.content)
                detail = ""
                if org or duration:
                    detail = f" ({org}{', ' if org and duration else ''}{duration})"
                lines.append(f"- {item.document.title}{detail}: {summary[:130]}{'...' if len(summary) > 130 else ''}")
            return _normalize_pronouns("\n".join(lines))

    if any(token in q for token in ["project", "built", "build"]):
        lines = ["Here are her key projects:", ""]
        for item in retrieved_items[:4]:
            summary = _first_meaningful_line(item.document.content)
            lines.append(f"- {item.document.title}: {summary[:130]}{'...' if len(summary) > 130 else ''}")
        return _normalize_pronouns("\n".join(lines))

    if any(token in q for token in ["skill", "tech", "technology", "stack", "language"]):
        lines = ["Her core skills include:", ""]
        seen: set[str] = set()
        for item in retrieved_items:
            title = item.document.title.strip()
            if title.lower() in seen:
                continue
            seen.add(title.lower())
            lines.append(f"- {title}")
            if len(lines) >= 6:
                break
        return _normalize_pronouns("\n".join(lines))

    lines = [
        "I could not access the live model right now, but here is what I can share from my portfolio data:",
        "",
    ]

    for item in retrieved_items[:3]:
        snippet = _strip_chunk_metadata(item.document.content)
        snippet = snippet.replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = snippet[:180].rstrip() + "..."
        lines.append(f"- {item.document.title}: {snippet}")

    lines.append("")
    lines.append(f"Question asked: {question}")
    return _normalize_pronouns("\n".join(lines))


class ChatService:
    """Service class to handle chat requests using RAG + LLM."""

    def __init__(self) -> None:
        self._session_store: dict[str, dict] = {}
        self._lock = RLock()
        self._max_history_turns = 6

    def _get_session_id(self, request: ChatRequest) -> str:
        session_id = (request.session_id or "").strip()
        return session_id or "default"

    def _ensure_session(self, session_id: str) -> dict:
        with self._lock:
            if session_id not in self._session_store:
                self._session_store[session_id] = {
                    "user_name": None,
                    "history": [],
                }
            return self._session_store[session_id]

    def _session_user_name(self, session_id: str) -> str | None:
        with self._lock:
            state = self._session_store.get(session_id, {})
            user_name = state.get("user_name")
            return user_name if isinstance(user_name, str) and user_name.strip() else None

    def _set_session_user_name(self, session_id: str, user_name: str) -> None:
        with self._lock:
            state = self._ensure_session(session_id)
            state["user_name"] = user_name

    def _history_for_llm(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            state = self._ensure_session(session_id)
            raw_history = state.get("history") or []
            turns = raw_history[-self._max_history_turns :]

        messages: list[dict[str, str]] = []
        for turn in turns:
            user_text = turn.get("user")
            assistant_text = turn.get("assistant")
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def _remember_turn(self, session_id: str, question: str, answer: str, intent: str) -> None:
        with self._lock:
            state = self._ensure_session(session_id)
            history = state.setdefault("history", [])
            history.append(
                {
                    "user": question,
                    "assistant": answer,
                    "intent": intent,
                }
            )
            state["history"] = history[-self._max_history_turns :]

    def answer(self, request: ChatRequest) -> ChatResponse:
        session_id = self._get_session_id(request)
        self._ensure_session(session_id)
        question = request.message.strip()
        user_name = _extract_user_name(question)
        if user_name:
            self._set_session_user_name(session_id, user_name)

        known_user_name = self._session_user_name(session_id)
        opening_reply = _build_opening_reply(question, known_user_name)
        if opening_reply:
            self._remember_turn(session_id, question, opening_reply, "conversation")
            return ChatResponse(answer=opening_reply, sources=[])

        section_filter, top_k = _detect_sections(question)
        intent = _detect_intent(question) or "general"

        # Deterministic structured answer for most common intents
        structured_answer, structured_docs = _build_structured_answer(question)
        if structured_answer:
            self._remember_turn(session_id, question, structured_answer, intent)
            sources = _source_models_from_docs(structured_docs)
            return ChatResponse(answer=structured_answer, sources=sources)

        # Get sources for the response
        retrieved = search_with_metadata(question, top_k=top_k, sections=section_filter)
        sources = [
            Source(
                id=item.document.id,
                title=item.document.title,
                section=item.document.section,
                score=round(item.score, 4)
            )
            for item in retrieved[:5]
        ]

        # Get answer from LLM
        llm_answer = ask_llm(
            question,
            history_messages=self._history_for_llm(session_id),
            user_name=known_user_name,
        )

        # Fallback if LLM fails
        if not llm_answer:
            llm_answer = _fallback_answer(question, retrieved)
        else:
            llm_answer = _normalize_pronouns(llm_answer)

        self._remember_turn(session_id, question, llm_answer, intent)

        return ChatResponse(answer=llm_answer, sources=sources)

    def stream_answer(self, request: ChatRequest) -> Generator[str, None, None]:
        """Yield SSE events for a streamed response."""
        session_id = self._get_session_id(request)
        self._ensure_session(session_id)
        question = request.message.strip()
        user_name = _extract_user_name(question)
        if user_name:
            self._set_session_user_name(session_id, user_name)

        known_user_name = self._session_user_name(session_id)
        opening_reply = _build_opening_reply(question, known_user_name)
        if opening_reply:
            self._remember_turn(session_id, question, opening_reply, "conversation")
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            for chunk in _stream_text_chunks(opening_reply):
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        section_filter, top_k = _detect_sections(question)
        intent = _detect_intent(question) or "general"

        # Deterministic structured answer for most common intents
        structured_answer, structured_docs = _build_structured_answer(question)
        if structured_answer:
            self._remember_turn(session_id, question, structured_answer, intent)
            structured_sources = _source_payload_from_docs(structured_docs)
            yield f"data: {json.dumps({'type': 'sources', 'sources': structured_sources})}\n\n"
            for chunk in _stream_text_chunks(structured_answer):
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        retrieved = search_with_metadata(question, top_k=top_k, sections=section_filter)
        sources = [
            {
                "id": item.document.id,
                "title": item.document.title,
                "section": item.document.section,
                "score": round(item.score, 4),
            }
            for item in retrieved[:5]
        ]

        # Send sources as the first SSE event
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream LLM tokens
        has_tokens = False
        full_text = ""
        for token in stream_llm(
            question,
            history_messages=self._history_for_llm(session_id),
            user_name=known_user_name,
        ):
            has_tokens = True
            full_text += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        if not has_tokens:
            fallback = _fallback_answer(question, retrieved)
            full_text = fallback
            yield f"data: {json.dumps({'type': 'token', 'token': fallback})}\n\n"

        stored_answer = _normalize_pronouns(full_text) if full_text else ""
        if stored_answer:
            self._remember_turn(session_id, question, stored_answer, intent)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"