"""Explore Q&A: retrieval-augmented answers grounded in retrieved speeches.

POST /explore/ask   HTMX partial: retrieve top-k speeches (hybrid search),
                     ask the local Ollama instruct model to answer using
                     only those speeches, return the answer + cited sources.
"""
from fastapi import APIRouter, Form, Request

from app.retrieval import Filters, hybrid_search
from app.templates_env import templates
from pipeline.db import get_conn
from pipeline.llm import generate_text

router = APIRouter(prefix="/explore", tags=["explore"])

RAG_PROMPT = """You are answering a question about Indian Lok Sabha parliamentary debates \
using ONLY the excerpts below. If the excerpts don't contain enough information to answer, \
say so plainly instead of guessing. Cite sources inline as [1], [2], etc. matching the excerpt \
numbers.

Question: {question}

Excerpts:
{excerpts}

Answer:"""


def _format_excerpts(rows):
    lines = []
    for i, row in enumerate(rows, start=1):
        speaker = row["speaker_raw"] or "Unknown speaker"
        date = row["sitting_date"] or "date unknown"
        text = (row["text_english"] or row["text_original"])[:1200]
        lines.append(f"[{i}] {speaker} (Lok Sabha {row['lok_sabha_number']}, {date}): {text}")
    return "\n\n".join(lines)


@router.post("/ask")
def ask(request: Request, question: str = Form(...)):
    question = question.strip()
    if not question:
        return templates.TemplateResponse(request, "_answer.html", {"asked": False})

    with get_conn() as conn:
        sources = hybrid_search(conn, question, Filters(), limit=8)

    if not sources:
        answer = "No matching speeches were found in the corpus for this question."
    else:
        prompt = RAG_PROMPT.format(question=question, excerpts=_format_excerpts(sources))
        answer = generate_text(prompt) or (
            "The local model is unavailable right now. Try again once Ollama is running "
            f"(see docker-compose.yml)."
        )

    return templates.TemplateResponse(
        request, "_answer.html", {"asked": True, "question": question, "answer": answer, "sources": sources}
    )
