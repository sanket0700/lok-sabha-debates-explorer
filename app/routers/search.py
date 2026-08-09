"""Explore: hybrid keyword+semantic search over speeches, with filters.

GET /explore            full page (HTMX-driven search box + filters)
GET /explore/search      HTMX partial: just the results list, re-rendered per keystroke/filter change
"""
from fastapi import APIRouter, Query, Request

from app.retrieval import Filters, hybrid_search, keyword_search, semantic_search
from app.templates_env import templates
from pipeline.db import get_conn

router = APIRouter(prefix="/explore", tags=["explore"])

MODES = {"hybrid": hybrid_search, "keyword": keyword_search, "semantic": semantic_search}


@router.get("")
def explore_page(request: Request):
    return templates.TemplateResponse(request, "explore.html", {})


@router.get("/search")
def search_partial(
    request: Request,
    q: str = "",
    mode: str = "hybrid",
    lok_sabha: int | None = None,
    speaker: str | None = None,
    language: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
):
    results = []
    if q.strip():
        search_fn = MODES.get(mode, hybrid_search)
        filters = Filters(
            lok_sabha=lok_sabha, speaker=speaker, language=language, date_from=date_from, date_to=date_to
        )
        with get_conn() as conn:
            results = search_fn(conn, q.strip(), filters, limit=25)

    return templates.TemplateResponse(
        request, "_results.html", {"results": results, "q": q, "searched": bool(q.strip())}
    )
