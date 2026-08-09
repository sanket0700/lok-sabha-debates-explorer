"""Shared search/retrieval logic used by both the Explore search page and
the RAG Q&A endpoint — one implementation of keyword, semantic, and hybrid
(reciprocal rank fusion) search over `speeches`.
"""
from dataclasses import dataclass

from pipeline.db import fetch_all
from pipeline.embed import load_model

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

SPEECH_COLUMNS = """
    sp.id, sp.speaker_raw, sp.language, sp.text_original, sp.text_english,
    sp.sequence_index, si.lok_sabha_number, si.sitting_date, si.source_pdf_url
"""


@dataclass
class Filters:
    lok_sabha: int | None = None
    speaker: str | None = None
    language: str | None = None
    date_from: str | None = None
    date_to: str | None = None


def _filter_clause(filters: Filters, params: dict) -> str:
    clauses = []
    if filters.lok_sabha:
        clauses.append("si.lok_sabha_number = %(lok_sabha)s")
        params["lok_sabha"] = filters.lok_sabha
    if filters.speaker:
        clauses.append("sp.speaker_raw ilike %(speaker)s")
        params["speaker"] = f"%{filters.speaker}%"
    if filters.language:
        clauses.append("sp.language = %(language)s")
        params["language"] = filters.language
    if filters.date_from:
        clauses.append("si.sitting_date >= %(date_from)s")
        params["date_from"] = filters.date_from
    if filters.date_to:
        clauses.append("si.sitting_date <= %(date_to)s")
        params["date_to"] = filters.date_to
    return (" and " + " and ".join(clauses)) if clauses else ""


def keyword_search(conn, query: str, filters: Filters, limit: int = 20):
    params = {"q": query, "limit": limit}
    where = _filter_clause(filters, params)
    rows = fetch_all(
        conn,
        f"""
        select {SPEECH_COLUMNS}, ts_rank(sp.search_tsv, websearch_to_tsquery('english', %(q)s)) as score
        from speeches sp
        join sittings si on si.id = sp.sitting_id
        where sp.search_tsv @@ websearch_to_tsquery('english', %(q)s) {where}
        order by score desc
        limit %(limit)s
        """,
        params,
    )
    return rows


def semantic_search(conn, query: str, filters: Filters, limit: int = 20):
    model = load_model()
    query_vector = model.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True).tolist()
    params = {"qvec": query_vector, "limit": limit}
    where = _filter_clause(filters, params)
    rows = fetch_all(
        conn,
        f"""
        select {SPEECH_COLUMNS}, 1 - (e.embedding <=> %(qvec)s::vector) as score
        from speech_embeddings e
        join speeches sp on sp.id = e.speech_id
        join sittings si on si.id = sp.sitting_id
        where true {where}
        order by e.embedding <=> %(qvec)s::vector
        limit %(limit)s
        """,
        params,
    )
    return rows


def hybrid_search(conn, query: str, filters: Filters, limit: int = 20, pool_size: int = 50, rrf_k: int = 60):
    """Reciprocal rank fusion over the keyword and semantic candidate lists."""
    keyword_rows = keyword_search(conn, query, filters, pool_size)
    semantic_rows = semantic_search(conn, query, filters, pool_size)

    fused_scores: dict[int, float] = {}
    rows_by_id: dict[int, dict] = {}
    for rows in (keyword_rows, semantic_rows):
        for rank, row in enumerate(rows):
            fused_scores[row["id"]] = fused_scores.get(row["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
            rows_by_id[row["id"]] = row

    ranked_ids = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)[:limit]
    return [rows_by_id[i] for i in ranked_ids]
