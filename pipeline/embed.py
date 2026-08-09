"""Stage 5e: sentence embeddings for semantic search + RAG retrieval.

Local sentence-embedding model (config.EMBEDDING_MODEL, default a small BGE
English model — dimension must match db/schema.sql's speech_embeddings
column, see config.EMBEDDING_DIM). Passages are embedded as-is; the BGE
query-side instruction prefix ("Represent this sentence for searching
relevant passages: ") belongs at *query* time in the app, not here.

Idempotent: only processes speeches with no row yet in `speech_embeddings`.

Usage:
    python -m pipeline.embed [--lok-sabha 16] [--limit 500]
"""
import argparse
import logging

from pipeline.config import EMBEDDING_DIM, EMBEDDING_MODEL, get_device
from pipeline.db import execute, fetch_all, get_conn

log = logging.getLogger("pipeline.embed")

_model = None


def load_model():
    """Return the cached embedding model, loading it on first call.

    Public on purpose: the app's semantic-search/RAG retrieval reuses this
    exact model (and cache) instead of loading a second copy — see
    app/retrieval.py.
    """
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer

    device = get_device()
    log.info("loading embedding model %s on device=%s (first call only)...", EMBEDDING_MODEL, device)
    _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    assert _model.get_sentence_embedding_dimension() == EMBEDDING_DIM, (
        f"{EMBEDDING_MODEL} produces {_model.get_sentence_embedding_dimension()}-dim vectors, "
        f"but db/schema.sql expects {EMBEDDING_DIM}. Update both together."
    )
    return _model


def run(lok_sabha_number: int | None, limit: int | None, batch_size: int = 64):
    model = load_model()
    with get_conn() as conn:
        query = """
            select s.id, s.text_english from speeches s
            left join speech_embeddings e on e.speech_id = s.id
            where s.text_english is not null and e.speech_id is null
        """
        params = []
        if lok_sabha_number is not None:
            query += " and s.sitting_id in (select id from sittings where lok_sabha_number = %s)"
            params.append(lok_sabha_number)
        query += " order by s.id"
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("embedding %d speeches", len(rows))

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            vectors = model.encode([r["text_english"] for r in batch], normalize_embeddings=True)
            for row, vector in zip(batch, vectors):
                execute(
                    conn,
                    "insert into speech_embeddings (speech_id, embedding) values (%s, %s) "
                    "on conflict (speech_id) do nothing",
                    (row["id"], vector.tolist()),
                )
            conn.commit()
            log.info("embedded: %d/%d", min(i + batch_size, len(rows)), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lok-sabha", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args.lok_sabha, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
