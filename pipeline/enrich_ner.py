"""Stage 5a: named entity recognition over text_english.

One model for both source languages (spaCy's transformer English NER
pipeline), run on the post-translation English text — see translate.py's
docstring for why. This deliberately replaces the old repo's fragmented
Stanza(English)/DeepPavlov(Hindi, failed)/Flair(Hindi, experimental) setup
with a single well-maintained pipeline.

Idempotent: only processes speeches where ner_processed_at IS NULL.

Usage:
    python -m pipeline.enrich_ner [--lok-sabha 16] [--limit 500]
"""
import argparse
import logging

from pipeline.config import NER_MODEL
from pipeline.db import execute, fetch_all, get_conn
from pipeline.text_normalize import strip_person_honorifics

log = logging.getLogger("pipeline.enrich_ner")


def canonicalize(text: str, entity_type: str) -> str:
    """Lowercase/trim; for PERSON entities also strip honorific prefixes/
    suffixes and a trailing possessive (see pipeline/text_normalize.py) so the
    same person doesn't get split across salutation variants — confirmed in
    practice: "Shri Narendra Modi", "Narendra Modi", "Modi ji", and "Narendra
    Modi's" were landing as four separate canonical_text values. Deliberately
    NOT attempting to merge a bare surname/short name into a full name (e.g.
    "Modi" vs "Narendra Modi", "Gandhi" vs "Rahul Gandhi") — several surnames
    in this corpus belong to more than one real, distinct MP (confirmed:
    "Nirav Modi" is a different, real person from "Narendra Modi"), so that
    merge can't be done safely by string matching alone; see
    pipeline/backfill_entity_canonicalization.py for the one-off fix applied
    to entities extracted before this existed.
    """
    if entity_type == "PERSON":
        return strip_person_honorifics(text)
    return text.strip().lower()


_nlp = None


def _load():
    global _nlp
    if _nlp is not None:
        return _nlp
    import spacy

    log.info("loading NER model %s (first call only)...", NER_MODEL)
    _nlp = spacy.load(NER_MODEL)
    return _nlp


def run(lok_sabha_number: int | None, limit: int | None, batch_size: int = 32):
    nlp = _load()
    with get_conn() as conn:
        query = (
            "select id, text_english from speeches "
            "where text_english is not null and ner_processed_at is null"
        )
        params = []
        if lok_sabha_number is not None:
            query += " and sitting_id in (select id from sittings where lok_sabha_number = %s)"
            params.append(lok_sabha_number)
        query += " order by id"
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("running NER on %d speeches", len(rows))

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            docs = nlp.pipe([r["text_english"] for r in batch])
            for row, doc in zip(batch, docs):
                for ent in doc.ents:
                    execute(
                        conn,
                        """
                        insert into entities (speech_id, text, type, canonical_text)
                        values (%s, %s, %s, %s)
                        """,
                        (row["id"], ent.text, ent.label_, canonicalize(ent.text, ent.label_)),
                    )
                execute(
                    conn, "update speeches set ner_processed_at = now() where id = %s", (row["id"],)
                )
            conn.commit()
            log.info("NER: %d/%d", min(i + batch_size, len(rows)), len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lok-sabha", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args.lok_sabha, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
