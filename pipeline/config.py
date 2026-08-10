"""Shared configuration for every pipeline stage.

Everything here is overridable via environment variables so the same code
runs the same way locally and inside docker-compose.
"""
import functools
import os
from pathlib import Path

# --- storage -----------------------------------------------------------

DATABASE_URL = os.environ.get(
    # Host port 5433, not 5432: docker-compose.yml maps db's 5432 to host
    # 5433 to avoid clashing with unrelated Postgres containers on 5432.
    "DATABASE_URL", "postgresql://loksabha:loksabha@localhost:5433/loksabha"
)

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
PDF_DIR = DATA_DIR / "pdfs"
for _dir in (DATA_DIR, PDF_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- scrape (pilot scope) ----------------------------------------------

# eparlib.nic.in was retired and its content migrated to elibrary.sansad.in
# (DSpace 7, Angular front end backed by a JSON REST API — the old
# BeautifulSoup-on-rendered-HTML approach no longer works, see scrape.py).
ELIBRARY_BASE_URL = "https://elibrary.sansad.in"
ELIBRARY_API_URL = f"{ELIBRARY_BASE_URL}/server/api"
# UUID of the "Lok Sabha Debates (Text)" collection, scoped via discover/search's
# `scope` param + a loksabhanumber facet filter (both id and filter name confirmed
# by hand against the live API).
ELIBRARY_DEBATES_SCOPE = "b2770639-c57b-4ba3-a9f1-57761c60d195"
# Pilot scope: 16th-18th Lok Sabha (older sessions are out of scope —
# see the rebuild plan's "Known deferred items" for why older sessions need
# more OCR/legacy-font work). Ordered by priority, not chronologically: LS18
# first — newer sittings use proper Unicode-mapped fonts (confirmed: no
# Aryan2/Utsaah-style legacy font garbling seen so far), so they sidestep the
# OCR/font edge cases LS16-17 turned out to need real investment in. LS16/17
# stay in scope, just deprioritized behind LS18 rather than dropped.
LOK_SABHA_NUMBERS = [18, 17, 16]
SCRAPE_REQUEST_DELAY_SECONDS = float(os.environ.get("SCRAPE_REQUEST_DELAY_SECONDS", "1.5"))
SCRAPE_USER_AGENT = os.environ.get(
    "SCRAPE_USER_AGENT",
    "loksabha-explorer-research-bot/0.1 (+contact: set SCRAPE_CONTACT env var)",
)

# --- local models --------------------------------------------------------
# All open-source / self-hosted, per the "no paid API" decision.

TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "ai4bharat/indictrans2-indic-en-1B")
NER_MODEL = os.environ.get("NER_MODEL", "en_core_web_trf")
TOPIC_MODEL = os.environ.get("TOPIC_MODEL", "facebook/bart-large-mnli")
SENTIMENT_MODEL = os.environ.get(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384  # must match db/schema.sql's speech_embeddings.embedding column

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    """Resolve the best available torch device: MPS (Apple Silicon) > CUDA > CPU.

    Lazy import so stages that never touch torch (scrape.py, aggregate.py) don't
    pay for importing it.
    """
    override = os.environ.get("TORCH_DEVICE")
    if override:
        return override
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# --- geocoding -----------------------------------------------------------

NOMINATIM_USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT", SCRAPE_USER_AGENT)
NOMINATIM_REQUEST_DELAY_SECONDS = float(os.environ.get("NOMINATIM_REQUEST_DELAY_SECONDS", "1.1"))

# --- fixed issue taxonomy for topic tagging -------------------------------
# Deliberately small and hand-picked so trends are legible on a dashboard;
# extend this list rather than letting a discovered-topic model sprawl.

TOPIC_TAXONOMY = [
    "agriculture and farmers",
    "defence and national security",
    "economy and finance",
    "education",
    "health and public welfare",
    "employment and labour",
    "infrastructure and transport",
    "environment and climate",
    "law and judiciary",
    "foreign affairs",
    "energy and power",
    "women and child development",
    "rural development",
    "science and technology",
    "communication and information technology",
]

ENTITY_TYPES_TO_GEOCODE = {"GPE", "LOC", "FAC"}

# --- rhetorical-mode taxonomy (pipeline/enrich_rhetoric.py) ---------------
# Parliamentary speech isn't well described by generic positive/negative
# sentiment (a substantive, well-argued critique of a policy reads as
# "negative" even though it's exactly what opposition scrutiny is supposed
# to look like) — this is a single-label classification of what the speaker
# is *doing* rhetorically, not how the text sounds. Keys are the short label
# stored in the DB; values are the phrasing actually sent to the zero-shot
# classifier, which needed to be more specific than a bare adjective to
# discriminate well (confirmed by hand: "critical" alone scored high on
# almost everything, including neutral procedural text).
RHETORIC_TAXONOMY: dict[str, str] = {
    "supportive": "praising or expressing support",
    "critical": "raising a concern or criticism",
    "constructive": "proposing a constructive solution",
    "confrontational": "confrontational or attacking in tone",
    "procedural": "a neutral factual or procedural statement",
}

# Below this length, a "speech" is usually a bare interjection ("Why is that
# relevant?") with no real rhetorical content to classify — confirmed by
# hand that the classifier still returns a confident-looking score on these,
# which overstates how much signal is actually there. No label is stored at
# all below this threshold, mirroring how NER/topic tagging can legitimately
# produce nothing for a speech rather than force a low-information guess.
RHETORIC_MIN_CHARS = 40

# spaCy entity types worth showing in a "most-mentioned entities" insight — excludes
# CARDINAL/ORDINAL/DATE/TIME/PERCENT/QUANTITY/MONEY, which are numeric/temporal values
# spaCy tags as "entities" but aren't meaningful subjects for a who/what-is-discussed
# view (confirmed in practice: '1', 'today', 'first' were outranking real places/orgs
# in the top-20 entities chart).
ENTITY_TYPES_FOR_TRENDS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT",
    "WORK_OF_ART", "LAW", "PRODUCT", "LANGUAGE",
}
