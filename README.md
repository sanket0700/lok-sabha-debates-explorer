# Lok Sabha Debates Explorer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A rebuild of the original *Analysis of Lok Sabha Dataset* project: a bilingual
(Hindi/English), speech-level corpus of Lok Sabha debates, with two ways to use it —
**Insights** (topic, entity, and sentiment/tone trends over time) and **Explore**
(hybrid search + retrieval-augmented Q&A over the corpus).

**Pilot scope**: 16th, 17th, and 18th Lok Sabha (18th prioritized first — newer sittings use
proper Unicode fonts and need far less OCR/legacy-font work). Everything runs on local/
self-hosted models — no paid LLM API. See `PROGRESS.md` for the full running decision/reasoning
log (read that first if you're picking this project back up), `CLAUDE.md` for the current
architecture, and `legacy/README.md` for the original 2021 notebook pipeline this replaces.

## Architecture

```
scrape (elibrary.sansad.in DSpace REST API, LS18+LS17+LS16)
   → extract raw text (PyMuPDF; OCR fallback for image-only pages)
   → segment into speeches (heuristics + local-LLM fallback via Ollama)
   → translate Hindi→English (IndicTrans2, local)
   → enrich (NER, geocode, topic tag, sentiment) — all on the English text
   → embed (local sentence-embedding model) → Postgres + pgvector
   → aggregate (precomputed trend tables)
   → serve via FastAPI: Explore (search + RAG chat) and Insights (dashboards)
```

- `pipeline/` — one script per stage, each idempotent (safe to re-run; only processes
  what isn't done yet). `pipeline/run.py` is the single entrypoint.
- `db/schema.sql` — Postgres + pgvector schema (loaded automatically by docker-compose).
- `app/` — FastAPI app (Jinja2 + HTMX templates, Chart.js/Leaflet for Insights).
- `legacy/` — the original 2021 Colab-notebook pipeline, kept for reference.

## Running it

### Recommended on macOS (Apple Silicon): native pipeline/app + Ollama, Dockerized Postgres only

Docker Desktop on macOS can't pass the GPU through to a Linux container, so the pipeline's
local models (translation, topic/sentiment classifiers, embeddings) and Ollama get real Metal
acceleration only when run natively. `pipeline/config.get_device()` auto-picks `mps` on Apple
Silicon (falls back to `cuda`, then `cpu` elsewhere).

```bash
docker compose up -d db                       # Postgres+pgvector on host port 5433 (not the default
                                                # 5432, in case that's already in use); schema auto-loads

python3 -m venv .venv && source .venv/bin/activate   # use Python 3.12, not a brand-new release —
pip install -r requirements.txt                      # ML-package wheel availability lags behind
python -m spacy download en_core_web_trf

brew install ollama
brew services start ollama                    # or: ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M       # used for segmentation fallback + RAG synthesis

python -m pipeline.run status                 # sanity check: DB reachable, schema loaded

# Pilot-safe: dry-run first, then a small real batch before scaling up. LS18 is scraped by
# default first (config.LOK_SABHA_NUMBERS priority) if you omit --lok-sabha.
python -m pipeline.run scrape --lok-sabha 18 --limit 5 --dry-run
python -m pipeline.run scrape --lok-sabha 18 --limit 5
python -m pipeline.run extract segment translate ner geocode topic sentiment embed aggregate --lok-sabha 18 --limit 5

uvicorn app.main:app --reload                 # http://localhost:8000
```

### Fully Dockerized (Linux, or macOS without GPU acceleration)

```bash
docker compose --profile docker-ollama --profile pipeline up -d db ollama
docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M

docker compose run --rm pipeline python -m pipeline.run status
docker compose run --rm pipeline python -m pipeline.run scrape --lok-sabha 18 --limit 5 --dry-run
docker compose run --rm pipeline python -m pipeline.run scrape --lok-sabha 18 --limit 5
docker compose run --rm pipeline python -m pipeline.run extract segment translate ner geocode topic sentiment embed aggregate --lok-sabha 18 --limit 5

# app's OLLAMA_HOST defaults to host.docker.internal; override to http://ollama:11434 if using this path
docker compose up -d app                      # http://localhost:8000
```

`python -m pipeline.run status` shows how many sittings/speeches are at each stage.
Once a small pilot batch looks right (spot-check a few segmented speeches against
the source PDF — see `PROGRESS.md`'s Verification notes), drop `--limit` and
scrape/process the rest of LS16, LS17, and LS18.

## Known limitations (see `PROGRESS.md` and `CLAUDE.md` for detail)

- Speaker/party rhetoric comparison is speaker-level only — no MP→party mapping yet.
- Older Lok Sabhas (pre-16th) are out of scope; scaling to them needs heavier OCR and the
  Aryan2 legacy-font decoding logic from `legacy/Dataset creation/Aryan2_to_Unicode_map.ipynb`
  reintroduced.
- `pipeline/scrape.py` targets `elibrary.sansad.in`'s DSpace 7 REST API (the original
  `eparlib.nic.in` it mirrored was retired and migrated there) — the collection scope UUID and
  `loksabhanumber` facet filter were confirmed by hand against the live API, but could still
  drift if the site is restructured; verify with `--dry-run --limit 5` before a real run. Bounded
  retry-with-backoff plus an end-of-run failure summary are in place for transient errors.
- Segmentation's LLM fallback (for genuinely ambiguous, unlabeled-boundary text) costs real
  time — measured at ~140s/call for `llama3.1:8b-instruct-q4_K_M` on a ~6000-character chunk.
  A structured accuracy audit (55-boundary sample) found and fixed real regex-coverage gaps and
  an LLM-hallucination failure mode; fallback usage on the pilot corpus is now near-zero.
- Geocoding resolves ~87% of extracted place-like entities; the remainder is a mix of generic
  non-place phrases, real places Nominatim's free-text search doesn't match well, and residual
  NER/OCR noise — confirmed by direct inspection, not assumed. A persistent `geocode_cache` table
  means re-runs don't re-fetch already-resolved (or already-confirmed-unresolvable) places.
- Sentiment/topic tagging use real tokenizer-based truncation (not a character-count proxy) —
  even so, roughly 22% of speeches (sentiment) / 13% (topic) still exceed each model's native
  context window and get truncated; chunk-and-aggregate scoring for long speeches is a real,
  not-yet-done improvement.
- Translation (IndicTrans2, greedy decoding to fit consumer-hardware memory) is generally
  accurate on real spot-checks but has a known degenerate-repetition failure mode on short
  repeated phrases, mitigated (not eliminated) with a repetition penalty + n-gram blocking.
- The pilot corpus (15 sittings across LS16/17/18, ~2500 speeches) is a proof of concept, not
  yet a citable source — see `PROGRESS.md` for the full, honest accounting of what's been
  measured and what hasn't at this scale.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for how to get set up, the
branch/PR workflow, and where to look first (the "Known limitations" list above and
`PROGRESS.md`'s ranked issue list are good starting points). This project follows the
[Contributor Covenant](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).
