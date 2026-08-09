# Lok Sabha Debates Explorer

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
docker compose up -d db                       # Postgres+pgvector on host port 5433 (not 5432 —
                                                # already taken by an unrelated container on the
                                                # dev machine this was built on); schema auto-loads

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
  drift if the site is restructured; verify with `--dry-run --limit 5` before a real run.
- Segmentation's LLM fallback (for oversized/ambiguous speaker-boundary matches) costs real
  time — measured at ~140s/call for `llama3.1:8b-instruct-q4_K_M` on a ~6000-character chunk.
  Full-day sittings with many such matches can need dozens of calls; budget accordingly before
  scraping/segmenting at volume.
