-- Lok Sabha debates rebuild: schema for the pilot (16th & 17th Lok Sabha).
--
-- Single datastore doing triple duty: structured metadata, full-text search
-- (tsvector/GIN), and semantic search (pgvector). No separate search engine
-- or vector DB needed at pilot scale.
--
-- Idempotency convention used by the pipeline scripts (see pipeline/run.py):
-- each stage looks for rows/columns that are still NULL/absent rather than
-- relying on a separate job-tracking table, e.g. "speeches with
-- text_english IS NULL" is exactly the translate-stage backlog.

create extension if not exists vector;
create extension if not exists pg_trgm;

-- Embedding dimension must match pipeline/config.py's EMBEDDING_DIM
-- (384 for the default local model, BAAI/bge-small-en). Change both
-- together if the embedding model changes.

create table if not exists sittings (
    id                  bigserial primary key,
    lok_sabha_number    smallint not null,
    sitting_date        date,
    source_pdf_url      text not null unique,
    source_pdf_path     text,               -- local/object-storage path once downloaded
    scraped_at          timestamptz,
    extracted_at        timestamptz,        -- raw text pulled out of the pdf
    segmented_at        timestamptz,        -- raw text split into speeches
    raw_text            text,               -- full extracted text, pre-segmentation
    created_at          timestamptz not null default now()
);

create index if not exists idx_sittings_ls_number on sittings (lok_sabha_number);
create index if not exists idx_sittings_date on sittings (sitting_date);

create table if not exists speeches (
    id                  bigserial primary key,
    sitting_id          bigint not null references sittings(id) on delete cascade,
    sequence_index      integer not null,          -- order within the sitting
    speaker_raw         text,                       -- as printed in the transcript, unresolved
    -- Resolved identity (pipeline/speakers.py): honorific/possessive-stripped,
    -- and a bare role attribution ("The Prime Minister") resolved to the
    -- person holding that role on sitting_date where known. speaker_raw is
    -- kept untouched alongside it for display/transparency — this mirrors
    -- entities.text vs entities.canonical_text.
    speaker_canonical   text,
    language            text not null check (language in ('en', 'hi', 'mixed')),
    text_original       text not null,
    text_english        text,                       -- null until pipeline/translate.py runs
    source_page         integer,
    -- NER and topic tagging can legitimately produce zero rows for a speech
    -- (no entities found / no topic clears the confidence threshold), so a
    -- child-row-exists check can't tell "not processed yet" from "processed,
    -- nothing found". These two columns disambiguate that; sentiment and
    -- embeddings don't need the same treatment since every processed speech
    -- always gets exactly one sentiment row and one embedding row.
    ner_processed_at    timestamptz,
    topics_processed_at timestamptz,
    search_tsv          tsvector generated always as (
                            to_tsvector('english', coalesce(text_english, text_original))
                        ) stored,
    created_at          timestamptz not null default now(),
    unique (sitting_id, sequence_index)
);

create index if not exists idx_speeches_sitting on speeches (sitting_id);
create index if not exists idx_speeches_speaker on speeches (speaker_raw);
create index if not exists idx_speeches_search_tsv on speeches using gin (search_tsv);
create index if not exists idx_speeches_speaker_trgm on speeches using gin (speaker_raw gin_trgm_ops);

create table if not exists entities (
    id                  bigserial primary key,
    speech_id           bigint not null references speeches(id) on delete cascade,
    text                text not null,              -- entity span as extracted
    type                text not null,               -- PERSON / ORG / GPE / LOC / DATE / ...
    canonical_text      text not null,               -- lowercased/trimmed, used for trend grouping
    created_at          timestamptz not null default now()
);

create index if not exists idx_entities_speech on entities (speech_id);
create index if not exists idx_entities_canonical on entities (canonical_text, type);

create table if not exists locations (
    entity_id           bigint primary key references entities(id) on delete cascade,
    formatted_address   text not null,
    lat                 double precision not null,
    lon                 double precision not null,
    geocoded_at         timestamptz not null default now()
);

create table if not exists topics (
    id                  serial primary key,
    name                text not null unique          -- fixed taxonomy, see pipeline/config.py
);

create table if not exists speech_topics (
    speech_id           bigint not null references speeches(id) on delete cascade,
    topic_id            integer not null references topics(id) on delete cascade,
    score                real not null,                -- zero-shot classifier confidence
    primary key (speech_id, topic_id)
);

create index if not exists idx_speech_topics_topic on speech_topics (topic_id);

create table if not exists sentiment (
    speech_id           bigint primary key references speeches(id) on delete cascade,
    score               real not null,                 -- normalized -1..1 (negative..positive)
    label               text not null                   -- negative / neutral / positive
);

create table if not exists speech_embeddings (
    speech_id           bigint primary key references speeches(id) on delete cascade,
    embedding           vector(384) not null
);

create index if not exists idx_speech_embeddings_hnsw
    on speech_embeddings using hnsw (embedding vector_cosine_ops);

-- Precomputed aggregates the Insights dashboards read from directly.
-- Rebuilt wholesale by pipeline/aggregate.py; cheap enough at pilot scale
-- to just truncate + repopulate rather than incrementally maintain.

create table if not exists topic_trends (
    topic_id            integer not null references topics(id) on delete cascade,
    lok_sabha_number    smallint not null,
    period_start        date not null,                 -- month bucket
    mention_count        integer not null,
    avg_score            real not null,
    primary key (topic_id, lok_sabha_number, period_start)
);

create table if not exists entity_trends (
    canonical_text       text not null,
    entity_type          text not null,
    lok_sabha_number     smallint not null,
    period_start         date not null,
    mention_count         integer not null,
    primary key (canonical_text, entity_type, lok_sabha_number, period_start)
);

create table if not exists sentiment_trends (
    lok_sabha_number     smallint not null,
    period_start         date not null,
    avg_score             real not null,
    speech_count          integer not null,
    -- Per-label counts, not just avg_score: a monthly average can be
    -- dominated by whichever handful of sittings happen to land in that
    -- bucket (a ceremonial address skews strongly positive, an adversarial
    -- debate day skews strongly negative) and opposite-signed speeches
    -- within a bucket can cancel out in the mean. The Insights "tone" chart
    -- shows %composition (positive/neutral/negative) instead of a single
    -- averaged line specifically to avoid losing that distribution.
    positive_count        integer not null default 0,
    neutral_count         integer not null default 0,
    negative_count        integer not null default 0,
    primary key (lok_sabha_number, period_start)
);

create table if not exists speaker_stats (
    -- Keyed by resolved identity (pipeline/speakers.py), not the raw printed
    -- label, so e.g. "Shri Narendra Modi" / "The Prime Minister" / "Modi ji"
    -- roll up into one cumulative row per real person instead of one row
    -- per alias. display_name is the most-frequent speaker_raw variant for
    -- that identity, kept purely so the UI shows a properly-cased human name
    -- rather than the lowercased canonical key.
    speaker_canonical     text not null,
    lok_sabha_number      smallint not null,
    display_name          text not null,
    speech_count           integer not null,
    avg_sentiment           real,
    top_topics              jsonb,                       -- [{"topic": "...", "count": n}, ...]
    primary key (speaker_canonical, lok_sabha_number)
);
