"""Stage 6: rebuild the precomputed trend tables the Insights dashboards read.

Cheap enough at pilot scale to truncate + fully repopulate rather than
incrementally maintain — run this after any enrichment batch, or on a
schedule once the pipeline is live.

Usage:
    python -m pipeline.aggregate
"""
import argparse
import logging

from pipeline.config import ENTITY_TYPES_FOR_TRENDS
from pipeline.db import get_conn

log = logging.getLogger("pipeline.aggregate")

TOPIC_TRENDS_SQL = """
truncate topic_trends;
insert into topic_trends (topic_id, lok_sabha_number, period_start, mention_count, avg_score)
select st.topic_id, si.lok_sabha_number, date_trunc('month', si.sitting_date)::date,
       count(*), avg(st.score)
from speech_topics st
join speeches sp on sp.id = st.speech_id
join sittings si on si.id = sp.sitting_id
where si.sitting_date is not null
group by st.topic_id, si.lok_sabha_number, date_trunc('month', si.sitting_date);
"""

ENTITY_TRENDS_SQL = """
truncate entity_trends;
insert into entity_trends (canonical_text, entity_type, lok_sabha_number, period_start, mention_count)
select e.canonical_text, e.type, si.lok_sabha_number, date_trunc('month', si.sitting_date)::date,
       count(*)
from entities e
join speeches sp on sp.id = e.speech_id
join sittings si on si.id = sp.sitting_id
where si.sitting_date is not null and e.type = any(%(entity_types)s)
group by e.canonical_text, e.type, si.lok_sabha_number, date_trunc('month', si.sitting_date);
"""

SENTIMENT_TRENDS_SQL = """
truncate sentiment_trends;
insert into sentiment_trends
    (lok_sabha_number, period_start, avg_score, speech_count,
     positive_count, neutral_count, negative_count)
select si.lok_sabha_number, date_trunc('month', si.sitting_date)::date,
       avg(sen.score), count(*),
       count(*) filter (where sen.label = 'positive'),
       count(*) filter (where sen.label = 'neutral'),
       count(*) filter (where sen.label = 'negative')
from sentiment sen
join speeches sp on sp.id = sen.speech_id
join sittings si on si.id = sp.sitting_id
where si.sitting_date is not null
group by si.lok_sabha_number, date_trunc('month', si.sitting_date);
"""

SPEAKER_STATS_SQL = """
truncate speaker_stats;
with topic_counts as (
    select sp.speaker_canonical, si.lok_sabha_number, t.name as topic, count(*) as cnt
    from speeches sp
    join sittings si on si.id = sp.sitting_id
    join speech_topics st on st.speech_id = sp.id
    join topics t on t.id = st.topic_id
    where sp.speaker_canonical is not null
    group by sp.speaker_canonical, si.lok_sabha_number, t.name
),
ranked_topics as (
    select *, row_number() over (
        partition by speaker_canonical, lok_sabha_number order by cnt desc
    ) as rn
    from topic_counts
),
top_topics as (
    select speaker_canonical, lok_sabha_number,
           jsonb_agg(jsonb_build_object('topic', topic, 'count', cnt) order by cnt desc) as top_topics
    from ranked_topics
    where rn <= 5
    group by speaker_canonical, lok_sabha_number
),
base as (
    select sp.speaker_canonical, si.lok_sabha_number, count(*) as speech_count, avg(sen.score) as avg_sentiment
    from speeches sp
    join sittings si on si.id = sp.sitting_id
    left join sentiment sen on sen.speech_id = sp.id
    where sp.speaker_canonical is not null
    group by sp.speaker_canonical, si.lok_sabha_number
)
insert into speaker_stats
    (speaker_canonical, lok_sabha_number, display_name, speech_count, avg_sentiment, top_topics)
-- initcap(), not "the most common speaker_raw variant" — a role-title label
-- ("The Prime Minister") can outnumber the person's actual name in the raw
-- data (confirmed: LS17 has 36 "The Prime Minister" vs. 12 "Shri Narendra
-- Modi"), and the whole point of resolving it was to show the *person*, not
-- whichever printed label happened to be more common.
select b.speaker_canonical, b.lok_sabha_number, initcap(b.speaker_canonical), b.speech_count, b.avg_sentiment, tt.top_topics
from base b
left join top_topics tt
  on tt.speaker_canonical = b.speaker_canonical and tt.lok_sabha_number = b.lok_sabha_number;
"""


def run():
    with get_conn() as conn:
        with conn.cursor() as cur:
            for name, sql, params in [
                ("topic_trends", TOPIC_TRENDS_SQL, None),
                ("entity_trends", ENTITY_TRENDS_SQL, {"entity_types": list(ENTITY_TYPES_FOR_TRENDS)}),
                ("sentiment_trends", SENTIMENT_TRENDS_SQL, None),
                ("speaker_stats", SPEAKER_STATS_SQL, None),
            ]:
                log.info("rebuilding %s...", name)
                cur.execute(sql, params)
        conn.commit()
    log.info("done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run()


if __name__ == "__main__":
    main()
