"""Insights dashboard: reads the precomputed trend tables built by
pipeline/aggregate.py — never scans raw speeches at request time.

GET /insights   full page: topic-over-time, top entities/places (+ map),
                 sentiment-over-time, and a per-speaker table.
"""
import json

from fastapi import APIRouter, Request

from app.templates_env import templates
from pipeline.config import LOK_SABHA_NUMBERS
from pipeline.db import fetch_all, get_conn

router = APIRouter(prefix="/insights", tags=["insights"])

TOP_TOPICS_SHOWN = 8


def _topic_chart_json(rows):
    periods = sorted({r["period_start"] for r in rows})
    totals: dict[str, int] = {}
    for r in rows:
        totals[r["topic"]] = totals.get(r["topic"], 0) + r["mention_count"]
    top_topics = sorted(totals, key=totals.get, reverse=True)[:TOP_TOPICS_SHOWN]

    by_topic_period = {(r["topic"], r["period_start"]): r["mention_count"] for r in rows}
    datasets = [
        {"label": topic, "data": [by_topic_period.get((topic, p), 0) for p in periods]}
        for topic in top_topics
    ]
    return json.dumps({"labels": [p.isoformat() for p in periods], "datasets": datasets})


def _sentiment_chart_json(rows):
    """%positive/neutral/negative composition per period, not an averaged score.

    A single averaged line hides more than it shows on this corpus: monthly
    buckets are often dominated by just one or two sittings at pilot scale (a
    ceremonial address skews strongly positive, an adversarial debate day
    skews strongly negative), and even with more data, positive and negative
    speeches within the same bucket can average out to a misleadingly "neutral"
    number. Composition preserves that distribution instead of collapsing it.
    """
    periods = sorted({r["period_start"] for r in rows})
    totals_by_period = {p: {"positive": 0, "neutral": 0, "negative": 0} for p in periods}
    for r in rows:
        bucket = totals_by_period[r["period_start"]]
        bucket["positive"] += r["positive_count"]
        bucket["neutral"] += r["neutral_count"]
        bucket["negative"] += r["negative_count"]

    def pct(period, label):
        bucket = totals_by_period[period]
        total = bucket["positive"] + bucket["neutral"] + bucket["negative"]
        return round(100 * bucket[label] / total, 1) if total else 0

    datasets = [
        {"label": "Positive", "key": "positive", "data": [pct(p, "positive") for p in periods]},
        {"label": "Neutral", "key": "neutral", "data": [pct(p, "neutral") for p in periods]},
        {"label": "Negative", "key": "negative", "data": [pct(p, "negative") for p in periods]},
    ]
    return json.dumps({"labels": [p.isoformat() for p in periods], "datasets": datasets})


def _rhetoric_chart_json(rows):
    """%composition per period across the 5 rhetorical-mode labels — same
    reasoning as _sentiment_chart_json: a monthly bucket's real composition
    (e.g. 40% critical / 35% supportive / 25% procedural) is the signal,
    not a single dominant/averaged label that would hide it.
    """
    periods = sorted({r["period_start"] for r in rows})
    labels = ["supportive", "critical", "constructive", "confrontational", "procedural"]
    totals_by_period = {p: {l: 0 for l in labels} for p in periods}
    for r in rows:
        bucket = totals_by_period[r["period_start"]]
        for l in labels:
            bucket[l] += r[f"{l}_count"]

    def pct(period, label):
        bucket = totals_by_period[period]
        total = sum(bucket.values())
        return round(100 * bucket[label] / total, 1) if total else 0

    datasets = [
        {"label": l.capitalize(), "key": l, "data": [pct(p, l) for p in periods]}
        for l in labels
    ]
    return json.dumps({"labels": [p.isoformat() for p in periods], "datasets": datasets})


def _entity_chart_json(rows):
    return json.dumps(
        {
            "labels": [f"{r['canonical_text']} ({r['entity_type']})" for r in rows],
            "data": [r["total"] for r in rows],
        }
    )


@router.get("")
def insights_page(request: Request, lok_sabha: int | None = None):
    with get_conn() as conn:
        topic_rows = fetch_all(
            conn,
            """
            select t.name as topic, tt.lok_sabha_number, tt.period_start, tt.mention_count
            from topic_trends tt join topics t on t.id = tt.topic_id
            where %(ls)s is null or tt.lok_sabha_number = %(ls)s
            order by tt.period_start
            """,
            {"ls": lok_sabha},
        )
        entity_rows = fetch_all(
            conn,
            """
            select canonical_text, entity_type, sum(mention_count) as total
            from entity_trends
            where %(ls)s is null or lok_sabha_number = %(ls)s
            group by canonical_text, entity_type
            order by total desc
            limit 20
            """,
            {"ls": lok_sabha},
        )
        sentiment_rows = fetch_all(
            conn,
            """
            select lok_sabha_number, period_start, avg_score, speech_count,
                   positive_count, neutral_count, negative_count
            from sentiment_trends
            where %(ls)s is null or lok_sabha_number = %(ls)s
            order by period_start
            """,
            {"ls": lok_sabha},
        )
        rhetoric_rows = fetch_all(
            conn,
            """
            select lok_sabha_number, period_start, speech_count,
                   supportive_count, critical_count, constructive_count,
                   confrontational_count, procedural_count
            from rhetoric_trends
            where %(ls)s is null or lok_sabha_number = %(ls)s
            order by period_start
            """,
            {"ls": lok_sabha},
        )
        speaker_rows = fetch_all(
            conn,
            """
            select display_name, lok_sabha_number, speech_count, avg_sentiment, top_topics
            from speaker_stats
            where %(ls)s is null or lok_sabha_number = %(ls)s
            order by speech_count desc
            limit 25
            """,
            {"ls": lok_sabha},
        )
        place_rows = fetch_all(
            conn,
            """
            select l.formatted_address, l.lat, l.lon, count(*) as mentions
            from locations l
            join entities e on e.id = l.entity_id
            join speeches sp on sp.id = e.speech_id
            join sittings si on si.id = sp.sitting_id
            where %(ls)s is null or si.lok_sabha_number = %(ls)s
            group by l.formatted_address, l.lat, l.lon
            order by mentions desc
            limit 300
            """,
            {"ls": lok_sabha},
        )

    return templates.TemplateResponse(
        request,
        "insights.html",
        {
            "lok_sabha": lok_sabha,
            "lok_sabha_numbers": LOK_SABHA_NUMBERS,
            "topic_chart_json": _topic_chart_json(topic_rows),
            "sentiment_chart_json": _sentiment_chart_json(sentiment_rows),
            "rhetoric_chart_json": _rhetoric_chart_json(rhetoric_rows),
            "entity_chart_json": _entity_chart_json(entity_rows),
            "place_rows_json": json.dumps(
                [dict(lat=r["lat"], lon=r["lon"], mentions=r["mentions"], address=r["formatted_address"]) for r in place_rows]
            ),
            "speaker_rows": speaker_rows,
            "has_data": bool(topic_rows or entity_rows or sentiment_rows or rhetoric_rows or speaker_rows),
        },
    )
