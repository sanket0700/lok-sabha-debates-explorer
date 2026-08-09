"""Thin Postgres connection helper shared by every pipeline stage.

No ORM on purpose — the pipeline is a handful of batch scripts, not a web
app; raw SQL keeps each stage's "what's left to do" query legible.
"""
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from pipeline.config import DATABASE_URL


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor(conn):
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield cur
    finally:
        cur.close()


def fetch_all(conn, query, params=None):
    with get_cursor(conn) as cur:
        cur.execute(query, params or ())
        return cur.fetchall()


def fetch_one(conn, query, params=None):
    with get_cursor(conn) as cur:
        cur.execute(query, params or ())
        return cur.fetchone()


def execute(conn, query, params=None):
    with get_cursor(conn) as cur:
        cur.execute(query, params or ())
