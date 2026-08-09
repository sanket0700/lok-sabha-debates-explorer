"""Thin client for the local Ollama instruct model.

Used sparingly: segment.py's fallback for ambiguous speaker boundaries, and
the app's RAG answer-synthesis endpoint. Both are the only two places in
this project that touch an LLM rather than a smaller task-specific model —
kept local/self-hosted throughout, no paid API.
"""
import json
import logging

import requests

from pipeline.config import OLLAMA_HOST, OLLAMA_MODEL

log = logging.getLogger("pipeline.llm")

# Measured by hand against a real ~6000-char segmentation chunk: a genuine,
# successful llama3.1:8b-instruct-q4_K_M JSON-mode response took ~138s
# (grammar-constrained decoding of a ~2000-token response). The old 120s
# default was *shorter* than typical real latency — most calls were timing
# out by design, not failing. Confirmed against a real batch run: 37/48
# segmentation LLM calls timed out at 120s (visible content silently lost
# per timed-out chunk, since a failed call just contributes zero speeches).
DEFAULT_TIMEOUT_SECONDS = 300


def generate_json(prompt: str, model: str = OLLAMA_MODEL, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    """Call Ollama's /api/generate with format=json and return the parsed object.

    Returns None (rather than raising) on any failure — callers should treat
    that as "fall back to the non-LLM path" so a down/missing Ollama doesn't
    hard-fail a batch job.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "format": "json", "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["response"])
    except (requests.RequestException, KeyError, json.JSONDecodeError) as exc:
        log.warning("ollama generate_json failed: %s", exc)
        return None


def generate_text(prompt: str, model: str = OLLAMA_MODEL, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str | None:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except (requests.RequestException, KeyError) as exc:
        log.warning("ollama generate_text failed: %s", exc)
        return None
