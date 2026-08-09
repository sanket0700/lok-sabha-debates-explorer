"""Stage 2: PDF -> raw text.

English-only sittings are almost all digital-native, so plain text-layer
extraction (PyMuPDF) covers those directly. Any sitting with meaningful
Hindi content always goes through OCR instead (ocrmypdf, English+Hindi
Tesseract models), regardless of which font the PDF embeds or how "clean"
the extracted Devanagari looks. This is a deliberate, hard-won choice, not
the obvious one: text-layer extraction of Devanagari has turned out to be
unreliable across a wide and unpredictable range of fonts — confirmed
broken for two *named* legacy fonts (LEGACY_HINDI_FONTS) that substitute
Latin/symbol glyphs for Devanagari wholesale, but *also* broken for at
least one anonymous CID-subset font (seen in LS18 sittings, generic
"CIDFont+FN" names carrying no identifying signal) that produces mostly-
correct Devanagari with specific silently-wrong matra substitutions, and
even for a font whose corruption slipped past initial spot-checking
entirely (LS16 sitting 5 — flagged as "clean" only because a check tuned
to one font's specific corruption signature doesn't generalize). Chasing
each new font by name or by content-ratio heuristics (both tried, both
missed real cases) doesn't scale; OCR reads the rendered glyphs directly
and sidesteps the whole class of problem. The cost is real — OCR is far
slower than text-layer extraction — but correctness on Hindi content isn't
something a font-detection allowlist can be trusted to guarantee. Porting
the old per-font glyph-remap tables instead (see
legacy/Dataset creation/Aryan2_to_Unicode_map.ipynb) was considered and
rejected for the same reason: fragile, one table per font, and there
isn't a way to know how many fonts this corpus actually contains.

Idempotent: only processes sittings where extracted_at IS NULL.

Usage:
    python -m pipeline.extract [--lok-sabha 16] [--limit 50]
"""
import argparse
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import requests

from pipeline.config import DATA_DIR
from pipeline.db import execute, fetch_all, get_conn

log = logging.getLogger("pipeline.extract")

# Below this average chars/page, assume the PDF has no usable text layer.
MIN_CHARS_PER_PAGE = 40

# Any sitting with at least this many real Devanagari codepoints in its
# text-layer extraction is treated as having meaningful Hindi content, and
# always gets OCR'd — see the module docstring for why detection can't stop
# at known-bad font names. Deliberately low: a false positive just costs
# some OCR time on an English-majority sitting with a short Hindi quote;
# a false negative silently ships corrupted Hindi.
MIN_DEVANAGARI_CHARS_FOR_OCR = 100
DEVANAGARI_PATTERN = re.compile(r"[ऀ-ॿ]")

# Case-insensitive substrings of embedded font basefont names known to be
# legacy non-Unicode Hindi fonts (confirmed present in LS16 pilot PDFs via
# fitz page.get_fonts()) — extend this list if other sittings turn up more.
# "aryan2" confirmed via 3 LS16 sittings; "utsaah" confirmed via a 4th
# (lsd_16_17_31-01-2019.pdf, the President's Address) whose garbled text
# initially slipped through this check entirely since it isn't Aryan2 — a
# reminder that this list is necessarily incomplete and any sitting could
# turn up yet another legacy font.
LEGACY_HINDI_FONTS = ("aryan2", "utsaah")

# Homebrew's tesseract-lang ships tessdata_fast, which measurably
# under-performs tessdata_best on Devanagari (confirmed by hand: ~25% fewer
# stray-glyph insertions across a 10-page sample of real Hindi debate pages
# — see rebuild plan Phase 3 OCR-quality notes). Downloaded once into
# DATA_DIR (gitignored) rather than overwriting the Homebrew-managed files.
TESSDATA_BEST_DIR = DATA_DIR / "tessdata_best"
TESSDATA_BEST_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main"
TESSDATA_BEST_LANGS = ("hin", "eng")

# Devanagari OCR (both fast and best tessdata) has a well-known failure mode
# around pre-base vowel signs (matras that render to the left of the
# consonant they modify), producing two distinct kinds of noise — confirmed
# by hand against real LS16 OCR output:
#  1. Tesseract's best (wrong) guess pulled from its full trained character
#     set instead of just eng+hin: Polish/Czech/Turkish accented letters
#     (Œ, Þ, Ě, Ū, ň, Ø...) and C1 control characters, which never
#     legitimately appear in an English/Hindi transcript — stripped by an
#     explicit allowlist rather than chasing each character individually.
#  2. An otherwise-ordinary ASCII character (digit, letter, punctuation)
#     wedged mid-word between two Devanagari characters with no spacing
#     (e.g. "सुवि)धा" instead of "सुविधा") — individually legitimate, but
#     wrong given where it landed, so this one has to be positional.
# Neither fix recovers the lost matra, but both keep the noise out of the
# text that gets fed to translation, which is confused far worse by
# embedded garbage than by a slightly-off spelling.
ALLOWED_OCR_CHARS = re.compile(r"[^\x09\x0A\x0D\x20-\x7Eऀ-ॿ‌‍₹]")
STRAY_MIDWORD_ASCII = re.compile(r"([ऀ-ॿ])[^\sऀ-ॿ]{1,2}(?=[ऀ-ॿ])")


def clean_ocr_text(text: str) -> str:
    text = ALLOWED_OCR_CHARS.sub("", text)
    return STRAY_MIDWORD_ASCII.sub(r"\1", text)


def _default_tessdata_dir() -> Path | None:
    """Ask tesseract itself where its bundled tessdata lives (configs/,
    tessconfigs/, pdf.ttf) — portable across Homebrew/Linux/etc, unlike
    hardcoding a path."""
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=15
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    m = re.search(r'"([^"]+)"', result.stdout)
    return Path(m.group(1)) if m else None


def ensure_tessdata_best() -> Path:
    TESSDATA_BEST_DIR.mkdir(parents=True, exist_ok=True)
    for lang in TESSDATA_BEST_LANGS:
        dest = TESSDATA_BEST_DIR / f"{lang}.traineddata"
        if dest.exists():
            continue
        log.info("downloading tessdata_best/%s.traineddata (one-time)...", lang)
        resp = requests.get(f"{TESSDATA_BEST_URL}/{lang}.traineddata", timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)

    # tesseract/ocrmypdf also need configs/, tessconfigs/, and pdf.ttf from
    # the default install (hocr/pdf output configs, the font used to render
    # the invisible OCR text layer) — symlink those in alongside the better
    # language models rather than only overriding the .traineddata files.
    default_dir = _default_tessdata_dir()
    if default_dir is not None:
        for name in ("configs", "tessconfigs", "pdf.ttf"):
            src = default_dir / name
            dst = TESSDATA_BEST_DIR / name
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

    return TESSDATA_BEST_DIR


def _format_table(rows: list[list]) -> str:
    lines = []
    for row in rows:
        cells = [str(c).replace("\n", " ").strip() for c in row if c is not None]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_page_text(page: fitz.Page) -> str:
    """Text extraction that keeps government annexure tables (state-wise
    budget figures, RTI/audit-report entries — common in "Written Answers"
    sittings, confirmed present in LS18's largest sittings) in correct
    reading order.

    Plain page.get_text() flattens table cells by raw block position, which
    for genuinely tabular content produces two distinct failure modes
    confirmed by hand: a wrapped footnote fragment sorting ahead of a
    same-row column physically to its right, and word-by-word-wrapped
    narrow columns flattening into unreadable word-salad with no coherent
    order at all (~17-20% of pages in LS18's two largest sittings, sampled).
    find_tables() does structural table detection (not just block position)
    and resolves both — only used on pages where it actually finds a table;
    normal prose pages go through the plain path unchanged.
    """
    tables = page.find_tables()
    if not tables.tables:
        return page.get_text()

    table_bboxes = [fitz.Rect(t.bbox) for t in tables.tables]

    def _mostly_inside_a_table(rect: fitz.Rect) -> bool:
        for tb in table_bboxes:
            if not tb.intersects(rect):
                continue
            if rect.get_area() > 0 and (tb & rect).get_area() / rect.get_area() > 0.5:
                return True
        return False

    # (y0, x0, text) so a normal top-to-bottom/left-to-right sort places
    # each table's formatted text at the position it visually occupies,
    # instead of at wherever its individual constituent blocks happened to
    # land.
    pieces = []
    for b in page.get_text("blocks"):
        rect = fitz.Rect(b[:4])
        if _mostly_inside_a_table(rect):
            continue
        if b[4].strip():
            pieces.append((b[1], b[0], b[4]))

    for table, bbox in zip(tables.tables, table_bboxes):
        formatted = _format_table(table.extract())
        if formatted.strip():
            pieces.append((bbox.y0, bbox.x0, formatted + "\n"))

    pieces.sort(key=lambda p: (p[0], p[1]))
    return "\n".join(p[2] for p in pieces)


def extract_text_layer(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(_extract_page_text(page) for page in doc)
    finally:
        doc.close()


def has_legacy_hindi_font(pdf_path: Path) -> bool:
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            for font in page.get_fonts():
                basefont = font[3].lower()
                if any(name in basefont for name in LEGACY_HINDI_FONTS):
                    return True
        return False
    finally:
        doc.close()


def needs_ocr(pdf_path: Path, text: str) -> bool:
    doc = fitz.open(pdf_path)
    try:
        page_count = max(doc.page_count, 1)
    finally:
        doc.close()
    if (len(text) / page_count) < MIN_CHARS_PER_PAGE:
        return True
    # Catches partial-corruption fonts (some real Devanagari extracted
    # alongside wrong matra substitutions) and any future unknown font.
    if len(DEVANAGARI_PATTERN.findall(text)) >= MIN_DEVANAGARI_CHARS_FOR_OCR:
        return True
    # Named-font fallback: catches full-substitution fonts (Aryan2-style)
    # whose corrupted output contains zero real Devanagari codepoints to
    # count above, so the content-based check alone can't see them.
    return has_legacy_hindi_font(pdf_path)


def ocr_pdf(pdf_path: Path) -> Path:
    """Run ocrmypdf (English + Hindi, tessdata_best) and return the OCR'd copy's path."""
    tessdata_dir = ensure_tessdata_best()
    ocred_path = Path(tempfile.gettempdir()) / f"ocr_{pdf_path.stem}.pdf"
    subprocess.run(
        [
            "ocrmypdf",
            "--force-ocr",
            "--deskew",
            "--language",
            "eng+hin",
            str(pdf_path),
            str(ocred_path),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "TESSDATA_PREFIX": str(tessdata_dir)},
    )
    return ocred_path


def extract_sitting(pdf_path_str: str) -> str:
    pdf_path = Path(pdf_path_str)
    text = extract_text_layer(pdf_path)
    if needs_ocr(pdf_path, text):
        log.info("no usable text layer, running OCR: %s", pdf_path)
        ocred = ocr_pdf(pdf_path)
        text = clean_ocr_text(extract_text_layer(ocred))
    return text


def run(lok_sabha_number: int | None, limit: int | None):
    with get_conn() as conn:
        query = "select id, source_pdf_path from sittings where extracted_at is null"
        params = []
        if lok_sabha_number is not None:
            query += " and lok_sabha_number = %s"
            params.append(lok_sabha_number)
        query += " order by id"
        if limit is not None:
            query += " limit %s"
            params.append(limit)

        rows = fetch_all(conn, query, params)
        log.info("extracting text for %d sittings", len(rows))

        for row in rows:
            try:
                text = extract_sitting(row["source_pdf_path"])
            except Exception:
                log.exception("failed to extract sitting %s", row["id"])
                continue
            execute(
                conn,
                "update sittings set raw_text = %s, extracted_at = now() where id = %s",
                (text, row["id"]),
            )
            conn.commit()
            log.info("sitting %s: %d chars extracted", row["id"], len(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lok-sabha", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run(args.lok_sabha, args.limit)


if __name__ == "__main__":
    main()
