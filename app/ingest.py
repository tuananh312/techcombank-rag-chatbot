"""
One-time ingestion script.

Downloads (or reads locally) the Techcombank FY25 press release PDF,
splits it into overlapping chunks, embeds each chunk with Bedrock Titan
Embeddings, and writes a FAISS index + chunk metadata to /app/data so the
FastAPI app can load them at startup with zero external dependencies at
runtime.

Run this ONCE locally before you build the Docker image:

    python ingest.py --source https://techcombank.com/.../fy25-press-release-eng-12022026.pdf

Or point it at a local file:

    python ingest.py --source ./fy25-press-release.pdf

The output files (data/index.faiss, data/chunks.json) are committed to the
repo and copied into the image at build time — no live network/AWS call is
required just to *start* the container, only to answer questions.
"""

import argparse
import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
CHUNK_SIZE = 500          # characters — smaller than before, so each chunk stays
                          # close to a single idea/fact rather than a whole
                          # multi-topic paragraph, which makes citations sharper.
CHUNK_OVERLAP = 80
DATA_DIR = Path(__file__).parent / "data"


def _table_to_markdown(table: list[list]) -> str:
    """Renders a pdfplumber-extracted table as a simple markdown table, so
    the LLM (and the citation snippet shown to the user) sees clean rows/
    columns instead of pypdf's jumbled linear text-flattening of tables."""
    rows = [[(cell or "").strip() for cell in row] for row in table]
    if not rows:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


_HEADER_CELL_RE = re.compile(r"^\s*(?:[1-4]Q\d{2}|FY\d{2}|QoQ|YoY)\s*$", re.IGNORECASE)


def _is_header_like_row(row: list) -> bool:
    """A row is treated as a (repeated) header row if it contains 2+ cells
    matching quarter/fiscal-year column labels (4Q24, FY25, QoQ, YoY, ...).
    Financial report tables often repeat the header mid-table as a section
    divider (e.g. 'Balance Sheet' / 'Capital & Liquidity' / 'Profitability'
    sections each restating the same column headers) — treating only the
    very first row as the header misreads every later section-header row as
    if it were a normal data row."""
    count = sum(1 for cell in row if cell and _HEADER_CELL_RE.match(str(cell)))
    return count >= 2


def _split_compound_table(table: list[list]) -> list[list[list]]:
    """Splits a single extracted table into multiple logical sub-tables at
    each repeated header row, so each section gets its own correct header
    instead of being misread as data under the first section's header."""
    if not table:
        return []
    split_points = [i for i, row in enumerate(table) if _is_header_like_row(row)]
    if not split_points:
        return [table]
    if split_points[0] != 0:
        split_points = [0] + split_points
    subtables = []
    for idx, start in enumerate(split_points):
        end = split_points[idx + 1] if idx + 1 < len(split_points) else len(table)
        sub = table[start:end]
        if sub:
            subtables.append(sub)
    return subtables


def _is_valid_table(table: list[list]) -> bool:
    """Heuristic filter against false positives from the text-based fallback
    table-detection strategy, which can misidentify aligned prose (lists,
    indented paragraphs) as a table — cropping real prose content out of the
    page. Requires a plausible row/column shape and enough filled-in cells
    to look like genuine tabular data rather than an accidental grid."""
    if len(table) < 2:
        return False
    col_count = len(table[0])
    if col_count < 2:
        return False
    total_cells = 0
    filled_cells = 0
    for row in table:
        for cell in row:
            total_cells += 1
            if cell and str(cell).strip():
                filled_cells += 1
    if total_cells == 0:
        return False
    return (filled_cells / total_cells) >= 0.5


def load_pdf_page_units(source: str) -> list[dict]:
    """Returns a list of {"page": int, "text": str, "kind": "prose"|"table"}
    units — tables are extracted and formatted separately from prose text so
    neither one garbles the other, and each unit already knows its page."""
    if source.startswith("http"):
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        tmp_path = Path("/tmp/source.pdf")
        tmp_path.write_bytes(resp.content)
        pdf_path = str(tmp_path)
    else:
        pdf_path = source

    units = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Default line-based table detection — works when tables have
            # visible ruling lines/borders.
            found_tables = page.find_tables()
            candidates = [(t, t.extract()) for t in found_tables]

            # Fallback: borderless/whitespace-aligned tables (common in
            # financial statement appendices) have no lines for the default
            # strategy to detect. Retry with a text-alignment based strategy
            # ONLY if the default found nothing — this fallback is prone to
            # false positives on ordinary aligned prose, so its results are
            # validated below before being trusted.
            if not candidates:
                fallback_tables = page.find_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    }
                )
                candidates = [(t, t.extract()) for t in fallback_tables]

            # Only keep candidates that pass validation — rejects false
            # positives (misidentified prose) before we crop anything out
            # of the page's prose extraction.
            valid = [(t, data) for t, data in candidates if _is_valid_table(data)]
            table_bboxes = [t.bbox for t, _ in valid]

            # Split each raw detected table at repeated header rows (see
            # _split_compound_table) so a single visually-contiguous table
            # with multiple stacked sections renders as separate,
            # correctly-headered tables instead of one misread blob.
            tables = []
            for _, data in valid:
                tables.extend(_split_compound_table(data))

            if table_bboxes:
                cropped = page
                for bbox in table_bboxes:
                    cropped = cropped.outside_bbox(bbox)
                prose_text = cropped.extract_text() or ""
            else:
                prose_text = page.extract_text() or ""

            prose_text = re.sub(r"\n{2,}", "\n", prose_text)
            prose_text = re.sub(r"[ \t]{2,}", " ", prose_text)

            if len(prose_text.strip()) < 30 and not tables:
                print(f"  [diagnostic] page {page_num}: only {len(prose_text.strip())} chars of "
                      f"extractable text and no tables found — likely image/graphic-based content")

            if prose_text.strip():
                units.append({"page": page_num, "text": prose_text, "kind": "prose"})

            for table in tables:
                md = _table_to_markdown(table)
                if md.strip():
                    units.append({"page": page_num, "text": md, "kind": "table"})
                    print(f"  [diagnostic] page {page_num}: extracted table with "
                          f"{len(table)} rows x {len(table[0]) if table else 0} cols")

    return units


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def chunk_units(units: list[dict], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Prose units are split on SENTENCE boundaries and packed up to ~size
    characters (never cutting mid-sentence, so a citation snippet always
    starts at a real sentence rather than an arbitrary character offset).
    Table units are kept as single, unsplit chunks — splitting a table mid-row
    would destroy the very structure we just worked to preserve."""
    chunks = []
    for unit in units:
        if unit["kind"] == "table":
            chunks.append({"page": unit["page"], "text": unit["text"], "kind": "table"})
            continue

        sentences = _split_sentences(unit["text"])
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current_len + len(sentence) > size and current:
                chunks.append(
                    {"page": unit["page"], "text": " ".join(current), "kind": "prose"}
                )
                # carry the last sentence forward as overlap context
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sentences
                current_len = overlap_len
            current.append(sentence)
            current_len += len(sentence)
        if current:
            chunks.append({"page": unit["page"], "text": " ".join(current), "kind": "prose"})

    return chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    # Runs locally on CPU inside the container — no external API calls, no
    # rate limits, no AWS quota dependency. Model weights (~80MB) download
    # once from Hugging Face on first run and are cached in the image layer
    # / container filesystem.
    print(f"Loading local embedding model '{EMBED_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Embedding {len(chunks)} chunks locally ...")
    texts = [c["text"] for c in chunks]
    # normalize_embeddings=True -> unit-length vectors, so inner product ==
    # cosine similarity. MiniLM (like most sentence-transformer models) was
    # trained/tuned for cosine similarity, not raw L2 distance — using L2
    # directly tends to produce loosely-related, inconsistent retrieval.
    vectors = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )
    return vectors.astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="URL or local path to the source PDF")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PDF from {args.source} ...")
    units = load_pdf_page_units(args.source)
    n_tables = sum(1 for u in units if u["kind"] == "table")
    print(f"Extracted {len(units)} page units ({n_tables} tables, {len(units) - n_tables} prose blocks)")

    print("Chunking (sentence-aware for prose, tables kept intact) ...")
    chunks = chunk_units(units)
    print(f"Created {len(chunks)} chunks")

    print("Embedding chunks locally ...")
    vectors = embed_chunks(chunks)

    print("Building FAISS index (cosine similarity via normalized inner product) ...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(DATA_DIR / "index.faiss"))
    with open(DATA_DIR / "chunks.json", "w") as f:
        json.dump(chunks, f)

    print(f"Done. Wrote {DATA_DIR / 'index.faiss'} and {DATA_DIR / 'chunks.json'}")


if __name__ == "__main__":
    main()
