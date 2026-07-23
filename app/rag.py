"""
Retrieval-augmented generation logic.

Heavy resources (embedding model, FAISS index, AWS/Anthropic clients) are
lazy-loaded on first use rather than at import time — this keeps the module
safely importable for unit tests (no AWS credentials, network access, or
prebuilt index required just to import it), and defers Lambda cold-start
cost to the first real request rather than paying it on every invocation
setup.

Exposes `answer_question()` which:
  1. embeds the user's question
  2. retrieves the top-k most similar chunks from the report
  3. asks the LLM to answer STRICTLY from those chunks
  4. returns "I don't know" if the answer isn't supported by the context

This is what satisfies the "must not hallucinate / must not go outside the
provided source data" acceptance criterion.
"""

import json
import os
import re
from pathlib import Path

import boto3
from botocore.config import Config
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
GEN_MODEL_ID = os.environ.get("GEN_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
TOP_K = int(os.environ.get("TOP_K", "8"))
DATA_DIR = Path(__file__).parent / "data"

# "bedrock" (default) or "anthropic_direct" — lets you bypass Bedrock quota
# approval entirely by calling Anthropic's API directly while waiting for
# AWS quota increases to clear. See README for trade-offs.
GENERATION_PROVIDER = os.environ.get("GENERATION_PROVIDER", "bedrock")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a financial assistant that answers questions ONLY using the
provided context, which comes from Techcombank's fiscal year press release.
Each context passage is labeled with the page number it came from, like
"[Page 3]". Some passages are markdown tables (rows/columns of figures) —
read them carefully by column header, since a number's meaning depends on
which column and row it's in.

Rules:
- Only use facts present in the CONTEXT below. Do not use outside knowledge.
- If the answer is not contained in the CONTEXT, reply exactly:
  "I don't have that information in the report I was given."
- Never guess, extrapolate, or fabricate numbers.
- Write in a natural, conversational tone — not a bulleted data dump.
- Do not use markdown formatting (no **bold**, no bullet points, no
  headers). Write in plain prose, like you're speaking the answer aloud.
- COMPLETENESS MATTERS: include every relevant fact from the CONTEXT that
  answers the question, even ones from different pages or different
  passages. Do not leave out a true, relevant fact just because you're
  unsure exactly which page it came from — citation confidence and fact
  inclusion are separate concerns. Always state the fact; only the page
  citation itself is optional when uncertain.
- When you ARE confident which page a fact came from, cite it naturally
  within the sentence, e.g. "Net profit rose 15% year-over-year (page 4)."
  If you're not confident of the exact page for a particular fact, state
  the fact anyway and simply omit that one citation — don't drop the fact.
  If a single answer draws on multiple pages, cite each relevant page near
  the fact it supports rather than bundling all citations at the end.
- You may use the conversation history to resolve follow-up / pronoun
  references (e.g. "what about the year before that"), but the underlying
  facts must still come only from CONTEXT.
"""

_bedrock_client = None
_embed_model_instance = None
_index_cache = None
_chunks_cache = None
_anthropic_client_instance = None


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            config=Config(region_name=AWS_REGION, retries={"max_attempts": 10, "mode": "adaptive"}),
        )
    return _bedrock_client


def _get_embed_model():
    global _embed_model_instance
    if _embed_model_instance is None:
        _embed_model_instance = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model_instance


def _get_index_and_chunks():
    global _index_cache, _chunks_cache
    if _index_cache is None:
        _index_cache = faiss.read_index(str(DATA_DIR / "index.faiss"))
        with open(DATA_DIR / "chunks.json") as f:
            _chunks_cache = json.load(f)  # each: {"page": int, "text": str, "kind": str}
    return _index_cache, _chunks_cache


def _get_anthropic_client():
    global _anthropic_client_instance
    if _anthropic_client_instance is None:
        import anthropic

        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "GENERATION_PROVIDER=anthropic_direct requires ANTHROPIC_API_KEY to be set"
            )
        _anthropic_client_instance = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client_instance


def _embed(text: str) -> np.ndarray:
    return _get_embed_model().encode(
        [text], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")


def _retrieve(question: str, k: int = TOP_K) -> list[dict]:
    index, chunks = _get_index_and_chunks()
    query_vec = _embed(question)
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0] if i != -1]


def _generate(question: str, context_chunks: list[dict], history: list[dict]) -> str:
    context = "\n\n---\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in context_chunks)
    user_turn = {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}

    if GENERATION_PROVIDER == "anthropic_direct":
        messages = list(history) + [user_turn]
        resp = _get_anthropic_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return resp.content[0].text

    if GEN_MODEL_ID.startswith("ai21."):
        # AI21 Jamba models: system prompt goes in as a normal "system" role
        # message inside the same `messages` array (no separate `system` key,
        # no `anthropic_version`), and the response shape differs too.
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + list(history)
            + [user_turn]
        )
        body = json.dumps({"messages": messages, "max_tokens": 800})
        resp = _get_bedrock().invoke_model(
            modelId=GEN_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        return payload["choices"][0]["message"]["content"]

    # Default: Anthropic Claude message format
    messages = list(history) + [user_turn]
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "system": SYSTEM_PROMPT,
            "max_tokens": 800,
            "messages": messages,
        }
    )
    resp = _get_bedrock().invoke_model(
        modelId=GEN_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload["content"][0]["text"]


_PAGE_MENTION_RE = re.compile(r"pages?\s+((?:\d+\s*(?:,|and)?\s*)+)", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")


def _extract_cited_pages(answer: str) -> set[int]:
    pages = set()
    for match in _PAGE_MENTION_RE.finditer(answer):
        for num in _DIGITS_RE.findall(match.group(1)):
            pages.add(int(num))
    return pages


_NUMBER_RE = re.compile(r"\d[\d,\.]*%?")


def _numbers_in(text: str) -> set[str]:
    # Normalize commas so "1,234" and "1234" are treated as the same figure
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def _make_snippet(text: str, kind: str, answer_numbers: set[str]) -> str:
    """Builds a short, targeted snippet instead of dumping the whole chunk:
    for prose, a window centered on the actual cited number; for tables,
    just the header + the specific row(s) containing that number."""
    if kind == "table":
        lines = text.split("\n")
        if len(lines) <= 2:
            return text
        header_lines = lines[:2]  # markdown header row + separator row
        matched = [
            line for line in lines[2:]
            if any(n.group().replace(",", "") in answer_numbers for n in _NUMBER_RE.finditer(line))
        ]
        if matched:
            return "\n".join(header_lines + matched[:3])
        return "\n".join(lines[:4])  # fallback: header + first couple rows

    # prose — find the first cited number's exact position and window around it
    pos = -1
    for m in _NUMBER_RE.finditer(text):
        if m.group().replace(",", "") in answer_numbers:
            pos = m.start()
            break

    if pos == -1:
        return text if len(text) <= 220 else text[:220] + "..."

    start = max(0, pos - 80)
    end = min(len(text), pos + 140)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    history = history or []
    retrieved = _retrieve(question)
    answer = _generate(question, retrieved, history)
    answer = answer.replace("**", "")  # safety net in case the model still emits markdown bold

    # Only surface sources for pages the model actually cited in its answer
    # text — `retrieved` is the full top-k candidate set considered during
    # generation, but not every candidate ends up being relevant/used, and
    # showing all of them makes the citations panel noisy with irrelevant
    # near-misses.
    cited_pages = _extract_cited_pages(answer)
    answer_numbers = _numbers_in(answer)

    seen_pages = set()
    sources = []
    for c in sorted(retrieved, key=lambda c: c["page"]):
        if cited_pages and c["page"] not in cited_pages:
            continue
        if c["page"] in seen_pages:
            continue

        # Verification: if the answer contains specific numbers, require the
        # cited page's actual chunk text to share at least one of them.
        # Catches the model mislabeling a real fact with the wrong (but
        # still-retrieved) page number, rather than trusting the label alone.
        if answer_numbers:
            chunk_numbers = _numbers_in(c["text"])
            if not (answer_numbers & chunk_numbers):
                continue

        seen_pages.add(c["page"])
        kind = c.get("kind", "prose")
        snippet = _make_snippet(c["text"], kind, answer_numbers)
        sources.append({"page": c["page"], "snippet": snippet, "kind": kind})

    # Fallback: if the model didn't emit any parseable page citation (e.g.
    # it answered "I don't have that information"), fall back to showing
    # nothing rather than the full noisy candidate set.
    if not cited_pages:
        sources = []

    return {"answer": answer, "sources": sources}
