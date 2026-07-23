"""
Unit tests for app/ingest.py's pure functions — table parsing, chunking,
sentence splitting. These don't touch the network, AWS, or the embedding
model, so they run fast and need no credentials or fixtures beyond plain
Python data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from ingest import (  # noqa: E402
    _is_header_like_row,
    _is_valid_table,
    _split_compound_table,
    _split_sentences,
    _table_to_markdown,
    chunk_units,
)


class TestTableToMarkdown:
    def test_renders_header_and_rows(self):
        table = [["Metric", "4Q24", "4Q25"], ["Total assets", "978,799", "1,192,344"]]
        md = _table_to_markdown(table)
        assert "| Metric | 4Q24 | 4Q25 |" in md
        assert "| --- | --- | --- |" in md
        assert "| Total assets | 978,799 | 1,192,344 |" in md

    def test_handles_none_cells(self):
        table = [["A", None, "C"], ["1", None, "3"]]
        md = _table_to_markdown(table)
        # None cells should render as empty, not the string "None"
        assert "None" not in md

    def test_empty_table_returns_empty_string(self):
        assert _table_to_markdown([]) == ""


class TestHeaderLikeRow:
    def test_detects_quarter_labels(self):
        row = ["Balance Sheet", "Unit:", "4Q24", "1Q25", "2Q25", "QoQ", "YoY"]
        assert _is_header_like_row(row) is True

    def test_rejects_normal_data_row(self):
        row = ["Total assets", "VND bn", "978,799", "989,216", "1,037,645"]
        assert _is_header_like_row(row) is False

    def test_rejects_row_with_single_header_cell(self):
        # only one cell matches (QoQ) — below the 2-cell threshold
        row = ["Something", "QoQ", "45%"]
        assert _is_header_like_row(row) is False


class TestSplitCompoundTable:
    def test_splits_at_repeated_header_rows(self):
        table = [
            ["Balance Sheet", "Unit:", "4Q24", "4Q25", "QoQ", "YoY"],
            ["Total assets", "VND bn", "978,799", "1,192,344", "5.6%", "21.8%"],
            ["Capital & Liquidity", "Unit:", "4Q24", "4Q25", "QoQ", "YoY"],
            ["Basel II CAR", "%", "15.4%", "14.6%", "-120 bps", "-78 bps"],
        ]
        subtables = _split_compound_table(table)
        assert len(subtables) == 2
        assert subtables[0][0][0] == "Balance Sheet"
        assert subtables[1][0][0] == "Capital & Liquidity"
        assert subtables[0][-1][0] == "Total assets"
        assert subtables[1][-1][0] == "Basel II CAR"

    def test_no_split_when_no_repeated_header(self):
        table = [
            ["Metric", "4Q24", "4Q25"],
            ["Total assets", "978,799", "1,192,344"],
            ["Deposits", "564,536", "665,550"],
        ]
        subtables = _split_compound_table(table)
        assert len(subtables) == 1
        assert len(subtables[0]) == 3

    def test_empty_table_returns_empty_list(self):
        assert _split_compound_table([]) == []


class TestIsValidTable:
    def test_accepts_well_formed_table(self):
        table = [["A", "B"], ["1", "2"], ["3", "4"]]
        assert _is_valid_table(table) is True

    def test_rejects_single_row(self):
        assert _is_valid_table([["A", "B"]]) is False

    def test_rejects_single_column(self):
        assert _is_valid_table([["A"], ["1"], ["2"]]) is False

    def test_rejects_mostly_empty_table(self):
        # Common false positive from text-based table detection on prose:
        # a sparse grid of mostly-empty cells.
        table = [["A", None, None, None], [None, None, "x", None], [None] * 4]
        assert _is_valid_table(table) is False

    def test_rejects_empty_input(self):
        assert _is_valid_table([]) is False


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self):
        text = "Net profit rose 15%. Total assets grew too. NPL improved."
        sentences = _split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Net profit rose 15%."

    def test_does_not_split_on_decimal_numbers(self):
        # A naive split-on-period would incorrectly break "15.4%" in two
        text = "Basel II CAR was 15.4% in the quarter."
        sentences = _split_sentences(text)
        assert len(sentences) == 1
        assert "15.4%" in sentences[0]

    def test_empty_text_returns_empty_list(self):
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []


class TestChunkUnits:
    def test_prose_is_split_into_sentence_bounded_chunks(self):
        long_prose = "Sentence one is here. " * 40  # well over default chunk size
        units = [{"page": 1, "text": long_prose, "kind": "prose"}]
        chunks = chunk_units(units, size=100, overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert c["page"] == 1
            assert c["kind"] == "prose"
            # never cut mid-sentence: every chunk should end with sentence punctuation
            assert c["text"].rstrip().endswith(".")

    def test_table_units_are_never_split(self):
        table_text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        units = [{"page": 3, "text": table_text, "kind": "table"}]
        chunks = chunk_units(units, size=5, overlap=1)  # tiny size — would force-split prose
        assert len(chunks) == 1
        assert chunks[0]["text"] == table_text
        assert chunks[0]["kind"] == "table"

    def test_page_number_preserved_across_chunks(self):
        units = [
            {"page": 1, "text": "First page sentence.", "kind": "prose"},
            {"page": 2, "text": "Second page sentence.", "kind": "prose"},
        ]
        chunks = chunk_units(units)
        pages = {c["page"] for c in chunks}
        assert pages == {1, 2}
