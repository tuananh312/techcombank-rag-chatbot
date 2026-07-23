"""
Unit tests for app/rag.py's pure text-processing functions (citation
parsing, number extraction, snippet building). These don't touch the
embedding model, FAISS index, or AWS/Anthropic clients — rag.py lazy-loads
all of those on first use specifically so this module stays safely
importable and testable without credentials, network access, or a
pre-built index (see the module docstring in rag.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from rag import _extract_cited_pages, _make_snippet, _numbers_in  # noqa: E402


class TestExtractCitedPages:
    def test_single_page_citation(self):
        answer = "Net profit rose 15% year-over-year (page 4)."
        assert _extract_cited_pages(answer) == {4}

    def test_multiple_separate_citations(self):
        answer = "Assets grew on page 3. Deposits grew on page 5."
        assert _extract_cited_pages(answer) == {3, 5}

    def test_multi_page_mention_with_and(self):
        answer = "This trend appears across pages 3 and 4 of the report."
        assert _extract_cited_pages(answer) == {3, 4}

    def test_multi_page_mention_with_comma(self):
        answer = "See pages 3, 4, 5 for details."
        assert _extract_cited_pages(answer) == {3, 4, 5}

    def test_no_citation_returns_empty_set(self):
        answer = "I don't have that information in the report I was given."
        assert _extract_cited_pages(answer) == set()

    def test_case_insensitive(self):
        answer = "Stated on Page 7."
        assert _extract_cited_pages(answer) == {7}


class TestNumbersIn:
    def test_extracts_plain_numbers(self):
        assert "1192344" in _numbers_in("Total assets were 1192344 VND bn.")

    def test_normalizes_comma_separators(self):
        # "1,192,344" and "1192344" should be treated as the same figure
        numbers = _numbers_in("Total assets were 1,192,344 VND bn.")
        assert "1192344" in numbers

    def test_extracts_percentages(self):
        numbers = _numbers_in("Net profit rose 15.4%.")
        assert "15.4%" in numbers

    def test_no_numbers_returns_empty_set(self):
        assert _numbers_in("There were no figures mentioned here.") == set()


class TestMakeSnippet:
    def test_prose_snippet_centers_on_cited_number(self):
        text = (
            "This is a long lead-in paragraph that talks about many unrelated "
            "topics before finally getting to the point. Net profit rose 15.4% "
            "year over year according to the latest figures released today."
        )
        snippet = _make_snippet(text, "prose", {"15.4%"})
        assert "15.4%" in snippet
        # should NOT just be the first 220 chars starting from the beginning
        assert not snippet.startswith("This is a long lead-in")

    def test_prose_snippet_falls_back_to_start_when_no_number_matches(self):
        text = "A" * 300
        snippet = _make_snippet(text, "prose", {"999"})
        assert snippet.startswith("AAA")
        assert snippet.endswith("...")

    def test_table_snippet_shows_only_matching_row(self):
        table_text = (
            "| Metric | 4Q24 | 4Q25 |\n"
            "| --- | --- | --- |\n"
            "| Total assets | 978,799 | 1,192,344 |\n"
            "| Deposits | 564,536 | 665,550 |"
        )
        snippet = _make_snippet(table_text, "table", {"1192344"})
        assert "Total assets" in snippet
        assert "Deposits" not in snippet
        # header should always be preserved for context
        assert "Metric" in snippet

    def test_table_snippet_falls_back_when_no_row_matches(self):
        table_text = (
            "| Metric | 4Q24 |\n"
            "| --- | --- |\n"
            "| Total assets | 978,799 |"
        )
        snippet = _make_snippet(table_text, "table", {"999999"})
        assert "Metric" in snippet  # header still shown as fallback
