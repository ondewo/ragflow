#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Markdown chunking of tables, and the trailing newline of the last section.

``max_inline_tokens`` decides per table whether it stays in the prose around it
or is pulled out as a standalone chunk, which keeps an oversized table from
taking a whole chunk to itself. The parser input carries a trailing newline for
the table patterns to anchor on, which must not reach the sections.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[4]))

from markdown import markdown  # noqa: E402

from common.token_utils import num_tokens_from_string  # noqa: E402

SMALL_TABLE = "| X | Y |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
BIG_TABLE = "| Spalte A | Spalte B | Spalte C |\n|---|---|---|\n" + "".join(f"| Zeile {i} | Wert {i} | Eine etwas laengere Beschreibung {i} |\n" for i in range(40))


def _restore_real_packages(top_level: tuple[str, ...] = ("deepdoc", "rag")) -> None:
    """Drop lightweight module stubs so the real packages can load.

    Sibling suites register stand-ins for parts of ``deepdoc`` and ``rag`` to import without
    their dependency chains -- ``test/unit_test/rag/conftest.py`` stubs
    ``deepdoc.parser.pdf_parser``, for instance. ``rag.app.naive`` imports both packages whole,
    which no stub satisfies, so restore the real modules when those suites share a session with
    this one. A stub is a bare ``types.ModuleType`` with no ``__file__``.
    """
    for name in top_level:
        keys = [key for key in sys.modules if key == name or key.startswith(f"{name}.")]
        if not any(getattr(sys.modules[key], "__file__", None) is None for key in keys):
            continue
        for key in keys:
            del sys.modules[key]
    importlib.invalidate_caches()


@pytest.fixture
def parser():
    """The markdown chunker, imported lazily so a hostile sys.modules cannot abort collection."""
    _restore_real_packages()
    from rag.app.naive import Markdown

    return Markdown(128)


def parse(parser, text, **kwargs):
    sections, tables, _ = parser("doc.md", binary=text.encode("utf-8"), return_section_images=True, **kwargs)
    return [content for content, _ in sections], tables


@pytest.mark.p2
def test_no_section_ends_with_the_newline_added_to_the_parser_input(parser):
    text = "# Titel\n\nEin Satz hier.\n\n## Zweiter Abschnitt\n\nNoch ein Satz."

    sections, _ = parse(parser, text)

    assert sections
    assert sections[-1] == "Noch ein Satz."
    assert not any(content.endswith("\n") for content in sections)


@pytest.mark.p2
@pytest.mark.parametrize("ending", ["", "\n"])
def test_the_newline_added_for_the_table_patterns_never_reaches_a_section(parser, ending):
    """The last section ends exactly where the document ends, with nothing the parser added."""
    last_paragraph = "Noch ein Satz."
    text = "# Titel\n\nEin Satz hier.\n\n## Zweiter Abschnitt\n\n" + last_paragraph + ending

    sections, _ = parse(parser, text)

    assert sections[-1] == last_paragraph + ending


@pytest.mark.p2
def test_table_within_the_inline_budget_stays_in_the_prose(parser):
    text = f"Prosa davor.\n\n{SMALL_TABLE}\nProsa danach."

    sections, tables = parse(parser, text, max_inline_tokens=512)

    assert tables == []
    assert "<table>" in "\n".join(sections)


@pytest.mark.p2
def test_table_over_the_inline_budget_becomes_a_standalone_chunk(parser):
    text = f"Prosa davor.\n\n{BIG_TABLE}\nProsa danach."

    sections, tables = parse(parser, text, max_inline_tokens=128)

    assert len(tables) == 1
    assert "<table>" not in "\n".join(sections)
    assert "Prosa davor." in "\n".join(sections)
    assert "Prosa danach." in "\n".join(sections)


@pytest.mark.p2
def test_inline_budget_decides_per_table(parser):
    text = f"Prosa davor.\n\n{SMALL_TABLE}\nProsa dazwischen.\n\n{BIG_TABLE}\nProsa danach."

    sections, tables = parse(parser, text, max_inline_tokens=128)

    assert len(tables) == 1
    assert num_tokens_from_string(tables[0][0][1]) > 128
    assert "<table>" in "\n".join(sections)
    assert "Zeile 39" not in "\n".join(sections)


@pytest.mark.p2
def test_standalone_table_chunk_is_rendered_html_without_an_image(parser):
    text = f"Prosa davor.\n\n{BIG_TABLE}"

    _, tables = parse(parser, text, max_inline_tokens=128)

    (image, rendered), positions = tables[0]
    assert image is None
    assert positions == ""
    assert rendered.startswith("<table>")
    assert rendered.rstrip().endswith("</table>")


@pytest.mark.p2
@pytest.mark.parametrize("table", [SMALL_TABLE, BIG_TABLE])
def test_default_budget_renders_tables_inline_whatever_their_size(parser, table):
    """Without a budget the table size must not matter: that is the upstream contract."""
    text = f"Prosa davor.\n\n{table}\nProsa danach.\n"

    remainder, tables = parser.extract_tables_and_remainder(text, separate_tables=False)

    assert "<table>" in remainder
    assert tables


@pytest.mark.p2
@pytest.mark.parametrize("table", [SMALL_TABLE, BIG_TABLE])
def test_default_budget_separates_tables_whatever_their_size(parser, table):
    text = f"Prosa davor.\n\n{table}\nProsa danach.\n"

    remainder, tables = parser.extract_tables_and_remainder(text, separate_tables=True)

    assert "<table>" not in remainder
    assert tables


@pytest.mark.p2
def test_raw_html_table_is_weighed_by_the_same_budget(parser):
    """The raw-html pass also sees what the markdown pass rendered inline, so it shares the budget."""
    small = "<table><tbody><tr><td>eins</td><td>zwei</td></tr><tr><td>drei</td><td>vier</td></tr></tbody></table>"
    big = "<table><tbody>" + "".join(f"<tr><td>Zeile {i}</td><td>Eine etwas laengere Beschreibung {i}</td></tr>" for i in range(60)) + "</tbody></table>"

    remainder, tables = parser.extract_tables_and_remainder(f"Prosa.\n\n{small}\n\nMitte.\n\n{big}\n\nEnde.\n", max_inline_tokens=128)

    assert len(tables) == 1
    assert num_tokens_from_string(tables[0]) > 128
    assert "eins" in remainder
    assert "Zeile 59" not in remainder


@pytest.mark.p2
def test_a_table_exactly_at_the_budget_is_weighed_the_same_by_both_passes(parser):
    """The raw-html pass must not re-extract what the markdown pass just kept inline.

    Its match swallows the blank lines around the table, so weighing the match verbatim
    made the budget one token tighter in the second pass than in the first.
    """
    table = "| A | B |\n|---|---|\n| Zeile 0 | Wert 0 |\n"
    rendered = markdown(table, extensions=["markdown.extensions.tables"])
    budget = num_tokens_from_string(rendered)

    remainder, tables = parser.extract_tables_and_remainder(f"Prosa davor.\n\n{table}\nProsa danach.\n", max_inline_tokens=budget)

    assert tables == []
    assert "<table>" in remainder
