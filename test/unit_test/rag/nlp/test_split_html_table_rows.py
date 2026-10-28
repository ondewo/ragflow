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

"""Row-group splitting of oversized rendered tables.

A single large table otherwise becomes one chunk that overruns the embedding and
reranker input limits, so ``split_html_table_rows`` cuts it into row groups that
each repeat the head and tail and therefore stay valid tables on their own.
"""

import re
import xml.etree.ElementTree as ElementTree

import pytest
from markdown import markdown

from common.token_utils import num_tokens_from_string
from rag import nlp
from rag.nlp import split_html_table_rows

ROW_RE = re.compile(r"<tr>.*?</tr>", re.DOTALL)
BODY_RE = re.compile(r"<tbody>(.*)</tbody>", re.DOTALL)


def render_table(row_count: int, columns: tuple[str, ...] = ("Spalte A", "Spalte B", "Spalte C")) -> str:
    """Render a markdown table the way the markdown chunker feeds it to tokenize_table."""
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = ["| " + " | ".join(f"Zeile {i} Spalte {c}" for c in range(len(columns))) + " |" for i in range(row_count)]
    source = "\n".join([header, separator, *rows]) + "\n"
    return markdown(source, extensions=["markdown.extensions.tables"])


def body_rows(html: str) -> list[str]:
    """The ``<tr>`` elements inside ``<tbody>``, i.e. the rows the split must conserve."""
    body = BODY_RE.search(html)
    assert body is not None
    return ROW_RE.findall(body.group(1))


@pytest.fixture
def big_table() -> str:
    return render_table(30)


@pytest.mark.p2
def test_table_within_budget_is_returned_unchanged(big_table):
    budget = num_tokens_from_string(big_table)

    assert split_html_table_rows(big_table, budget) == [big_table]


@pytest.mark.p2
def test_oversized_table_splits_into_several_groups(big_table):
    budget = num_tokens_from_string(big_table) // 4

    groups = split_html_table_rows(big_table, budget)

    assert len(groups) > 1


@pytest.mark.p2
def test_every_group_is_a_well_formed_table_with_the_original_head_and_tail(big_table):
    rendered_rows = body_rows(big_table)
    head = big_table[: big_table.index(rendered_rows[0])]
    tail = big_table[big_table.rindex(rendered_rows[-1]) + len(rendered_rows[-1]) :]
    header_cells = ROW_RE.search(big_table).group()

    groups = split_html_table_rows(big_table, num_tokens_from_string(big_table) // 4)

    for group in groups:
        # parses as XML => every tag opened in the group is also closed in it
        ElementTree.fromstring(group)
        assert group.startswith(head.rstrip("\n"))
        assert group.endswith(tail.lstrip("\n"))
        assert header_cells in group


@pytest.mark.p2
@pytest.mark.parametrize("divisor", [2, 3, 4, 7, 11])
def test_groups_concatenate_back_to_exactly_the_input_rows(big_table, divisor):
    expected = body_rows(big_table)

    groups = split_html_table_rows(big_table, num_tokens_from_string(big_table) // divisor)

    recovered = [row for group in groups for row in body_rows(group)]
    assert recovered == expected


@pytest.mark.p2
@pytest.mark.parametrize("oversized_index", [0, 2, 4])
def test_row_larger_than_the_budget_is_emitted_on_its_own(oversized_index):
    table = render_table(5)
    rows = body_rows(table)
    oversized_row = rows[oversized_index].replace("Zeile", "Zeile " + " sehr lange Zelle" * 60)
    table = table.replace(rows[oversized_index], oversized_row)
    # the budget fits a normal row but not the inflated one
    budget = num_tokens_from_string(rows[0]) * 2

    groups = split_html_table_rows(table, budget)

    assert [row for group in groups for row in body_rows(group)] == body_rows(table)
    assert [body_rows(group) for group in groups].count([oversized_row]) == 1


@pytest.mark.p2
def test_table_without_tbody_is_returned_unchanged():
    html = "<table>\n<tr>\n<td>" + "Wort " * 400 + "</td>\n</tr>\n</table>"

    assert split_html_table_rows(html, 10) == [html]


@pytest.mark.p2
@pytest.mark.parametrize("row_count", [0, 1])
def test_table_with_fewer_than_two_body_rows_is_returned_unchanged(row_count):
    # the markdown renderer always emits a body row, so build these two shapes directly
    long_cell = "Wort " * 400
    rows = "".join(f"<tr>\n<td>{long_cell}</td>\n</tr>\n" for _ in range(row_count))
    table = f"<table>\n<thead>\n<tr>\n<th>{long_cell}</th>\n</tr>\n</thead>\n<tbody>\n{rows}</tbody>\n</table>"
    assert num_tokens_from_string(table) > 10
    assert len(body_rows(table)) == row_count

    assert split_html_table_rows(table, 10) == [table]


@pytest.mark.p2
@pytest.mark.parametrize("max_tokens", [0, -1, -512])
def test_non_positive_budget_disables_splitting(big_table, max_tokens):
    assert split_html_table_rows(big_table, max_tokens) == [big_table]


@pytest.mark.p2
def test_tokenize_table_splits_a_string_table_into_one_chunk_per_group(monkeypatch, big_table):
    monkeypatch.setattr(nlp, "tokenize", lambda doc, text, eng, language="English": None)
    budget = num_tokens_from_string(big_table) // 4

    chunks = nlp.tokenize_table([((None, big_table), "")], {}, True, max_table_tokens=budget)

    expected = split_html_table_rows(big_table, budget)
    assert len(expected) > 1
    assert [chunk["content_with_weight"] for chunk in chunks] == expected
    assert {chunk["doc_type_kwd"] for chunk in chunks} == {"table"}


@pytest.mark.p2
def test_tokenize_table_default_budget_keeps_one_chunk_per_table(monkeypatch, big_table):
    monkeypatch.setattr(nlp, "tokenize", lambda doc, text, eng, language="English": None)

    chunks = nlp.tokenize_table([((None, big_table), "")], {}, True)

    assert [chunk["content_with_weight"] for chunk in chunks] == [big_table]
