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
"""The ``ID: n`` a knowledge block carries is the citation the answer uses.

``decorate_answer`` resolves ``[ID:n]`` as ``kbinfos["chunks"][n]``, and the
reference list a client receives is in the same order, so the first block must
be ``ID: 0``. A block numbered from 1 makes every citation point to the next
chunk and drops the citation of the last one.
"""

import re

import pytest

from rag.advanced_rag.agentic_rag import _expand_range_citation_markers, _repair_slot_citation_markers
from rag.prompts.generator import kb_prompt

ID_RE = re.compile(r"ID: (\d+)\n")


def _chunk(chunk_id: str) -> dict:
    return {"id": chunk_id, "content_with_weight": f"content of {chunk_id}", "docnm_kwd": f"{chunk_id}.md"}


@pytest.mark.p1
def test_blocks_are_numbered_by_their_index_in_the_chunk_list():
    kbinfos = {"chunks": [_chunk("a"), _chunk("b"), _chunk("c")]}

    rendered = kb_prompt(kbinfos, max_tokens=10000)

    assert [int(ID_RE.search(block).group(1)) for block in rendered] == [0, 1, 2]
    for index, block in enumerate(rendered):
        assert f"content of {kbinfos['chunks'][index]['id']}" in block


@pytest.mark.p1
def test_a_slot_marker_becomes_the_index_of_its_evidence_chunk():
    answer = "Fact [ID:Slot 7]."

    repaired = _repair_slot_citation_markers(answer, {"7": {"evidence_ids": ["b"]}}, ["a", "b", "c"])

    assert repaired == "Fact [ID:1]."


@pytest.mark.p1
def test_a_range_expands_over_valid_indexes_only():
    assert _expand_range_citation_markers("[ID:0-2]", pool_size=3) == "[ID:0][ID:1][ID:2]"
    assert _expand_range_citation_markers("[ID:1-3]", pool_size=3) == ""
