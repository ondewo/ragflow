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
"""Tests for the `orderby` parameter of `GET /datasets/<dataset_id>/documents`.

A document list can be ordered by every field the response carries that has a
sortable column, named as the response names it. The value reaches
`Model.getter_by`, which resolves any attribute path, so anything outside the
list is refused before the query is built.
"""

import pytest
from test_list_docs_id_ids_conflict import DATASET_ID, _load_document_api

from common.constants import RetCode


def _orderby_column(monkeypatch: pytest.MonkeyPatch, orderby: str) -> tuple[tuple, str | None]:
    """Run the list helper with `orderby` and return its result and the column it ordered by."""
    module = _load_document_api(monkeypatch, args=[("orderby", orderby)], owned_doc_ids=set())
    seen: dict = {}

    def _get_by_kb_id(_kb_id, _page, _page_size, column, *_args, **_kwargs):
        seen["column"] = column
        return [], 0

    module.DocumentService.get_by_kb_id = _get_by_kb_id
    result = module._get_docs_with_request(module.request, DATASET_ID)
    return result, seen.get("column")


@pytest.mark.p2
class TestListDocsOrderby:
    @pytest.mark.parametrize(
        ("orderby", "column"),
        [
            ("create_time", "create_time"),
            ("update_time", "update_time"),
            ("name", "name"),
            ("size", "size"),
            ("status", "status"),
            ("run", "run"),
            ("progress", "progress"),
            ("type", "type"),
            ("process_begin_at", "process_begin_at"),
            ("process_duration", "process_duration"),
            ("chunk_count", "chunk_num"),
            ("token_count", "token_num"),
            ("dataset_id", "kb_id"),
            ("chunk_method", "parser_id"),
        ],
    )
    def test_response_field_orders_by_its_column(self, monkeypatch: pytest.MonkeyPatch, orderby: str, column: str) -> None:
        result, ordered_by = _orderby_column(monkeypatch, orderby)
        assert result[0] == RetCode.SUCCESS, result
        assert ordered_by == column

    @pytest.mark.parametrize("orderby", ["parser_config", "progress_msg", "thumbnail", "chunk_num", "kb_id.name", "__class__"])
    def test_field_without_a_sortable_column_is_refused(self, monkeypatch: pytest.MonkeyPatch, orderby: str) -> None:
        result, ordered_by = _orderby_column(monkeypatch, orderby)
        assert result == (RetCode.ARGUMENT_ERROR, f"invalid orderby field: {orderby}", [], 0)
        assert ordered_by is None
