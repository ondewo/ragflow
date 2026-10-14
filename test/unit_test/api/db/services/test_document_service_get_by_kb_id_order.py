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
"""Tests for the ordering DocumentService.get_by_kb_id() pages by.

A document list is paginated with OFFSET/LIMIT. Ordered by a column with
repeated values (status, run, type, size …) alone, the database may return the
tied rows in a different order for every page, so consecutive pages overlap and
skip documents. The id breaks those ties in the requested direction.
"""

import warnings

import pytest

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from api.db.db_models import Document  # noqa: E402
from api.db.services import document_service  # noqa: E402
from api.db.services.document_service import DocumentService  # noqa: E402

pytestmark = pytest.mark.p2


class _FakeQuery:
    """Peewee-like query that records the ordering instead of hitting a database."""

    def __init__(self, seen):
        self._seen = seen

    def join(self, *_args, **_kwargs):
        return self

    def where(self, *_args, **_kwargs):
        return self

    def count(self):
        return 0

    def order_by(self, *orderings):
        self._seen["order_by"] = [(o.node.name, o.direction) for o in orderings]
        return self

    def paginate(self, *_args):
        return self

    def dicts(self):
        return []


@pytest.mark.parametrize(("desc", "direction"), [(True, "DESC"), (False, "ASC")])
def test_ties_are_broken_by_id_in_the_requested_direction(monkeypatch, desc, direction):
    seen = {}
    monkeypatch.setattr(document_service.DB, "connect", lambda *_a, **_k: None)
    monkeypatch.setattr(document_service.DB, "close", lambda *_a, **_k: None)
    monkeypatch.setattr(document_service.DocMetadataService, "get_metadata_for_documents", classmethod(lambda _cls, *_a: {}))
    monkeypatch.setattr(Document, "select", classmethod(lambda _cls, *_a: _FakeQuery(seen)))

    DocumentService.get_by_kb_id("kb-1", 2, 30, "status", desc, "", [], [], [])

    assert seen["order_by"] == [("status", direction), ("id", direction)]
