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
"""Whole-dataset metadata reads on an OpenSearch backend.

Two defects are covered here:

* A whole-dataset read paged with offset/limit is capped at
  ``index.max_result_window`` (10,000), because both OpenSearch and Elasticsearch
  implement offset/limit as from + size and reject a deeper window. A crawl dataset
  with more documents than that silently lost the metadata of everything past the
  first 10,000 rows. ``_FakeOpenSearchStore.search`` raises exactly like the engine
  does, so a read that still goes through it fails here instead of truncating.
* Metadata filter push-down resolved its client as ``docStoreConn.es``, which is
  absent on the OpenSearch connector (it exposes ``docStoreConn.os``), so every
  push-down returned ``None`` and fell back to the in-memory path.
"""

from types import SimpleNamespace

import pytest

from api.db.db_models import DB
from api.db.services.doc_metadata_service import (
    METADATA_SCROLL_BATCH_SIZE,
    METADATA_SCROLL_KEEPALIVE,
    DocMetadataService,
)
from common import settings

pytestmark = pytest.mark.p2

# index.max_result_window, the point past which from + size is refused.
MAX_RESULT_WINDOW = 10000

TENANT_ID = "tenant-1"
KB_ID = "kb-1"
INDEX_NAME = f"ragflow_doc_meta_{TENANT_ID}"


def _hits(total: int, extra_key_at: int | None = None) -> list[dict]:
    """Build ``total`` metadata hits, optionally giving one of them an extra key."""
    hits = []
    for i in range(total):
        meta_fields = {"canon": "0" if i < 30 else "1"}
        if extra_key_at is not None and i == extra_key_at:
            meta_fields["late_key"] = "present"
        hits.append({"_id": f"doc-{i}", "_source": {"id": f"doc-{i}", "kb_id": KB_ID, "meta_fields": meta_fields}})
    return hits


class _FakeScrollClient:
    """Stands in for the raw OpenSearch client's scroll API."""

    def __init__(self, hits: list[dict]):
        self._hits = hits
        self._cursor = 0
        self._size = METADATA_SCROLL_BATCH_SIZE
        self.searches: list[dict] = []
        self.scrolls: list[dict] = []
        self.cleared: list[str] = []

    def search(self, index, body, scroll=None):
        self.searches.append({"index": index, "body": body, "scroll": scroll})
        self._cursor = 0
        self._size = body["size"]
        return self._page()

    def scroll(self, scroll_id, scroll=None):
        self.scrolls.append({"scroll_id": scroll_id, "scroll": scroll})
        return self._page()

    def clear_scroll(self, scroll_id):
        self.cleared.append(scroll_id)
        return {"succeeded": True}

    def _page(self) -> dict:
        page = self._hits[self._cursor : self._cursor + self._size]
        self._cursor += len(page)
        return {"_scroll_id": "cursor-1", "hits": {"hits": page, "total": {"value": len(self._hits)}}}


class _FakePushdownClient:
    """Stands in for the raw OpenSearch client's plain search API."""

    def __init__(self, hits: list[dict]):
        self._hits = hits
        self.searches: list[dict] = []

    def search(self, index, body):
        self.searches.append({"index": index, "body": body})
        return {"hits": {"hits": self._hits, "total": {"value": len(self._hits)}}}


class _FakeOpenSearchStore:
    """A ``docStoreConn`` shaped like ``OSConnection``: a raw client on ``os``.

    ``search`` honors offset/limit and refuses a window past
    ``index.max_result_window``, the way OpenSearch itself does, so a caller that
    paginates a whole dataset through it fails loudly rather than truncating.
    """

    def __init__(self, client, hits: list[dict] | None = None, index_exists: bool = True):
        self.os = client
        self._hits = hits or []
        self._index_exists = index_exists
        self.searches: list[dict] = []
        self.deletes: list[tuple] = []
        self.deleted_count = 0
        self.delete_error: Exception | None = None

    def index_exist(self, index_name, kb_id):
        return self._index_exists

    def search(self, select_fields, highlight_fields, condition, match_expressions, order_by, offset, limit, index_names, knowledgebase_ids, agg_fields=None, rank_feature=None):
        self.searches.append({"condition": dict(condition), "offset": offset, "limit": limit, "index_names": index_names})
        if offset + limit > MAX_RESULT_WINDOW:
            raise RuntimeError(f"Result window is too large, from + size must be less than or equal to {MAX_RESULT_WINDOW}")

        hits = self._hits
        if condition.get("id"):
            wanted = set(condition["id"])
            hits = [hit for hit in hits if hit["_id"] in wanted]
        page = [{"_id": hit["_id"], "_source": dict(hit["_source"])} for hit in hits[offset : offset + limit]]
        return {"hits": {"hits": page, "total": {"value": len(hits)}}}

    def delete(self, condition, index_name, kb_id):
        self.deletes.append((dict(condition), index_name, kb_id))
        if self.delete_error is not None:
            raise self.delete_error
        return self.deleted_count


class _FakeHybridSearchClient:
    """Stands in for OceanBase's hybrid search client: it speaks the query DSL, but has no scroll API."""

    def __init__(self):
        self.searches: list[dict] = []

    def search(self, index, body, **kwargs):
        self.searches.append({"index": index, "body": body})
        return {"hits": {"hits": [], "total": {"value": 0}}}


class _FakeClientlessStore(_FakeOpenSearchStore):
    """A ``docStoreConn`` with no raw ES/OpenSearch client, like the Infinity connector."""

    def __init__(self, hits: list[dict] | None = None, index_exists: bool = True):
        super().__init__(client=None, hits=hits, index_exists=index_exists)
        del self.os


class _FakeOceanBaseStore(_FakeClientlessStore):
    """A ``docStoreConn`` whose native client sits on ``es`` and implements no scroll API."""

    def __init__(self, hits: list[dict] | None = None, index_exists: bool = True):
        super().__init__(hits=hits, index_exists=index_exists)
        self.es = _FakeHybridSearchClient()


@pytest.fixture
def doc_store(monkeypatch):
    """Install a fake doc store and stub out the DB access the service does."""
    monkeypatch.setattr(DB, "connect", lambda *args, **kwargs: None)
    monkeypatch.setattr(DB, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(settings, "DOC_ENGINE_INFINITY", False)
    monkeypatch.setattr(settings, "DOC_ENGINE_GAUSSDB", False)
    monkeypatch.setattr(
        "api.db.services.doc_metadata_service.Knowledgebase.get_by_id",
        lambda kb_id: SimpleNamespace(tenant_id=TENANT_ID),
    )

    def install(store):
        monkeypatch.setattr(settings, "docStoreConn", store)
        return store

    return install


class TestWholeDatasetReadsScrollPastTheWindow:
    def test_get_flatted_meta_by_kbs_reads_every_row(self, doc_store):
        total = 25000
        client = _FakeScrollClient(_hits(total))
        store = doc_store(_FakeOpenSearchStore(client))

        meta = DocMetadataService.get_flatted_meta_by_kbs([KB_ID])

        assert len(meta["canon"]["0"]) + len(meta["canon"]["1"]) == total
        assert store.searches == [], "a whole-dataset read must not go through the from + size reader"

    def test_get_flatted_meta_by_kbs_scrolls_the_index(self, doc_store):
        total = 25000
        client = _FakeScrollClient(_hits(total))
        doc_store(_FakeOpenSearchStore(client))

        DocMetadataService.get_flatted_meta_by_kbs([KB_ID])

        assert client.searches == [
            {
                "index": INDEX_NAME,
                "body": {"query": {"terms": {"kb_id": [KB_ID]}}, "size": METADATA_SCROLL_BATCH_SIZE},
                "scroll": METADATA_SCROLL_KEEPALIVE,
            }
        ]
        # Two full batches, one partial, then the empty page that ends the walk.
        assert len(client.scrolls) == total // METADATA_SCROLL_BATCH_SIZE + 1
        assert all(call["scroll_id"] == "cursor-1" for call in client.scrolls)
        assert client.cleared == ["cursor-1"]

    def test_get_flatted_meta_by_kbs_without_a_metadata_index(self, doc_store):
        client = _FakeScrollClient(_hits(10))
        doc_store(_FakeOpenSearchStore(client, index_exists=False))

        assert DocMetadataService.get_flatted_meta_by_kbs([KB_ID]) == {}
        assert client.searches == []

    def test_get_metadata_summary_counts_every_row(self, doc_store):
        total = 12000
        client = _FakeScrollClient(_hits(total))
        store = doc_store(_FakeOpenSearchStore(client))

        summary = DocMetadataService.get_metadata_summary(KB_ID)

        assert sum(count for _value, count in summary["canon"]["values"]) == total
        assert store.searches == []

    def test_get_metadata_for_documents_returns_every_row(self, doc_store):
        total = 25000
        client = _FakeScrollClient(_hits(total))
        store = doc_store(_FakeOpenSearchStore(client))

        mapping = DocMetadataService.get_metadata_for_documents(None, KB_ID)

        assert len(mapping) == total
        assert mapping["doc-24999"] == {"canon": "1"}
        assert store.searches == []

    def test_get_metadata_keys_by_kbs_sees_a_key_past_the_window(self, doc_store):
        total = 25000
        client = _FakeScrollClient(_hits(total, extra_key_at=total - 1))
        store = doc_store(_FakeOpenSearchStore(client))

        keys = DocMetadataService.get_metadata_keys_by_kbs([KB_ID])

        assert keys == ["canon", "late_key"]
        assert store.searches == []

    def test_clientless_backend_keeps_using_the_paged_reader(self, doc_store):
        total = 2500
        store = doc_store(_FakeClientlessStore(_hits(total)))

        rows = list(DocMetadataService._iter_all_metadata([KB_ID]))

        assert [doc_id for doc_id, _source in rows] == [f"doc-{i}" for i in range(total)]
        assert [call["offset"] for call in store.searches] == [0, 1000, 2000]

    def test_backend_without_a_scroll_api_keeps_using_the_paged_reader(self, doc_store):
        total = 2500
        store = doc_store(_FakeOceanBaseStore(_hits(total)))

        rows = list(DocMetadataService._iter_all_metadata([KB_ID]))

        assert [doc_id for doc_id, _source in rows] == [f"doc-{i}" for i in range(total)]
        assert [call["offset"] for call in store.searches] == [0, 1000, 2000]
        assert store.es.searches == []


class TestIdFilteredReadsStayOnTheBoundedReader:
    def test_get_metadata_for_documents_with_doc_ids(self, doc_store):
        client = _FakeScrollClient([])
        store = doc_store(_FakeOpenSearchStore(client, hits=_hits(50)))

        mapping = DocMetadataService.get_metadata_for_documents(["doc-31", "doc-32"], KB_ID)

        assert sorted(mapping) == ["doc-31", "doc-32"]
        assert [call["condition"]["id"] for call in store.searches] == [["doc-31", "doc-32"]]
        assert client.searches == []

    def test_get_metadata_summary_with_doc_ids(self, doc_store):
        client = _FakeScrollClient([])
        store = doc_store(_FakeOpenSearchStore(client, hits=_hits(50)))

        summary = DocMetadataService.get_metadata_summary(KB_ID, ["doc-0", "doc-31"])

        assert sorted(summary["canon"]["values"]) == [("0", 1), ("1", 1)]
        assert [call["condition"]["id"] for call in store.searches] == [["doc-0", "doc-31"]]
        assert client.searches == []


class TestMetadataFilterPushdownOnOpenSearch:
    def test_pushdown_uses_the_opensearch_client(self, doc_store):
        client = _FakePushdownClient([{"_id": "doc-1"}, {"_id": "doc-2"}])
        doc_store(_FakeOpenSearchStore(client))

        doc_ids = DocMetadataService.filter_doc_ids_by_meta_pushdown(
            [KB_ID],
            [{"key": "canon", "op": "=", "value": "1"}],
        )

        assert doc_ids == ["doc-1", "doc-2"]
        assert len(client.searches) == 1
        request = client.searches[0]
        assert request["index"] == INDEX_NAME
        assert request["body"]["query"]["bool"]["filter"][0] == {"terms": {"kb_id": [KB_ID]}}
        assert request["body"]["track_total_hits"] is True

    def test_pushdown_falls_back_without_a_raw_client(self, doc_store):
        doc_store(_FakeClientlessStore(_hits(10)))

        doc_ids = DocMetadataService.filter_doc_ids_by_meta_pushdown(
            [KB_ID],
            [{"key": "canon", "op": "=", "value": "1"}],
        )

        assert doc_ids is None


class TestDeleteKbMetadata:
    def test_deletes_every_row_of_the_dataset(self, doc_store):
        store = doc_store(_FakeOpenSearchStore(_FakeScrollClient([])))
        store.deleted_count = 7

        assert DocMetadataService.delete_kb_metadata(KB_ID, TENANT_ID) == 7
        assert store.deletes == [({"kb_id": KB_ID}, INDEX_NAME, KB_ID)]

    def test_without_a_metadata_index_deletes_nothing(self, doc_store):
        store = doc_store(_FakeOpenSearchStore(_FakeScrollClient([]), index_exists=False))

        assert DocMetadataService.delete_kb_metadata(KB_ID, TENANT_ID) == 0
        assert store.deletes == []

    def test_reports_zero_when_the_delete_fails(self, doc_store):
        store = doc_store(_FakeOpenSearchStore(_FakeScrollClient([])))
        store.deleted_count = 7
        store.delete_error = RuntimeError("delete_by_query failed")

        assert DocMetadataService.delete_kb_metadata(KB_ID, TENANT_ID) == 0
