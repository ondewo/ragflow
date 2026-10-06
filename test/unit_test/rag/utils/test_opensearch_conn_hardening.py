#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
"""
Tests for the OSConnection hardening a concurrent OpenSearch deployment needs.

Four behaviours, each with its own class below:

* the urllib3 connection pool is sized from ``os.pool_size`` instead of leaving
  urllib3's default of 10, which serialises the task-executor threads and the
  API workers against each other;
* ``insert()`` reports the per-item errors of a rejected bulk batch instead of
  returning an empty list, which a caller reads as "everything landed";
* a search against an index that is not there is logged at DEBUG, because a
  per-tenant metadata index is created on first write and dropped once empty;
* a result window reaching past ``index.max_result_window`` is walked with
  ``search_after``, which OpenSearch accepts, rather than ``from`` + ``size``,
  which it rejects.

The OpenSearch client is mocked throughout, so no cluster is needed.
"""

from __future__ import annotations

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# Importing OSConnection touches opensearchpy at module load, so guard for
# environments where the package isn't installed.
opensearchpy = pytest.importorskip("opensearchpy")


def _install_module(name: str, **attrs) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for key, value in attrs.items():
        if not hasattr(mod, key):
            setattr(mod, key, value)
    return mod


def _install_module_stubs() -> None:
    """Replace the heavy modules opensearch_conn imports at load time.

    ``rag.utils.opensearch_conn`` imports ``common.settings`` (which pulls every
    storage backend) and ``rag.nlp`` at module load. We stub just those so the
    real ``OSConnection`` class can be imported without a live environment.
    """
    _install_module(
        "common.settings",
        OS={"hosts": "stub", "username": "u", "password": "p"},
        ES={},
        DOC_ENGINE_INFINITY=False,
        DOC_ENGINE_OCEANBASE=False,
        DOC_ENGINE="opensearch",
        docStoreConn=None,
    )
    _install_module(
        "rag.nlp",
        is_english=lambda *_args, **_kwargs: False,
        rag_tokenizer=MagicMock(),
    )


_install_module_stubs()

from common import settings  # noqa: E402
from common.doc_store.doc_store_base import MatchDenseExpr, OrderByExpr  # noqa: E402
from rag.utils import opensearch_conn  # noqa: E402

LOGGER_NAME = "ragflow.opensearch_conn"


def _resolve_os_connection_class():
    """Return the real OSConnection class.

    ``@singleton`` wraps the class in a closure that returns a cached instance
    on call, so ``opensearch_conn.OSConnection`` at module scope is a function,
    not a type. Unwrap it so we can ``__new__`` an instance directly and bypass
    the network-dependent ``__init__``.
    """
    candidate = opensearch_conn.OSConnection
    if isinstance(candidate, type):
        return candidate
    closure = getattr(candidate, "__closure__", None) or ()
    for cell in closure:
        contents = cell.cell_contents
        if isinstance(contents, type):
            return contents
    raise RuntimeError("Could not locate the OSConnection class in module scope")


def _make_os_connection():
    """Build an OSConnection without invoking its real ``__init__``."""
    cls = _resolve_os_connection_class()
    conn = cls.__new__(cls)
    conn.os = MagicMock()
    conn.os.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}, "timed_out": False}
    conn.info = {"version": {"number": "2.18.0"}}
    conn.hybrid_search_enabled = False
    conn._hybrid_pipeline = "ragflow_hybrid_pipeline"
    return conn


class TestConnectionPoolSize:
    """The client must be built with a pool big enough for concurrent callers."""

    @staticmethod
    def _client_kwargs(os_config: dict) -> dict:
        """Run the real ``__init__`` against a mocked client and return its kwargs."""
        cls = _resolve_os_connection_class()
        conn = cls.__new__(cls)
        client = MagicMock()
        client.info.return_value = {"version": {"number": "2.18.0"}}
        client.ping.return_value = True

        with (
            patch.dict(settings.OS, os_config, clear=True),
            patch.object(opensearch_conn, "OpenSearch", return_value=client) as ctor,
        ):
            conn.__init__()

        return ctor.call_args.kwargs

    def test_defaults_to_the_module_default(self):
        kwargs = self._client_kwargs({"hosts": "http://localhost:1201"})
        assert kwargs["pool_maxsize"] == opensearch_conn.DEFAULT_POOL_SIZE

    def test_default_is_larger_than_the_urllib3_default_of_ten(self):
        assert opensearch_conn.DEFAULT_POOL_SIZE > 10

    def test_configured_pool_size_wins(self):
        configured = opensearch_conn.DEFAULT_POOL_SIZE * 2
        kwargs = self._client_kwargs({"hosts": "http://localhost:1201", "pool_size": configured})
        assert kwargs["pool_maxsize"] == configured

    def test_quoted_yaml_value_is_coerced_to_int(self):
        """A YAML scalar may arrive as a string; urllib3 needs an int."""
        kwargs = self._client_kwargs({"hosts": "http://localhost:1201", "pool_size": "48"})
        assert kwargs["pool_maxsize"] == 48


class TestInsertErrorReporting:
    """A batch that did not land must never read as an empty error list."""

    @staticmethod
    def _documents(count: int = 2) -> list[dict]:
        return [{"id": f"doc-{i}", "kb_id": "kb1"} for i in range(count)]

    def test_accepted_batch_reports_no_errors(self):
        conn = _make_os_connection()
        conn.os.bulk.return_value = {
            "errors": False,
            "items": [{"index": {"_id": "doc-0", "result": "created"}}, {"index": {"_id": "doc-1", "result": "created"}}],
        }

        assert conn.insert(self._documents(), "ragflow_doc_meta_t1") == []

    def test_rejected_items_are_returned(self):
        conn = _make_os_connection()
        conn.os.bulk.return_value = {
            "errors": True,
            "items": [
                {"index": {"_id": "doc-0", "error": {"type": "mapper_parsing_exception"}}},
                {"index": {"_id": "doc-1", "result": "created"}},
            ],
        }

        errors = conn.insert(self._documents(), "ragflow_doc_meta_t1")

        assert len(errors) == 1
        assert errors[0].startswith("doc-0:")
        assert "mapper_parsing_exception" in errors[0]

    def test_item_errors_are_reported_without_a_usable_errors_flag(self):
        """The per-item records decide, not the batch-level flag.

        The flag is only a hint to look; a response that omits it, or carries
        something other than a bool, must not turn a rejected batch into a
        success.
        """
        conn = _make_os_connection()
        conn.os.bulk.return_value = {"items": [{"index": {"_id": "doc-0", "error": {"type": "cluster_block_exception"}}}]}

        errors = conn.insert(self._documents(1), "ragflow_doc_meta_t1")

        assert len(errors) == 1
        assert "cluster_block_exception" in errors[0]

    def test_non_timeout_failure_is_returned_and_not_retried(self):
        conn = _make_os_connection()
        conn.os.bulk.side_effect = RuntimeError("circuit_breaking_exception: parent breaker tripped")

        errors = conn.insert(self._documents(), "ragflow_doc_meta_t1")

        assert len(errors) == 1
        assert "circuit_breaking_exception" in errors[0]
        assert conn.os.bulk.call_count == 1

    def test_repeated_timeout_is_reported_after_every_attempt(self):
        conn = _make_os_connection()
        conn.os.bulk.side_effect = ConnectionError("Connection timeout caused by: ReadTimeoutError")

        with patch.object(opensearch_conn.time, "sleep"):
            errors = conn.insert(self._documents(), "ragflow_doc_meta_t1")

        assert len(errors) == 1
        assert "timeout" in errors[0].lower()
        assert conn.os.bulk.call_count == opensearch_conn.ATTEMPT_TIME


class TestMissingIndexLogging:
    """A search against an index that is not there is routine, not an incident."""

    def test_missing_index_is_logged_at_debug_and_reraised(self, caplog):
        conn = _make_os_connection()
        conn.os.search.side_effect = opensearchpy.NotFoundError(404, "index_not_found_exception", {})

        with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME), pytest.raises(opensearchpy.NotFoundError):
            conn.search(
                select_fields=["*"],
                highlight_fields=[],
                condition={"kb_id": "kb1"},
                match_expressions=[],
                order_by=None,
                offset=0,
                limit=10,
                index_names="ragflow_doc_meta_t1",
                knowledgebase_ids=["kb1"],
            )

        records = [record for record in caplog.records if record.name == LOGGER_NAME]
        assert [record for record in records if record.levelno >= logging.WARNING] == []
        assert any("missing index" in record.getMessage() for record in records if record.levelno == logging.DEBUG)

    def test_the_search_is_not_retried_on_a_missing_index(self):
        conn = _make_os_connection()
        conn.os.search.side_effect = opensearchpy.NotFoundError(404, "index_not_found_exception", {})

        with pytest.raises(opensearchpy.NotFoundError):
            conn.search(
                select_fields=["*"],
                highlight_fields=[],
                condition={"kb_id": "kb1"},
                match_expressions=[],
                order_by=None,
                offset=0,
                limit=10,
                index_names="ragflow_doc_meta_t1",
                knowledgebase_ids=["kb1"],
            )

        assert conn.os.search.call_count == 1


class _FakeIndex:
    """A sorted index that answers ``from``/``size`` and ``search_after`` queries.

    ``from`` + ``size`` is rejected past ``index.max_result_window`` exactly as
    OpenSearch rejects it, so a test that loses the search_after path fails
    instead of quietly reading the wrong rows.
    """

    def __init__(self, size: int, with_sort: bool = True, max_result_window: int = opensearch_conn.MAX_RESULT_WINDOW) -> None:
        self.ids: list[str] = [f"doc-{i:06d}" for i in range(size)]
        self.with_sort = with_sort
        self.max_result_window = max_result_window
        self.bodies: list[dict] = []

    def search(self, index=None, body=None, **_kwargs) -> dict:
        self.bodies.append(body)
        size = body.get("size", 10)
        search_after = body.get("search_after")
        with_source = _kwargs.get("_source", True)

        if search_after is None:
            start = body.get("from", 0)
            if start + size > self.max_result_window:
                raise opensearchpy.RequestError(
                    400,
                    "search_phase_execution_exception",
                    {"error": f"Result window is too large, from + size must be less than or equal to [{self.max_result_window}]"},
                )
        else:
            start = self.ids.index(search_after[0]) + 1

        window = self.ids[start : start + size]
        hits: list[dict] = []
        for doc_id in window:
            hit = {"_id": doc_id}
            if with_source:
                hit["_source"] = {"id": doc_id, "kb_id": "kb1"}
            if self.with_sort:
                hit["sort"] = [doc_id]
            hits.append(hit)

        res: dict = {"timed_out": False, "hits": {"total": {"value": len(self.ids), "relation": "eq"}, "hits": hits}}
        if "aggs" in body:
            res["aggregations"] = {"aggs_kb_id": {"buckets": [{"key": "kb1", "doc_count": len(self.ids)}]}}
        return res


def _search_page(conn, offset: int, limit: int, order_by=None, **kwargs):
    """Read one page of the doc-meta index the way DocMetadataService does."""
    return conn.search(
        select_fields=["*"],
        highlight_fields=[],
        condition={"kb_id": "kb1"},
        match_expressions=[],
        order_by=order_by if order_by is not None else OrderByExpr().asc("id"),
        offset=offset,
        limit=limit,
        index_names="ragflow_doc_meta_t1",
        knowledgebase_ids=["kb1"],
        **kwargs,
    )


class TestDeepPagination:
    """Reading a whole per-tenant metadata index must not stop at 10k rows."""

    def test_window_past_max_result_window_returns_the_right_rows(self):
        conn = _make_os_connection()
        index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 2500)
        conn.os.search.side_effect = index.search

        offset = opensearch_conn.MAX_RESULT_WINDOW
        limit = 1000
        res = _search_page(conn, offset=offset, limit=limit)

        returned = [hit["_id"] for hit in res["hits"]["hits"]]
        assert returned == index.ids[offset : offset + limit]

    def test_every_request_of_a_deep_window_stays_within_the_result_window(self):
        conn = _make_os_connection()
        index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 2500)
        conn.os.search.side_effect = index.search

        _search_page(conn, offset=opensearch_conn.MAX_RESULT_WINDOW, limit=1000)

        assert len(index.bodies) > 1, "a deep window must be walked in pages"
        assert all("from" not in body for body in index.bodies)
        assert all(body["size"] <= opensearch_conn.SEARCH_AFTER_BATCH_SIZE for body in index.bodies)
        assert [body for body in index.bodies[1:] if "search_after" not in body] == []

    def test_pages_walked_past_to_reach_the_offset_do_not_ship_their_documents(self):
        """The skipped hits are discarded, so transferring their _source is pure cost.

        The returned window still carries its _source: a walk that dropped it there
        would hand DocMetadataService rows with no fields.
        """
        conn = _make_os_connection()
        index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 2500)
        conn.os.search.side_effect = index.search

        offset = opensearch_conn.MAX_RESULT_WINDOW
        limit = 1000
        res = _search_page(conn, offset=offset, limit=limit)

        source_flags = [call.kwargs["_source"] for call in conn.os.search.call_args_list]
        assert len(source_flags) == offset // opensearch_conn.SEARCH_AFTER_BATCH_SIZE + 1
        assert source_flags == [False] * (len(source_flags) - 1) + [True]
        assert all("_source" in hit for hit in res["hits"]["hits"])
        assert [hit["_source"]["id"] for hit in res["hits"]["hits"]] == index.ids[offset : offset + limit]

    def test_the_last_page_of_the_index_ends_the_walk(self):
        """A window that runs off the end returns what exists, without looping."""
        conn = _make_os_connection()
        index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 500)
        conn.os.search.side_effect = index.search

        offset = opensearch_conn.MAX_RESULT_WINDOW
        res = _search_page(conn, offset=offset, limit=1000)

        returned = [hit["_id"] for hit in res["hits"]["hits"]]
        assert returned == index.ids[offset:]

    def test_total_and_aggregations_survive_the_walk(self):
        conn = _make_os_connection()
        index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 2500)
        conn.os.search.side_effect = index.search

        res = conn.search(
            select_fields=["*"],
            highlight_fields=[],
            condition={"kb_id": "kb1"},
            match_expressions=[],
            order_by=OrderByExpr().asc("id"),
            offset=opensearch_conn.MAX_RESULT_WINDOW,
            limit=1000,
            index_names="ragflow_doc_meta_t1",
            knowledgebase_ids=["kb1"],
            agg_fields=["kb_id"],
        )

        assert res["hits"]["total"]["value"] == len(index.ids)
        assert res["aggregations"]["aggs_kb_id"]["buckets"][0]["key"] == "kb1"
        # Only the page kept as the response template pays for the aggregation.
        assert [body for body in index.bodies if "aggs" in body] == [index.bodies[0]]

    def test_a_shallow_window_keeps_using_from_size(self):
        conn = _make_os_connection()
        index = _FakeIndex(500)
        conn.os.search.side_effect = index.search

        res = _search_page(conn, offset=0, limit=10)

        assert len(index.bodies) == 1
        assert "search_after" not in index.bodies[0]
        assert [hit["_id"] for hit in res["hits"]["hits"]] == index.ids[:10]

    def test_deep_pagination_flag_opts_in_without_a_deep_window(self):
        conn = _make_os_connection()
        index = _FakeIndex(500)
        conn.os.search.side_effect = index.search

        res = _search_page(conn, offset=0, limit=10, deep_pagination=True)

        assert "from" not in index.bodies[0]
        assert [hit["_id"] for hit in res["hits"]["hits"]] == index.ids[:10]

    def test_an_order_by_that_builds_no_sort_clause_does_not_opt_in(self, caplog):
        """The gate reads the built sort clauses, not the order_by object.

        An empty ``OrderByExpr`` is still truthy, so gating on the argument
        would send a search_after query with nothing to page on.
        """
        conn = _make_os_connection()
        index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 2500)
        conn.os.search.side_effect = index.search

        assert bool(OrderByExpr()) is True, "an empty OrderByExpr must stay truthy for this test to mean anything"

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME), pytest.raises(opensearchpy.RequestError):
            _search_page(conn, offset=opensearch_conn.MAX_RESULT_WINDOW, limit=1000, order_by=OrderByExpr())

        assert all("search_after" not in body for body in index.bodies)
        assert any("search_after paging needs a sort clause" in record.getMessage() for record in caplog.records)

    def test_a_knn_leg_does_not_opt_in(self, caplog):
        """knn hits are score-ordered and carry no sort value to page on."""
        conn = _make_os_connection()
        index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 2500)
        conn.os.search.side_effect = index.search
        dense = MatchDenseExpr(
            vector_column_name="q_1024_vec",
            embedding_data=[0.1] * 8,
            embedding_data_type="float",
            distance_type="cosine",
            topn=5,
            extra_options={"similarity": 0.0},
        )

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME), pytest.raises(opensearchpy.RequestError):
            conn.search(
                select_fields=["*"],
                highlight_fields=[],
                condition={"kb_id": "kb1"},
                match_expressions=[dense],
                order_by=OrderByExpr().asc("id"),
                offset=opensearch_conn.MAX_RESULT_WINDOW,
                limit=1000,
                index_names="ragflow_doc_meta_t1",
                knowledgebase_ids=["kb1"],
            )

        assert all("search_after" not in body for body in index.bodies)
        assert any("no knn leg" in record.getMessage() for record in caplog.records)

    def test_a_page_without_a_sort_key_fails_loudly(self):
        """Silently truncating the window is the failure this guards against."""
        conn = _make_os_connection()
        index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 2500, with_sort=False)
        conn.os.search.side_effect = index.search

        with pytest.raises(Exception, match="carries no sort values"):
            _search_page(conn, offset=opensearch_conn.MAX_RESULT_WINDOW, limit=1000)

    def test_a_timed_out_page_is_retried_and_then_raised(self):
        """A partial result is a timeout, not a short window."""
        conn = _make_os_connection()
        fake_index = _FakeIndex(opensearch_conn.MAX_RESULT_WINDOW + 2500)

        def _timing_out(index=None, body=None, **kwargs):
            res = fake_index.search(index=index, body=body, **kwargs)
            res["timed_out"] = True
            return res

        conn.os.search.side_effect = _timing_out

        with pytest.raises(Exception, match="OSConnection.search timeout"):
            _search_page(conn, offset=opensearch_conn.MAX_RESULT_WINDOW, limit=1000)

        assert conn.os.search.call_count == opensearch_conn.ATTEMPT_TIME, "the walk must restart once per attempt"
