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

"""Event-loop tests for the model rerank in ``rag.nlp.search.Dealer.retrieval``.

A rerank is a blocking call to a remote model that takes seconds. The API
server runs one event loop per process, so the rerank must run off that loop,
and a request cancelled by its client must release the loop without waiting
for the rerank to return.
"""

import asyncio
import threading

import numpy as np
import pytest

# `rag.nlp.query` imports `rag.utils.redis_conn`, which imports `common.settings`,
# which imports `rag.utils.redis_conn` back. That cycle only resolves when
# `common.settings` is initialised first -- which is what the API server and task
# executor do at startup, so importing it here reproduces the runtime order.
import common.settings  # noqa: F401

from rag.nlp.search import Dealer, settings

# Bounds every wait so a rerank stuck on the loop fails the test instead of hanging it.
WAIT_SECONDS = 5


class _BlockingRerank:
    """Reranker double that blocks until the test releases it."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.thread_id = None

    def similarity(self, query, docs):
        self.thread_id = threading.get_ident()
        self.started.set()
        self.release.wait(WAIT_SECONDS)
        self.finished.set()
        return np.zeros(len(docs)), 0


def _search_result():
    fields = {
        "chunk-0": {
            "_score": 1.0,
            "content_ltks": "pannenhilf kommt",
            "content_with_weight": "Die Pannenhilfe kommt.",
            "doc_id": "doc-1",
            "docnm_kwd": "doc",
            "kb_id": "kb-1",
        }
    }
    return Dealer.SearchResult(total=1, ids=["chunk-0"], query_vector=[0.1], field=fields, highlight={})


def _retrieve(monkeypatch, rerank_mdl):
    """Dealer.retrieval with the doc store faked and the real model-rerank branch."""

    async def fake_search(self, req, *args, **search_kwargs):
        return _search_result()

    async def keep_all_chunks(self, search_result):
        return search_result

    class _Queryer:
        def question(self, query):
            return None, ["pannenhilf"]

        def token_similarity(self, keywords, ins_tw):
            return [0.0] * len(ins_tw)

    monkeypatch.setattr(Dealer, "search", fake_search)
    monkeypatch.setattr(Dealer, "_prune_deleted_chunks", keep_all_chunks)
    monkeypatch.setattr(settings, "DOC_ENGINE_INFINITY", False, raising=False)
    monkeypatch.setattr(settings, "DOC_ENGINE_OCEANBASE", False, raising=False)
    monkeypatch.setattr(settings, "DOC_ENGINE_SERENEDB", False, raising=False)
    monkeypatch.setattr(settings, "DOC_ENGINE_GAUSSDB", False, raising=False)

    dealer = Dealer.__new__(Dealer)
    dealer.qryr = _Queryer()
    # Tag/pagerank feature scoring is out of scope here.
    dealer._rank_feature_scores = lambda rank_feature, sres: np.zeros(len(sres.ids))
    return dealer.retrieval(
        question="wie lange dauert die pannenhilfe",
        embd_mdl=object(),
        tenant_ids=["tenant-1"],
        kb_ids=["kb-1"],
        page=1,
        page_size=10,
        similarity_threshold=0.0,
        aggs=False,
        rerank_mdl=rerank_mdl,
    )


async def _until_rerank_started(rerank_mdl):
    """Return once the loop sees the rerank in progress; it only can if the rerank runs elsewhere."""
    assert await asyncio.to_thread(rerank_mdl.started.wait, WAIT_SECONDS)


@pytest.mark.p1
async def test_the_event_loop_keeps_running_while_a_rerank_is_in_progress(monkeypatch):
    rerank_mdl = _BlockingRerank()
    retrieval = asyncio.create_task(_retrieve(monkeypatch, rerank_mdl))
    try:
        await _until_rerank_started(rerank_mdl)

        assert not rerank_mdl.finished.is_set()
        assert rerank_mdl.thread_id != threading.get_ident()
    finally:
        rerank_mdl.release.set()

    ranks = await retrieval
    assert [chunk["chunk_id"] for chunk in ranks["chunks"]] == ["chunk-0"]


@pytest.mark.p1
async def test_a_cancelled_retrieval_returns_without_waiting_for_the_rerank(monkeypatch):
    rerank_mdl = _BlockingRerank()
    retrieval = asyncio.create_task(_retrieve(monkeypatch, rerank_mdl))
    try:
        await _until_rerank_started(rerank_mdl)

        retrieval.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retrieval

        assert not rerank_mdl.finished.is_set()
    finally:
        rerank_mdl.release.set()
        # The abandoned rerank still runs to the end in its worker; let it finish before the next test.
        await asyncio.to_thread(rerank_mdl.finished.wait, WAIT_SECONDS)
