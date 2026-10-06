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
"""Unit tests for near-duplicate chunk suppression in Dealer.retrieval.

A web-crawl corpus repeats the same passage across many pages, so an
unsuppressed top-N can be several copies of one fact. Dealer collapses such a
group with a shingle Jaccard comparison, either before reranking (so the
reranker spends its candidate budget on distinct chunks) or after scoring, on
the ranked list.
"""

import sys
import types

import numpy as np
import pytest

# Stub the heavy / circular-importing dependencies before importing search,
# mirroring test_search_pagination.py so the module imports in isolation.
_fake_query = types.ModuleType("rag.nlp.query")


class _DummyFulltextQueryer:
    pass


_fake_query.FulltextQueryer = _DummyFulltextQueryer
_fake_tokenizer = types.ModuleType("rag.nlp.rag_tokenizer")
sys.modules.setdefault("rag.nlp.query", _fake_query)
sys.modules.setdefault("rag.nlp.rag_tokenizer", _fake_tokenizer)
sys.modules.setdefault("common.settings", types.ModuleType("common.settings"))

from rag.nlp.search import DEDUP_SHINGLE_SIZE, Dealer, settings  # noqa: E402

# Shingle Jaccard of these three: TEXT/TEXT_NEAR = 0.6, TEXT/TEXT_OTHER = 0.0.
TEXT = "Die Pannenhilfe kommt innerhalb von 30 Minuten zum liegengebliebenen Fahrzeug."
TEXT_NEAR = "Die Pannenhilfe kommt innerhalb von 30 Minuten zum liegengebliebenen Fahrzeug und schleppt es ab."
TEXT_OTHER = "Die Mitgliedschaft verlaengert sich automatisch um ein weiteres Jahr."


def _search_result(texts):
    """A SearchResult over `texts`, in ranking order, scored descending."""
    ids = [f"chunk-{i}" for i in range(len(texts))]
    fields = {
        chunk_id: {
            "_score": float(len(texts) - i),
            "content_ltks": text,
            "content_with_weight": text,
            "doc_id": "doc-1",
            "docnm_kwd": "doc",
            "kb_id": "kb-1",
        }
        for i, (chunk_id, text) in enumerate(zip(ids, texts))
    }
    return Dealer.SearchResult(total=len(ids), ids=ids, query_vector=[0.1], field=fields, highlight={})


async def _retrieve(monkeypatch, texts, engine="infinity", **kwargs):
    """Run Dealer.retrieval over `texts` with the doc store and the scoring branch faked."""

    async def fake_search(self, req, *args, **search_kwargs):
        return _search_result(texts)

    async def keep_all_chunks(self, search_result):
        return search_result

    def fake_rerank_with_knn(self, sres, question, knn_scores, *args, **rerank_kwargs):
        scores = np.array([float(len(sres.ids) - i) for i in range(len(sres.ids))])
        return scores, scores, scores

    async def no_knn_scores(self, sres, idx_names, kb_ids):
        return {}

    monkeypatch.setattr(Dealer, "search", fake_search)
    monkeypatch.setattr(Dealer, "_prune_deleted_chunks", keep_all_chunks)
    monkeypatch.setattr(Dealer, "_knn_scores", no_knn_scores)
    monkeypatch.setattr(Dealer, "rerank_with_knn", fake_rerank_with_knn)
    # Infinity scores in the engine, the OpenSearch path merges a second KNN call locally.
    monkeypatch.setattr(settings, "DOC_ENGINE_INFINITY", engine == "infinity", raising=False)
    monkeypatch.setattr(settings, "DOC_ENGINE_OCEANBASE", False, raising=False)
    monkeypatch.setattr(settings, "DOC_ENGINE_SERENEDB", False, raising=False)
    monkeypatch.setattr(settings, "DOC_ENGINE_GAUSSDB", False, raising=False)

    return await Dealer.__new__(Dealer).retrieval(
        question="wie lange dauert die pannenhilfe",
        embd_mdl=object(),
        tenant_ids=["tenant-1"],
        kb_ids=["kb-1"],
        page=1,
        page_size=10,
        similarity_threshold=0.0,
        aggs=False,
        **kwargs,
    )


def test_dedup_shingles_of_a_text_without_words_is_empty():
    assert Dealer._dedup_shingles("") == set()
    assert Dealer._dedup_shingles("  ...  ") == set()


def test_dedup_shingles_of_a_short_text_is_the_whole_text():
    # Fewer words than the shingle size: the signature is the single joined text,
    # lower-cased and stripped of everything the word pattern does not match.
    assert Dealer._dedup_shingles("Hallo, Welt!") == {"hallo welt"}


def test_dedup_shingles_slide_over_the_words():
    words = [f"wort{i}" for i in range(DEDUP_SHINGLE_SIZE + 2)]
    shingles = Dealer._dedup_shingles(" ".join(words))

    assert len(shingles) == len(words) - DEDUP_SHINGLE_SIZE + 1
    assert " ".join(words[:DEDUP_SHINGLE_SIZE]) in shingles
    assert " ".join(words[-DEDUP_SHINGLE_SIZE:]) in shingles


def test_dedup_shingles_ignore_case_and_punctuation():
    assert Dealer._dedup_shingles("Der Wagen, der fährt!") == Dealer._dedup_shingles("der wagen der fährt")


@pytest.mark.parametrize("threshold", [0.0, -1.0])
def test_suppress_near_duplicates_is_a_no_op_without_a_threshold(threshold):
    texts = [TEXT, TEXT, TEXT]

    assert Dealer._suppress_near_duplicates(texts, threshold) == list(range(len(texts)))


def test_suppress_near_duplicates_collapses_identical_texts():
    assert Dealer._suppress_near_duplicates([TEXT, TEXT, TEXT], 0.8) == [0]


def test_suppress_near_duplicates_keeps_the_best_ranked_member_of_a_group():
    # `texts` is in ranking order, so the survivor of the TEXT/TEXT_NEAR group
    # must be index 0; TEXT_OTHER shares no shingle with either and survives.
    keep = Dealer._suppress_near_duplicates([TEXT, TEXT_NEAR, TEXT_OTHER], 0.5)

    assert keep == [0, 2]


def test_suppress_near_duplicates_keeps_a_pair_below_the_threshold():
    # The same pair at a threshold above their 0.6 similarity: both survive.
    assert Dealer._suppress_near_duplicates([TEXT, TEXT_NEAR], 0.8) == [0, 1]


def test_suppress_near_duplicates_collapses_a_pair_exactly_at_the_threshold():
    # The comparison is inclusive, so a similarity equal to the threshold already
    # counts as a duplicate. Pinning the boundary keeps a `>=` from drifting to `>`,
    # which no other case here would notice.
    similarity = 0.6
    assert Dealer._suppress_near_duplicates([TEXT, TEXT_NEAR], similarity) == [0]
    assert Dealer._suppress_near_duplicates([TEXT, TEXT_NEAR], similarity + 0.01) == [0, 1]


def test_suppress_near_duplicates_handles_empty_and_short_texts():
    # A chunk with no words has no signature and is always kept; a text shorter
    # than the shingle size is compared as a whole.
    texts = ["", "   ", "kurzer text", "kurzer text", ""]

    assert Dealer._suppress_near_duplicates(texts, 0.9) == [0, 1, 2, 4]


def test_without_near_duplicates_drops_the_later_copies():
    sres = _search_result([TEXT, TEXT, TEXT_OTHER])

    filtered = Dealer.__new__(Dealer)._without_near_duplicates(sres, 0.8)

    assert filtered.ids == ["chunk-0", "chunk-2"]
    assert filtered.total == 2
    assert sorted(filtered.field) == ["chunk-0", "chunk-2"]
    assert filtered.query_vector == sres.query_vector
    # The search result handed in stays usable for the caller that still holds it.
    assert sres.ids == ["chunk-0", "chunk-1", "chunk-2"]


def test_without_near_duplicates_returns_the_input_when_nothing_collapses():
    sres = _search_result([TEXT, TEXT_OTHER])

    assert Dealer.__new__(Dealer)._without_near_duplicates(sres, 0.8) is sres


@pytest.mark.parametrize("engine", ["infinity", "opensearch"])
async def test_retrieval_suppresses_duplicates_after_scoring(monkeypatch, engine):
    ranks = await _retrieve(monkeypatch, [TEXT, TEXT, TEXT_OTHER, TEXT], engine=engine, dedup_threshold=0.8)

    assert [chunk["chunk_id"] for chunk in ranks["chunks"]] == ["chunk-0", "chunk-2"]
    assert ranks["total"] == 2


@pytest.mark.parametrize("engine", ["infinity", "opensearch"])
async def test_retrieval_keeps_every_chunk_by_default(monkeypatch, engine):
    # Suppression is opt-in: an existing caller that passes no threshold sees
    # the full candidate list, duplicates and all.
    ranks = await _retrieve(monkeypatch, [TEXT, TEXT, TEXT_OTHER, TEXT], engine=engine)

    assert [chunk["chunk_id"] for chunk in ranks["chunks"]] == ["chunk-0", "chunk-1", "chunk-2", "chunk-3"]
    assert ranks["total"] == 4


async def test_retrieval_suppresses_duplicates_before_rerank(monkeypatch):
    # With dedup_before_rerank the reranker must never see the copies, so its
    # candidate budget is spent on distinct chunks.
    reranked = {}

    def fake_rerank_by_model(self, rerank_mdl, sres, question, *args, **kwargs):
        reranked["ids"] = list(sres.ids)
        scores = np.array([float(len(sres.ids) - i) for i in range(len(sres.ids))])
        return scores, scores, scores

    monkeypatch.setattr(Dealer, "rerank_by_model", fake_rerank_by_model)

    ranks = await _retrieve(
        monkeypatch,
        [TEXT, TEXT, TEXT_OTHER, TEXT],
        rerank_mdl=object(),
        dedup_threshold=0.8,
        dedup_before_rerank=True,
    )

    assert reranked["ids"] == ["chunk-0", "chunk-2"]
    assert [chunk["chunk_id"] for chunk in ranks["chunks"]] == ["chunk-0", "chunk-2"]
    assert ranks["total"] == 2


async def test_retrieval_reranks_every_candidate_without_dedup_before_rerank(monkeypatch):
    # The default suppresses after scoring, so the reranker still sees them all.
    reranked = {}

    def fake_rerank_by_model(self, rerank_mdl, sres, question, *args, **kwargs):
        reranked["ids"] = list(sres.ids)
        scores = np.array([float(len(sres.ids) - i) for i in range(len(sres.ids))])
        return scores, scores, scores

    monkeypatch.setattr(Dealer, "rerank_by_model", fake_rerank_by_model)

    ranks = await _retrieve(monkeypatch, [TEXT, TEXT, TEXT_OTHER, TEXT], rerank_mdl=object(), dedup_threshold=0.8)

    assert reranked["ids"] == ["chunk-0", "chunk-1", "chunk-2", "chunk-3"]
    assert [chunk["chunk_id"] for chunk in ranks["chunks"]] == ["chunk-0", "chunk-2"]
    assert ranks["total"] == 2
