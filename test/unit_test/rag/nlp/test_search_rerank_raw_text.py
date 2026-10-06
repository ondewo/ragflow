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

"""What ``Dealer.rerank_by_model`` hands the reranker model.

A cross-encoder is trained on natural language, so it must see the chunk's raw
text rather than the whitespace-joined tokenizer output -- the tokenizer splits
compounds the model expects whole. The token list keeps feeding the
term-similarity leg of the blend.
"""

from unittest.mock import MagicMock

import numpy as np

# `rag.nlp.query` imports `rag.utils.redis_conn`, which imports `common.settings`,
# which imports `rag.utils.redis_conn` back. That cycle only resolves when
# `common.settings` is initialised first -- which is what the API server and task
# executor do at startup, so importing it here reproduces the runtime order.
import common.settings  # noqa: F401

from rag.nlp.search import Dealer

RAW_TEXT = "Die Pannenhilfe kommt innerhalb von 30 Minuten zum liegengebliebenen Fahrzeug."


def _dealer(captured):
    """Dealer with ``__init__`` bypassed (it builds a real FulltextQueryer,
    which loads the tokenizer) and ``qryr`` stubbed so we can capture ins_tw."""
    dealer = Dealer.__new__(Dealer)
    qryr = MagicMock()
    qryr.question.return_value = (None, ["pannenhilfe"])

    def _token_similarity(keywords, ins_tw):
        captured["ins_tw"] = ins_tw
        return [0.5] * len(ins_tw)

    qryr.token_similarity.side_effect = _token_similarity
    dealer.qryr = qryr
    return dealer


def _sres(**field_overrides):
    field = {
        "content_ltks": "pannenhilfe kommt innerhalb 30 minuten liegengebliebene fahrzeug",
        "title_tks": "pannenhilfe ratgeber",
        "question_tks": "wie lange dauert die pannenhilfe",
        "important_kwd": ["pannenhilfe"],
        "content_with_weight": RAW_TEXT,
    }
    field.update(field_overrides)
    return Dealer.SearchResult(total=1, ids=["c1"], query_vector=[0.1, 0.2, 0.3], field={"c1": field})


def _rerank_mdl():
    rerank_mdl = MagicMock()
    rerank_mdl.similarity.return_value = (np.array([0.5]), 0)
    return rerank_mdl


def test_rerank_by_model_scores_the_raw_chunk_text():
    captured = {}
    rerank_mdl = _rerank_mdl()

    _dealer(captured).rerank_by_model(rerank_mdl, _sres(), "pannenhilfe")

    docs = rerank_mdl.similarity.call_args[0][1]
    assert docs == [RAW_TEXT]


def test_rerank_by_model_still_feeds_the_tokens_to_token_similarity():
    # The cfield tokens only drive the term-similarity leg now, so they must
    # still reach `token_similarity` with every field assembled as before.
    captured = {}

    _dealer(captured).rerank_by_model(_rerank_mdl(), _sres(), "pannenhilfe")

    tokens = captured["ins_tw"][0]
    assert "liegengebliebene" in tokens
    assert "ratgeber" in tokens
    assert "wie" in tokens


def test_rerank_by_model_falls_back_to_the_joined_tokens_without_raw_text():
    # Not every chunk carries content_with_weight; `docs` must stay non-empty,
    # because an empty document makes some reranker providers answer 400.
    captured = {}
    rerank_mdl = _rerank_mdl()
    sres = _sres()
    del sres.field["c1"]["content_with_weight"]

    _dealer(captured).rerank_by_model(rerank_mdl, sres, "pannenhilfe")

    docs = rerank_mdl.similarity.call_args[0][1]
    assert docs[0] == " ".join(captured["ins_tw"][0])


def test_rerank_by_model_falls_back_on_an_empty_raw_text():
    captured = {}
    rerank_mdl = _rerank_mdl()

    _dealer(captured).rerank_by_model(rerank_mdl, _sres(content_with_weight=""), "pannenhilfe")

    docs = rerank_mdl.similarity.call_args[0][1]
    assert docs[0] == " ".join(captured["ins_tw"][0])
