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
"""The OpenAI-compatible reranker asks the server to truncate each document.

tiktoken counts differently from the reranker's own tokenizer, so a document
truncated to ``max_token`` locally can still overflow the server's context.
``max_tokens_per_doc`` lets the server truncate with its own tokenizer.
"""

import pytest

from rag.llm import rerank_model
from rag.llm.rerank_model import FuturMixRerank, OpenAI_APIRerank

pytestmark = pytest.mark.p1

BASE_URL = "http://reranker.test/v1"
QUERY = "was beinhaltet die motorrad mitgliedschaft?"
DOCUMENTS = ["Die Motorrad-Mitgliedschaft umfasst Pannenhilfe.", "Kindersitztest 2026"]


@pytest.fixture
def rerank_posts(monkeypatch):
    """Capture the JSON body of every rerank HTTP POST instead of issuing it."""
    posts: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": [{"index": index, "relevance_score": 0.5} for index in range(len(DOCUMENTS))]}

    def fake_post(url, **kwargs):
        posts.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr(rerank_model.requests, "post", fake_post)
    return posts


def test_the_request_carries_the_document_budget(rerank_posts):
    max_token = 8192
    mdl = OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL, max_token)

    mdl.similarity(QUERY, DOCUMENTS)

    expected = max_token - rerank_model.num_tokens_from_string(QUERY) - rerank_model.RERANK_PROMPT_MARGIN_TOKENS
    assert [post["max_tokens_per_doc"] for post in rerank_posts] == [expected]


def test_a_budget_that_is_not_positive_is_not_sent(rerank_posts):
    mdl = OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL, rerank_model.RERANK_PROMPT_MARGIN_TOKENS)

    mdl.similarity(QUERY, DOCUMENTS)

    assert len(rerank_posts) == 1
    assert "max_tokens_per_doc" not in rerank_posts[0]


def test_the_budget_is_not_sent_unless_the_mode_truncates(rerank_posts, monkeypatch):
    monkeypatch.setattr(rerank_model, "RERANK_TOKEN_LIMIT_MODE", "passthrough")
    mdl = OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL)

    mdl.similarity(QUERY, DOCUMENTS)

    assert len(rerank_posts) == 1
    assert "max_tokens_per_doc" not in rerank_posts[0]


def test_futurmix_does_not_send_the_budget(rerank_posts):
    mdl = FuturMixRerank("key", "rerank-model")

    mdl.similarity(QUERY, DOCUMENTS)

    assert len(rerank_posts) == 1
    assert "max_tokens_per_doc" not in rerank_posts[0]
