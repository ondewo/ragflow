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
"""
Custom per-model HTTP headers on the OpenAI-API-compatible clients.

A self-hosted chat, embedding or rerank endpoint is frequently published behind
a gateway that demands a header of its own on top of the bearer key. The three
client classes therefore accept a ``default_headers`` mapping and hand it to the
HTTP layer they use: the OpenAI SDK clients for chat and embedding, the plain
``requests`` header dict for rerank.

The feature is opt-in, and both halves of that are pinned here: a configured
mapping has to arrive at the HTTP layer, and an absent one must leave the client
constructed exactly as every existing provider expects it.
"""

import sys
import types

import pytest

try:  # the Zhipu SDK is an optional provider dependency
    import zai  # noqa: F401
except ModuleNotFoundError:
    # `rag.llm` imports every provider module eagerly, so a single missing
    # optional SDK makes the whole registry unimportable. Stand in for it:
    # nothing here touches Zhipu.
    _zai = types.ModuleType("zai")
    _zai.ZhipuAiClient = object
    sys.modules["zai"] = _zai

from rag.llm import chat_model, embedding_model, rerank_model
from rag.llm.chat_model import OpenAI_APIChat
from rag.llm.embedding_model import OpenAI_APIEmbed
from rag.llm.rerank_model import OpenAI_APIRerank

pytestmark = pytest.mark.p1

BASE_URL = "http://vllm.internal:8000/v1"
GATEWAY_HEADERS = {"X-Gateway-Token": "abc123", "X-Tenant": "oeamtc"}


@pytest.fixture
def openai_calls(monkeypatch):
    """Record the constructor kwargs of every OpenAI SDK client the chat and embedding classes build."""
    calls: list[dict] = []

    class RecordingOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    # Patch the module objects rather than dotted-path strings: resolving
    # "rag.llm.chat_model.OpenAI" walks `rag.llm` by attribute, and a test that
    # ran earlier in the session may have dropped the submodule attribute off
    # the package while leaving the module itself imported.
    monkeypatch.setattr(chat_model, "OpenAI", RecordingOpenAI)
    monkeypatch.setattr(chat_model, "AsyncOpenAI", RecordingOpenAI)
    monkeypatch.setattr(embedding_model, "OpenAI", RecordingOpenAI)
    return calls


@pytest.fixture
def rerank_posts(monkeypatch):
    """Capture the kwargs of every rerank HTTP POST instead of issuing it."""
    posts: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    def fake_post(url, **kwargs):
        posts.append({"url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setattr(rerank_model.requests, "post", fake_post)
    return posts


def test_chat_client_forwards_configured_headers(openai_calls):
    OpenAI_APIChat("key", "qwen3", BASE_URL, default_headers=dict(GATEWAY_HEADERS))

    assert len(openai_calls) == 2  # the sync and the async client
    for call in openai_calls:
        assert call["default_headers"] == GATEWAY_HEADERS


def test_chat_client_without_headers_passes_no_such_kwarg(openai_calls):
    OpenAI_APIChat("key", "qwen3", BASE_URL)

    assert openai_calls
    for call in openai_calls:
        assert "default_headers" not in call


def test_chat_client_treats_an_empty_mapping_as_absent(openai_calls):
    OpenAI_APIChat("key", "qwen3", BASE_URL, default_headers={})

    assert openai_calls
    for call in openai_calls:
        assert "default_headers" not in call


def test_chat_client_keeps_its_other_constructor_arguments(openai_calls):
    mdl = OpenAI_APIChat("key", "qwen3___VLLM", BASE_URL, default_headers=dict(GATEWAY_HEADERS))

    assert mdl.model_name == "qwen3"
    assert openai_calls[0]["api_key"] == "key"
    assert openai_calls[0]["base_url"] == BASE_URL


def test_embedding_client_forwards_configured_headers(openai_calls):
    OpenAI_APIEmbed("key", "bge-m3", BASE_URL, default_headers=dict(GATEWAY_HEADERS))

    assert len(openai_calls) == 1
    assert openai_calls[0]["default_headers"] == GATEWAY_HEADERS


def test_embedding_client_without_headers_passes_no_such_kwarg(openai_calls):
    OpenAI_APIEmbed("key", "bge-m3", BASE_URL)

    assert openai_calls == [{"api_key": "key", "base_url": BASE_URL}]


def test_embedding_client_treats_an_empty_mapping_as_absent(openai_calls):
    OpenAI_APIEmbed("key", "bge-m3", BASE_URL, default_headers={})

    assert openai_calls == [{"api_key": "key", "base_url": BASE_URL}]


def test_rerank_merges_configured_headers_into_its_header_dict():
    mdl = OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL, default_headers=dict(GATEWAY_HEADERS))

    assert mdl.headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer key",
        **GATEWAY_HEADERS,
    }


def test_rerank_without_headers_keeps_the_exact_default_set():
    mdl = OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL)

    assert mdl.headers == {"Content-Type": "application/json", "Authorization": "Bearer key"}


def test_rerank_treats_an_empty_mapping_as_absent():
    mdl = OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL, default_headers={})

    assert mdl.headers == {"Content-Type": "application/json", "Authorization": "Bearer key"}


def test_rerank_sends_the_configured_headers_on_the_wire(rerank_posts):
    mdl = OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL, default_headers=dict(GATEWAY_HEADERS))

    mdl.similarity("wie wechsle ich einen reifen", ["reifenwechsel anleitung"])

    assert len(rerank_posts) == 1
    sent = rerank_posts[0]["headers"]
    assert sent == mdl.headers
    for name, value in GATEWAY_HEADERS.items():
        assert sent[name] == value


def test_rerank_keeps_max_token_positional():
    """``FuturMixRerank`` and ``MWSRerank`` pass ``max_token`` positionally; the new kwarg must not shift it."""
    mdl = OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL, 1024)

    assert mdl.max_token == 1024
