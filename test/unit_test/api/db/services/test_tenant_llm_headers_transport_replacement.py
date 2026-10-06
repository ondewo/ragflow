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
Chat clients that inherit header support and then throw the header-carrying client away.

`rag.llm.chat_model.Base` consumes ``default_headers`` and hands it to the OpenAI SDK clients it builds, so
inheriting from it is what lets a chat factory send custom headers without naming the keyword. A handful of its
subclasses then overwrite ``self.client`` in their own ``__init__`` with a transport the mapping was never given:
another SDK (``mistralai``, ``replicate``, ``qianfan``, ``google.genai``, ``jina``), a second bare ``OpenAI(...)``
built without the headers, or a ``requests`` path driven by a header dict of their own.

Judging those clients capable is the exact failure the construction-site filter exists to prevent: the request
succeeds, nothing is logged, and the operator believes a gateway credential is in effect when nothing ever sent
it. So the verdict is pinned here from both ends -- every chat factory judged capable must really put a
configured mapping on the client it ends up using, and the classes that replace their transport must be skipped
with the warning instead.
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

from unittest.mock import patch

import rag.llm
from api.db.services.tenant_llm_service import TenantLLMService, _transport_replacing_clients, client_sends_default_headers, default_headers_kwarg
from rag.llm import chat_model
from rag.llm.chat_model import BaiduYiyanChat, LmStudioChat, LocalAIChat, ReplicateChat

pytestmark = pytest.mark.p1

BASE_URL = "http://vllm.internal:8000/v1"
GATEWAY_HEADERS = {"X-Gateway-Token": "abc123", "X-Tenant": "oeamtc"}
# `SparkChat` asserts its model name against a fixed list; every other chat class
# treats the name as opaque, so one of Spark's own names drives all of them.
MODEL_NAME = "Spark-Max"


class FakeOpenAI:
    """Stand in for the OpenAI SDK client, recording the keywords it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def with_options(self, **kwargs):
        """Mirror the SDK, whose ``copy()`` MERGES a new ``default_headers`` into the existing ones."""
        merged = dict(self.kwargs)
        for key, value in kwargs.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return FakeOpenAI(**merged)


@pytest.fixture(autouse=True)
def fake_openai_sdk(monkeypatch):
    """Build every OpenAI-SDK-backed chat client against the recording stand-in."""
    monkeypatch.setattr(chat_model, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(chat_model, "AsyncOpenAI", FakeOpenAI)


def capable_chat_factories() -> dict[str, type]:
    """The registered chat factories the construction-site filter is willing to hand the headers to."""
    return {factory: client_cls for factory, client_cls in rag.llm.ChatModel.items() if client_sends_default_headers(client_cls)}


def transport_replacing_factories() -> dict[str, type]:
    """The registered chat factories whose client rebuilds its transport, derived from the live registry."""
    replacing = _transport_replacing_clients()
    return {factory: client_cls for factory, client_cls in rag.llm.ChatModel.items() if any(klass in replacing for klass in client_cls.__mro__)}


def headers_on(client) -> dict[str, str] | None:
    """The ``default_headers`` the client a chat instance actually uses was constructed with, if any."""
    if not isinstance(client, FakeOpenAI):
        return None

    return client.kwargs.get("default_headers")


def model_config(factory: str, **extra) -> dict:
    config = {
        "llm_factory": factory,
        "api_key": "key",
        "llm_name": MODEL_NAME,
        "api_base": BASE_URL,
        "model_type": "chat",
        "max_tokens": 8192,
    }
    config.update(extra)
    return config


def build(config: dict, **kwargs):
    """Call ``model_instance`` without a database behind it."""
    with patch("api.db.db_models.DB.connect"), patch("api.db.db_models.DB.close"):
        return TenantLLMService.model_instance(config, **kwargs)


# The capable verdict has to be true of the client the instance ends up using


def test_every_chat_factory_judged_capable_puts_the_headers_on_the_client_it_uses():
    """Derived from the live registry, so a provider added upstream cannot quietly become an accepted-and-dropped header."""
    capable = capable_chat_factories()
    assert capable, "no chat factory is judged capable -- the corpus check would be vacuous"

    dropped: dict[str, str] = {}
    for factory, client_cls in sorted(capable.items()):
        instance = client_cls("key", MODEL_NAME, base_url=BASE_URL, default_headers=dict(GATEWAY_HEADERS))
        sent = headers_on(getattr(instance, "client", None))
        if sent is None or not set(GATEWAY_HEADERS.items()) <= set(sent.items()):
            dropped[factory] = f"{client_cls.__name__} -> client={type(getattr(instance, 'client', None)).__name__} headers={sent}"

    assert dropped == {}, f"judged capable but the headers never reach the client in use: {dropped}"


@pytest.mark.parametrize(
    ("factory", "expected_client"),
    [
        ("LocalAI", "LocalAIChat"),
        ("LM-Studio", "LmStudioChat"),
        ("Replicate", "ReplicateChat"),
        ("BaiduYiyan", "BaiduYiyanChat"),
        ("Mistral", "MistralChat"),
        ("Google Cloud", "GoogleChat"),
        ("MWS", "MWSChat"),
    ],
)
def test_a_chat_factory_that_rebuilds_its_transport_is_not_handed_the_headers(factory, expected_client, caplog):
    client_cls = rag.llm.ChatModel[factory]
    assert client_cls.__name__ == expected_client

    with caplog.at_level("WARNING"):
        kwarg = default_headers_kwarg(client_cls=client_cls, default_headers=dict(GATEWAY_HEADERS), factory_name=factory)

    assert kwarg == {}
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert factory in warnings[0]
    assert expected_client in warnings[0]


def test_the_transport_replacing_set_is_not_stale():
    """Every named class must still be reachable through the registry, or the entry is protecting nothing."""
    factories = transport_replacing_factories()

    assert sorted(factories) == ["BaiduYiyan", "Google Cloud", "LM-Studio", "LocalAI", "MWS", "Mistral", "Replicate"]
    assert set(factories) & set(capable_chat_factories()) == set()


def test_a_subclass_of_a_transport_replacing_client_is_judged_incapable():
    """The walk starts at the concrete class, so a client inheriting the replaced transport inherits the verdict too."""

    class DerivedLocalAIChat(LocalAIChat):
        pass

    assert client_sends_default_headers(DerivedLocalAIChat) is False


# Why those classes are named: the headers really do not survive their constructor


@pytest.mark.parametrize("client_cls", [LocalAIChat, LmStudioChat])
def test_a_client_that_rebuilds_a_bare_openai_client_loses_the_headers(client_cls):
    instance = client_cls("key", MODEL_NAME, base_url=BASE_URL, default_headers=dict(GATEWAY_HEADERS))

    assert isinstance(instance.client, FakeOpenAI)
    assert headers_on(instance.client) is None


@pytest.mark.parametrize("client_cls", [ReplicateChat, BaiduYiyanChat])
def test_a_client_that_swaps_in_another_sdk_loses_the_headers(client_cls):
    instance = client_cls("key", MODEL_NAME, base_url=BASE_URL, default_headers=dict(GATEWAY_HEADERS))

    assert not isinstance(instance.client, FakeOpenAI)


# The production path, and the no-headers path it must leave alone


def test_model_instance_builds_a_transport_replacing_chat_client_without_the_keyword(caplog):
    with caplog.at_level("WARNING"):
        built = build(model_config("LocalAI", default_headers=dict(GATEWAY_HEADERS)))

    assert type(built) is LocalAIChat
    assert headers_on(built.client) is None
    assert len([record for record in caplog.records if record.levelname == "WARNING"]) == 1


def test_model_instance_builds_a_transport_replacing_chat_client_unchanged_when_no_headers_are_configured(caplog):
    with caplog.at_level("WARNING"):
        built = build(model_config("LocalAI"))

    assert type(built) is LocalAIChat
    assert headers_on(built.client) is None
    assert [record for record in caplog.records if record.levelname == "WARNING"] == []
