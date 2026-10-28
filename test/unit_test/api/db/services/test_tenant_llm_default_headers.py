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
Custom per-model HTTP headers on the read path that builds a model client.

``TenantLLMService.model_instance`` is the single place the product turns a
resolved model config into a client, so it is where the ``default_headers`` a
tenant configured for a model instance are validated and handed on.

Two properties matter. Validation has to happen before any client exists,
because the header names and values arrive through the tenant API and a value
carrying CR/LF would inject further headers into every request made with that
model. And the kwarg has to stay strictly opt-in: most provider classes take no
``default_headers`` argument at all, so an unconfigured model must be
constructed with exactly the arguments it was constructed with before.
"""

import sys
import types
from unittest.mock import patch

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

import rag.llm
from api.db.services.tenant_llm_service import TenantLLMService, validate_default_headers

pytestmark = pytest.mark.p1

GATEWAY_HEADERS = {"X-Gateway-Token": "abc123", "X-Tenant": "oeamtc"}


class RecordingModel:
    """Provider stand-in that records exactly how it was constructed.

    ``default_headers`` is named explicitly, exactly as the real header-capable
    clients name it (``OpenAI_APIEmbed``, ``OpenAI_APIRerank``). A stand-in that
    only absorbed it into ``**kwargs`` would be judged incapable by
    ``client_sends_default_headers`` -- correctly, since absorbing a keyword is no
    evidence of sending it -- and these tests would then exercise the skip path
    while reading as coverage of the delivery path.
    """

    instances: list["RecordingModel"] = []

    def __init__(self, key, model_name, default_headers=None, **kwargs):
        self.key = key
        self.model_name = model_name
        self.kwargs = dict(kwargs)
        if default_headers is not None:
            self.kwargs["default_headers"] = default_headers
        type(self).instances.append(self)


@pytest.fixture
def registered_provider(monkeypatch):
    """Register ``RecordingModel`` as the provider of every model type and reset its log."""
    RecordingModel.instances = []
    for registry in (rag.llm.ChatModel, rag.llm.EmbeddingModel, rag.llm.RerankModel):
        monkeypatch.setitem(registry, "Recording", RecordingModel)
    return RecordingModel


def model_config(model_type: str, **extra) -> dict:
    config = {
        "llm_factory": "Recording",
        "api_key": "key",
        "llm_name": "qwen3",
        "api_base": "http://vllm.internal:8000/v1",
        "model_type": model_type,
        "max_tokens": 8192,
    }
    config.update(extra)
    return config


def build(config: dict, **kwargs):
    """Call ``model_instance`` without a database behind it."""
    with patch("api.db.db_models.DB.connect"), patch("api.db.db_models.DB.close"):
        return TenantLLMService.model_instance(config, **kwargs)


# The validator


def test_a_legal_header_mapping_is_returned_unchanged():
    headers = dict(GATEWAY_HEADERS)

    assert validate_default_headers(headers) is headers


def test_no_headers_at_all_is_legal():
    assert validate_default_headers(None) is None


def test_tab_is_a_legal_header_value_character():
    assert validate_default_headers({"X-A": "a\tb"}) == {"X-A": "a\tb"}


def test_an_empty_header_value_is_legal():
    """RFC 7230 allows an empty field-value, and the provider write path stores one, so the read path must accept it."""
    assert validate_default_headers({"X-A": ""}) == {"X-A": ""}


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param("X-A: b", id="not-a-mapping"),
        pytest.param({"X-A": None}, id="value-not-a-string"),
        pytest.param({7: "b"}, id="name-not-a-string"),
        pytest.param({"": "b"}, id="empty-name"),
        pytest.param({"X A": "b"}, id="space-in-name"),
        pytest.param({"X-A:": "b"}, id="colon-in-name"),
        pytest.param({"X-A\n": "b"}, id="newline-in-name"),
        pytest.param({"X-A": "b\r\nX-Injected: yes"}, id="crlf-in-value"),
        pytest.param({"X-A": "b\nX-Injected: yes"}, id="lf-in-value"),
        pytest.param({"X-A": "b\x00"}, id="nul-in-value"),
        pytest.param({"X-A": "schöne-grüße"}, id="non-ascii-in-value"),
        pytest.param({"Host": "elsewhere.example"}, id="transport-managed-host"),
        pytest.param({"Content-Length": "0"}, id="transport-managed-content-length"),
        pytest.param({"X-A": "b", "x-a": "c"}, id="case-insensitive-duplicate"),
        pytest.param({f"X-{index}": "b" for index in range(33)}, id="too-many-headers"),
        pytest.param({"X-A": "b" * 4097}, id="value-too-long"),
        pytest.param({"X" * 129: "b"}, id="name-too-long"),
    ],
)
def test_an_unusable_header_mapping_is_rejected(headers):
    with pytest.raises(ValueError):
        validate_default_headers(headers)


def test_the_longest_legal_name_and_value_are_accepted():
    headers = {"X" * 128: "b" * 4096}

    assert validate_default_headers(headers) is headers


# model_instance


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_configured_headers_reach_the_client(registered_provider, model_type):
    build(model_config(model_type, default_headers=dict(GATEWAY_HEADERS)))

    assert registered_provider.instances[0].kwargs["default_headers"] == GATEWAY_HEADERS


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_no_configured_headers_passes_no_such_kwarg(registered_provider, model_type):
    build(model_config(model_type))

    assert "default_headers" not in registered_provider.instances[0].kwargs


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_an_empty_header_mapping_passes_no_such_kwarg(registered_provider, model_type):
    build(model_config(model_type, default_headers={}))

    assert "default_headers" not in registered_provider.instances[0].kwargs


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_an_illegal_header_value_is_rejected_before_any_client_is_built(registered_provider, model_type):
    config = model_config(model_type, default_headers={"X-A": "b\r\nX-Injected: yes"})

    with pytest.raises(ValueError):
        build(config)

    assert registered_provider.instances == []


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_an_explicit_caller_kwarg_wins_over_the_configured_headers(registered_provider, model_type):
    """The chat branch forwards caller kwargs as well, so the two sources must not collide either."""
    caller_headers = {"X-From-Caller": "yes"}

    build(model_config(model_type, default_headers=dict(GATEWAY_HEADERS)), default_headers=caller_headers)

    assert registered_provider.instances[0].kwargs["default_headers"] == caller_headers


def test_the_other_constructor_arguments_are_untouched(registered_provider):
    build(model_config("rerank", default_headers=dict(GATEWAY_HEADERS)))

    built = registered_provider.instances[0]
    assert built.key == "key"
    assert built.model_name == "qwen3"
    assert built.kwargs["base_url"] == "http://vllm.internal:8000/v1"
    assert built.kwargs["max_token"] == 8192
