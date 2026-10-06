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
Configuring custom HTTP headers on a provider whose client cannot send them.

``default_headers`` is stored per provider instance and handed to the client
``TenantLLMService.model_instance`` builds. Only a handful of the registered
provider classes put such a mapping on the wire, so the keyword has to be
filtered at that one construction site: a client that cannot take it must be
built without it and the skip logged, never crash the model it serves.

Two shapes of "cannot take it" are pinned here, because they fail differently
and only one of them is loud:

* a constructor with no matching parameter and no ``**kwargs`` -- passing the
  keyword raises ``TypeError`` and takes down every request the model serves;
* a constructor that absorbs unknown keywords into ``**kwargs`` and drops them
  -- the request succeeds without the header, so the operator believes a
  gateway credential is in effect when nothing ever sent it.

Both must end as a skip plus a warning naming the factory, which is why
collecting ``**kwargs`` is not read as support anywhere below.
"""

import inspect
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
from api.db.services.tenant_llm_service import TenantLLMService, _transport_replacing_clients, client_sends_default_headers, default_headers_kwarg
from rag.llm.chat_model import Base as ChatClientBase
from rag.llm.chat_model import LiteLLMBase
from rag.llm.embedding_model import OpenAI_APIEmbed
from rag.llm.rerank_model import OpenAI_APIRerank

pytestmark = pytest.mark.p1

BASE_URL = "http://vllm.internal:8000/v1"
GATEWAY_HEADERS = {"X-Gateway-Token": "abc123", "X-Tenant": "oeamtc"}

REGISTRIES = {
    "chat": rag.llm.ChatModel,
    "embedding": rag.llm.EmbeddingModel,
    "rerank": rag.llm.RerankModel,
}


class StrictModel:
    """A provider stand-in in the shape of the majority: no header parameter, no ``**kwargs``."""

    instances: list["StrictModel"] = []

    # `provider` is named because the chat branch of `model_instance` forwards it
    # to every chat client; leaving it out would make this stand-in fail for a
    # reason that has nothing to do with the headers under test.
    def __init__(self, key, model_name, base_url=None, max_token=None, provider=None):
        self.key = key
        self.model_name = model_name
        self.base_url = base_url
        self.max_token = max_token
        self.provider = provider
        type(self).instances.append(self)


class SwallowingModel(StrictModel):
    """A provider stand-in in the shape of ``LiteLLMBase``: absorbs unknown keywords and never sends them."""

    instances: list["SwallowingModel"] = []

    def __init__(self, key, model_name, base_url=None, max_token=None, provider=None, **kwargs):
        super().__init__(key, model_name, base_url, max_token, provider)
        self.swallowed = kwargs


class HeaderAwareModel(StrictModel):
    """A provider stand-in that names the keyword, which is how a client declares that it sends the headers."""

    instances: list["HeaderAwareModel"] = []

    def __init__(self, key, model_name, base_url=None, max_token=None, provider=None, default_headers=None):
        super().__init__(key, model_name, base_url, max_token, provider)
        self.default_headers = default_headers


@pytest.fixture(autouse=True)
def _clear_recorded_instances():
    """Keep each stand-in's construction log to the test that caused it."""
    for model in (StrictModel, SwallowingModel, HeaderAwareModel):
        model.instances = []


@pytest.fixture
def register(monkeypatch):
    """Register a stand-in class under the factory name ``Recording`` in every registry that takes headers."""

    def _register(model_cls):
        for registry in REGISTRIES.values():
            monkeypatch.setitem(registry, "Recording", model_cls)
        return model_cls

    return _register


def model_config(model_type: str, factory: str = "Recording", **extra) -> dict:
    config = {
        "llm_factory": factory,
        "api_key": "key",
        "llm_name": "qwen3",
        "api_base": BASE_URL,
        "model_type": model_type,
        "max_tokens": 8192,
    }
    config.update(extra)
    return config


def build(config: dict, **kwargs):
    """Call ``model_instance`` without a database behind it."""
    with patch("api.db.db_models.DB.connect"), patch("api.db.db_models.DB.close"):
        return TenantLLMService.model_instance(config, **kwargs)


def keyword_is_bindable(model_cls: type) -> bool:
    """Report whether ``default_headers`` can be passed to `model_cls` at all without a ``TypeError``."""
    try:
        inspect.signature(model_cls).bind_partial(default_headers={})
    except TypeError:
        return False

    return True


# The construction site: a client that cannot take the keyword


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_a_client_without_the_parameter_is_built_without_the_headers(register, model_type):
    register(StrictModel)

    built = build(model_config(model_type, default_headers=dict(GATEWAY_HEADERS)))

    assert built is StrictModel.instances[0]
    assert built.model_name == "qwen3"


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_a_client_that_only_swallows_unknown_keywords_is_built_without_the_headers(register, model_type):
    """The dropped-and-never-sent case: a request would succeed with no header, so it must be skipped instead."""
    register(SwallowingModel)

    built = build(model_config(model_type, default_headers=dict(GATEWAY_HEADERS)))

    assert "default_headers" not in built.swallowed


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_a_client_that_names_the_parameter_still_receives_the_headers(register, model_type):
    register(HeaderAwareModel)

    built = build(model_config(model_type, default_headers=dict(GATEWAY_HEADERS)))

    assert built.default_headers == GATEWAY_HEADERS


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_the_skip_is_logged_as_a_warning_naming_the_factory(register, model_type, caplog):
    register(StrictModel)

    with caplog.at_level("WARNING"):
        build(model_config(model_type, factory="Recording", default_headers=dict(GATEWAY_HEADERS)))

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "Recording" in message
    assert StrictModel.__name__ in message


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_an_unsupported_client_logs_nothing_when_no_headers_are_configured(register, model_type, caplog):
    register(StrictModel)

    with caplog.at_level("WARNING"):
        build(model_config(model_type))

    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_an_empty_header_mapping_is_not_worth_a_warning(register, model_type, caplog):
    register(StrictModel)

    with caplog.at_level("WARNING"):
        build(model_config(model_type, default_headers={}))

    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


def test_a_caller_supplied_mapping_is_filtered_on_the_same_terms(register):
    """``model_instance`` also takes the headers as a caller kwarg; that route must not bypass the filter."""
    register(StrictModel)

    built = build(model_config("chat"), default_headers={"X-From-Caller": "yes"})

    assert built is StrictModel.instances[0]


# The reported regression, on the real registry


def test_the_vllm_rerank_factory_tolerates_configured_headers():
    """``VLLM`` resolves rerank to ``CoHereRerank``, which has no header parameter and no ``**kwargs``."""
    factory = "VLLM"
    client_cls = rag.llm.RerankModel[factory]
    assert not keyword_is_bindable(client_cls)

    built = build(model_config("rerank", factory=factory, default_headers=dict(GATEWAY_HEADERS)))

    assert type(built) is client_cls


def test_a_client_that_replaces_a_header_aware_constructor_is_not_mistaken_for_one():
    """``MWSRerank`` and ``FuturMixRerank`` inherit from the one capable rerank class and then narrow it."""
    narrowed = [factory for factory, cls in rag.llm.RerankModel.items() if issubclass(cls, OpenAI_APIRerank) and cls is not OpenAI_APIRerank]
    assert narrowed, "expected at least one rerank subclass of the header-aware class"

    for factory in narrowed:
        client_cls = rag.llm.RerankModel[factory]
        assert issubclass(client_cls, OpenAI_APIRerank)
        assert not keyword_is_bindable(client_cls)
        assert not client_sends_default_headers(client_cls)


def test_the_litellm_chat_factories_are_skipped_rather_than_handed_a_header_they_drop():
    """``LiteLLMBase`` absorbs the keyword into ``**kwargs`` and never sends it, so it must not be handed one."""
    litellm_factories = [factory for factory, cls in rag.llm.ChatModel.items() if issubclass(cls, LiteLLMBase)]
    assert litellm_factories, "expected the LiteLLM-backed chat factories to be registered"

    for factory in litellm_factories:
        client_cls = rag.llm.ChatModel[factory]
        assert keyword_is_bindable(client_cls), f"{factory} would not even accept the keyword"
        assert not client_sends_default_headers(client_cls), f"{factory} drops the headers but was told to take them"


def test_the_chat_classes_built_on_the_openai_sdk_base_do_receive_the_headers():
    """Their shared ``Base.__init__`` pops the keyword into the OpenAI and AsyncOpenAI clients it builds."""
    # Inheriting from `Base` is necessary but not sufficient: a subclass that
    # overwrites `self.client` in its own `__init__` never uses the client `Base`
    # built for it, so the headers go nowhere. Those classes are judged incapable
    # and pinned in test_tenant_llm_headers_transport_replacement.py.
    transport_replacing = _transport_replacing_clients()
    sdk_factories = [
        factory
        for factory, cls in rag.llm.ChatModel.items()
        if issubclass(cls, ChatClientBase) and not any(klass in transport_replacing for klass in cls.__mro__)
    ]
    assert sdk_factories, "expected the OpenAI-SDK-backed chat factories to be registered"

    for factory in sdk_factories:
        assert client_sends_default_headers(rag.llm.ChatModel[factory]), factory


# Corpus-wide invariants over the live registries


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_no_factory_judged_capable_could_reject_the_keyword(model_type):
    """The whole point of the filter: a "sends" verdict must never be able to raise at construction."""
    for factory, client_cls in REGISTRIES[model_type].items():
        if client_sends_default_headers(client_cls):
            assert keyword_is_bindable(client_cls), f"{model_type}/{factory} is judged capable but rejects the keyword"


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_every_factory_that_cannot_bind_the_keyword_is_judged_incapable(model_type):
    for factory, client_cls in REGISTRIES[model_type].items():
        if not keyword_is_bindable(client_cls):
            assert not client_sends_default_headers(client_cls), f"{model_type}/{factory} cannot bind the keyword"


@pytest.mark.parametrize("model_type", ["chat", "embedding", "rerank"])
def test_without_configured_headers_no_factory_is_handed_the_keyword(model_type):
    """The no-headers path has to stay exactly what it was for every registered factory."""
    for factory, client_cls in REGISTRIES[model_type].items():
        assert default_headers_kwarg(client_cls=client_cls, default_headers=None, factory_name=factory) == {}
        assert default_headers_kwarg(client_cls=client_cls, default_headers={}, factory_name=factory) == {}


def test_the_openai_api_compatible_factories_are_the_capable_ones_for_embedding_and_rerank():
    """ONDEWO registers its vLLM endpoints under this factory, so these are the verdicts the deployment relies on."""
    assert client_sends_default_headers(rag.llm.EmbeddingModel["OpenAI-API-Compatible"]) is True
    assert client_sends_default_headers(rag.llm.RerankModel["OpenAI-API-Compatible"]) is True
    assert rag.llm.EmbeddingModel["OpenAI-API-Compatible"] is OpenAI_APIEmbed
    assert rag.llm.RerankModel["OpenAI-API-Compatible"] is OpenAI_APIRerank


# The two widened clients name the keyword, so a typo in it is no longer swallowed


def test_the_embedding_client_rejects_a_misspelled_header_keyword():
    with pytest.raises(TypeError):
        OpenAI_APIEmbed("key", "bge-m3", BASE_URL, defualt_headers=dict(GATEWAY_HEADERS))


def test_the_rerank_client_rejects_a_misspelled_header_keyword():
    with pytest.raises(TypeError):
        OpenAI_APIRerank("key", "qwen3-reranker", BASE_URL, defualt_headers=dict(GATEWAY_HEADERS))
