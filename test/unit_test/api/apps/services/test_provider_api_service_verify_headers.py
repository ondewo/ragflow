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
"""The verification probe in provider_api_service.py uses the instance's custom headers.

ONDEWO CAI registers vLLM endpoints that sit behind a gateway requiring an extra HTTP
header, under the `OpenAI-API-Compatible` factory. Verification runs *before* the
instance is written, so a probe built without those headers answers a correct
configuration with the gateway's 401 and the instance is never saved.

Pinned here:

* `verify_api_key` hands the configured headers to the embedding, chat and rerank probe
  clients -- the same three types `TenantLLMService.model_instance` forwards them to.
* The probe and the live request agree on whether a given client is sent the headers,
  because both ask `default_headers_kwarg`: a client with a fixed argument list and one
  that only absorbs `**kwargs` are probed without the headers rather than failing
  verification, and neither is verified with a header the live request would drop.
* The probe sees the headers that will be in effect *after* the write: the submitted set
  on create, the submitted set on update, the stored set when an update omits the field,
  and none at all when an update submits `{}` to clear them.
* The headers go through the single contract validator, so a stored value carrying CRLF
  is refused instead of being injected into the probe request.

`validate_default_headers` is imported for real, before any stubbing, so the module under
test binds the contract validator rather than a copy of its rules. The module's other
imports are stubbed: the real ones reach PostgreSQL, OpenSearch and the model registry.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

# Imported for the side effect of putting the real module in sys.modules: the module under
# test must bind the one contract validator and the one send-the-headers verdict, not a
# stub of either. `default_headers_kwarg` is also called directly, to derive what the probe
# is expected to carry from the rule the live request is built with.
from api.db.services.tenant_llm_service import (  # noqa: F401
    _header_aware_kwargs_constructors,
    default_headers_kwarg,
    validate_default_headers,
)

pytestmark = pytest.mark.p2

PROVIDER = "OpenAI-API-Compatible"
BASE_URL = "http://vllm.internal/v1"
GATEWAY_HEADERS = {"X-Gateway-Token": "gw-secret", "X-Tenant": "oeamtc"}

_MISSING = object()


# ---------------------------------------------------------------------------
# Probe clients standing in for the rag.llm registries
# ---------------------------------------------------------------------------


class _ProbeClient:
    """Records how the service constructed it, then answers the probe successfully.

    Names `default_headers`, as `OpenAI_APIEmbed` and `OpenAI_APIRerank` do -- the
    classes the `OpenAI-API-Compatible` factory resolves to. The sentinel default keeps
    "the keyword was not passed" distinguishable from "it was passed as None".
    """

    built: ClassVar[list["_ProbeClient"]] = []

    def __init__(self, key, model_name, base_url=None, default_headers=_MISSING, **kwargs):
        self.key = key
        self.model_name = model_name
        self.base_url = base_url
        self.header_keyword_passed = default_headers is not _MISSING
        self.default_headers = None if default_headers is _MISSING else default_headers
        self.extra_kwargs = kwargs
        _ProbeClient.built.append(self)


class _ProbeEmbed(_ProbeClient):
    def encode(self, texts):
        return [[0.1, 0.2]], 4


class _ProbeChat(_ProbeClient):
    async def async_chat_streamly(self, system, history, gen_conf):
        yield "hi"


class _ProbeRerank(_ProbeClient):
    def similarity(self, query, texts):
        return [0.9], 4


class _ChatClientBase:
    """Stands in for `rag.llm.chat_model.Base`, which pops the headers out of `**kwargs`."""

    def __init__(self, key, model_name, base_url=None, **kwargs):
        self.default_headers = kwargs.pop("default_headers", None)


class _HeaderlessEmbed:
    """A client with a fixed argument list, like most of the real provider classes."""

    built: ClassVar[list["_HeaderlessEmbed"]] = []

    def __init__(self, key, model_name, base_url=None):
        self.key = key
        _HeaderlessEmbed.built.append(self)

    def encode(self, texts):
        return [[0.1, 0.2]], 4


class _KwargsOnlyEmbed:
    """Absorbs `**kwargs` and never sends them, like `HuggingFaceEmbed` and a dozen others.

    `client_sends_default_headers` deliberately reads this as "does not send the headers",
    so the live request drops them -- and the probe has to drop them too, or verification
    would pass against a header the endpoint will never actually be sent.
    """

    attempts: ClassVar[list[dict]] = []

    def __init__(self, key, model_name, base_url=None, **kwargs):
        _KwargsOnlyEmbed.attempts.append(dict(kwargs))

    def encode(self, texts):
        return [[0.1, 0.2]], 4


@pytest.fixture(autouse=True)
def _clear_probe_records():
    _ProbeClient.built = []
    _HeaderlessEmbed.built = []
    _KwargsOnlyEmbed.attempts = []
    # The chat-client allowlist is cached for the process, and these tests stub
    # `rag.llm.chat_model`. Drop the cache on both sides so neither the real `Base` nor
    # the stub leaks across test boundaries.
    _header_aware_kwargs_constructors.cache_clear()
    yield
    _ProbeClient.built = []
    _HeaderlessEmbed.built = []
    _KwargsOnlyEmbed.attempts = []
    _header_aware_kwargs_constructors.cache_clear()


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_FACTORIES = [{"name": PROVIDER, "url": BASE_URL, "llm": []}]


def _stub(monkeypatch, name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, _, child_name = name.rpartition(".")
        parent_module = sys.modules.get(parent_name)
        if parent_module is not None:
            monkeypatch.setattr(parent_module, child_name, module, raising=False)
    return module


class _Recorder:
    """The writes the service performed and the probes verification was asked for."""

    def __init__(self):
        self.instances_created: list[dict] = []
        self.instance_updates: list[tuple[str, dict]] = []
        self.models_inserted: list[dict] = []
        self.verify_calls: list[dict] = []


def _load_service(monkeypatch, *, instance_extra="{}", embedding_cls=_ProbeEmbed):
    """Load provider_api_service against one provider holding one instance."""
    recorder = _Recorder()

    provider = SimpleNamespace(id="provider-1", provider_name=PROVIDER, tenant_id="tenant-1")
    instance = SimpleNamespace(
        id="instance-1",
        provider_id="provider-1",
        instance_name="primary",
        api_key="sk-stored",
        extra=instance_extra,
        status="active",
        create_time=1,
    )

    _stub(monkeypatch, "common.settings", FACTORY_LLM_INFOS=_FACTORIES)
    _stub(monkeypatch, "api.db.db_models", DB=SimpleNamespace(atomic=lambda: None))
    _stub(
        monkeypatch,
        "api.db.joint_services.tenant_model_service",
        resolve_model_config=lambda *args, **kwargs: {},
        delete_models_by_instance_ids=lambda instance_ids: None,
        delete_instances_by_provider_ids=lambda provider_ids: None,
    )
    _stub(
        monkeypatch,
        "api.db.services.tenant_model_provider_service",
        TenantModelProviderService=SimpleNamespace(
            get_by_tenant_id_and_provider_id=lambda tenant_id, provider_id: provider if provider_id == provider.id else None,
            get_by_tenant_id_and_provider_name=lambda tenant_id, name: provider if name == provider.provider_name else None,
            get_by_id=lambda provider_id: (False, None),
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.tenant_model_instance_service",
        TenantModelInstanceService=SimpleNamespace(
            get_by_id=lambda instance_id: (True, instance) if instance_id == instance.id else (False, None),
            get_by_provider_id_and_instance_name=lambda provider_id, name: instance if name == instance.instance_name else None,
            update_by_id=lambda instance_id, update: recorder.instance_updates.append((instance_id, update)),
            create_instance=lambda **kwargs: recorder.instances_created.append(kwargs) or instance,
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.tenant_model_service",
        TenantModelService=SimpleNamespace(
            get_models_by_instance_id=lambda instance_id: [],
            get_by_provider_id_and_instance_id_and_model_name=lambda provider_id, instance_id, model_name: None,
            insert=lambda **kwargs: recorder.models_inserted.append(kwargs) or SimpleNamespace(id="model-1"),
            delete_by_ids=lambda model_ids: None,
        ),
    )
    _stub(monkeypatch, "rag", __path__=[])
    # `client_sends_default_headers` consults `rag.llm.chat_model.Base`, the one class
    # allowed to take the headers out of `**kwargs`.
    _stub(monkeypatch, "rag.llm.chat_model", Base=_ChatClientBase)
    _stub(
        monkeypatch,
        "rag.llm",
        __path__=[],
        chat_model=sys.modules["rag.llm.chat_model"],
        ChatModel={PROVIDER: _ProbeChat},
        CvModel={},
        EmbeddingModel={PROVIDER: embedding_cls},
        ModelMeta={},
        OcrModel={},
        RerankModel={PROVIDER: _ProbeRerank},
        Seq2txtModel={},
        TTSModel={},
    )

    module_path = Path(__file__).resolve().parents[5] / "api" / "apps" / "services" / "provider_api_service.py"
    spec = importlib.util.spec_from_file_location("provider_api_service_verify_headers_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "provider_api_service_verify_headers_mod", module)
    spec.loader.exec_module(module)
    return module, recorder, SimpleNamespace(provider=provider, instance=instance)


def _record_verification(monkeypatch, module, recorder):
    """Replace the probe with a recorder of the header set it was asked to verify with."""

    async def _verified(*args, **kwargs):
        recorder.verify_calls.append(kwargs)
        return True, "success", {}

    monkeypatch.setattr(module, "verify_api_key", _verified)


def _model_info(model_type: str, model_name: str = "my-model") -> list[dict]:
    return [{"model_type": [model_type], "model_name": model_name, "max_tokens": 4096}]


# ---------------------------------------------------------------------------
# (a) the probe clients carry the configured headers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", ["embedding", "chat", "rerank"])
async def test_verify_builds_the_probe_client_with_the_configured_headers(monkeypatch, model_type):
    """A gateway-fronted endpoint has to be probed with the header it demands."""
    module, _recorder, _rows = _load_service(monkeypatch)

    success, message, _result = await module.verify_api_key(
        PROVIDER,
        "sk-live",
        BASE_URL,
        "",
        _model_info(model_type),
        default_headers=GATEWAY_HEADERS,
    )

    assert (success, message) == (True, "success")
    assert [client.default_headers for client in _ProbeClient.built] == [GATEWAY_HEADERS]


async def test_verify_keeps_the_provider_kwarg_the_chat_probe_already_got(monkeypatch):
    """Threading the headers must not displace the `provider` kwarg the chat probe takes."""
    module, _recorder, _rows = _load_service(monkeypatch)

    await module.verify_api_key(PROVIDER, "sk-live", BASE_URL, "", _model_info("chat"), default_headers=GATEWAY_HEADERS)

    built = _ProbeClient.built[0]
    assert built.default_headers == GATEWAY_HEADERS
    assert built.extra_kwargs == {"provider": PROVIDER}


async def test_verify_passes_no_header_keyword_when_none_are_configured(monkeypatch):
    """With no headers configured the client is constructed exactly as before."""
    module, _recorder, _rows = _load_service(monkeypatch)

    success, _message, _result = await module.verify_api_key(PROVIDER, "sk-live", BASE_URL, "", _model_info("embedding"))

    assert success is True
    assert _ProbeClient.built[0].header_keyword_passed is False


# ---------------------------------------------------------------------------
# (b) a factory that cannot take the keyword stays verifiable, and the probe never
#     disagrees with the live request about whether the headers are sent
# ---------------------------------------------------------------------------


async def test_verify_degrades_for_a_client_that_takes_no_header_keyword(monkeypatch):
    """Most provider clients have a fixed argument list; they are probed without headers."""
    module, _recorder, _rows = _load_service(monkeypatch, embedding_cls=_HeaderlessEmbed)

    success, message, _result = await module.verify_api_key(
        PROVIDER,
        "sk-live",
        BASE_URL,
        "",
        _model_info("embedding"),
        default_headers=GATEWAY_HEADERS,
    )

    assert (success, message) == (True, "success")
    assert len(_HeaderlessEmbed.built) == 1


async def test_verify_sends_no_header_keyword_to_a_client_that_only_absorbs_kwargs(monkeypatch):
    """A client that swallows unknown keywords is probed without them, as the live request is.

    Passing the keyword here would be invisible rather than harmless: the probe would look
    header-aware while every live request the model serves goes out without the header.
    """
    module, _recorder, _rows = _load_service(monkeypatch, embedding_cls=_KwargsOnlyEmbed)

    success, message, _result = await module.verify_api_key(
        PROVIDER,
        "sk-live",
        BASE_URL,
        "",
        _model_info("embedding"),
        default_headers=GATEWAY_HEADERS,
    )

    assert (success, message) == (True, "success")
    assert _KwargsOnlyEmbed.attempts == [{}]


@pytest.mark.parametrize("embedding_cls", [_ProbeEmbed, _HeaderlessEmbed, _KwargsOnlyEmbed])
async def test_probe_and_live_request_agree_on_sending_the_headers(monkeypatch, embedding_cls):
    """Both sides ask `default_headers_kwarg`, so the probe cannot be built differently.

    The expectation is derived from that helper rather than written down per class: it is
    the verdict `TenantLLMService.model_instance` builds the live client with, and a probe
    that answered it on its own would drift from it on the next client that is added.
    """
    probed: list[dict] = []

    class _Recording(embedding_cls):
        """Keeps the parent's constructor shape, and so the parent's verdict."""

        def __init__(self, *args, **kwargs):
            probed.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    module, _recorder, _rows = _load_service(monkeypatch, embedding_cls=_Recording)
    live_kwarg = default_headers_kwarg(client_cls=_Recording, default_headers=GATEWAY_HEADERS, factory_name=PROVIDER)

    success, message, _result = await module.verify_api_key(
        PROVIDER,
        "sk-live",
        BASE_URL,
        "",
        _model_info("embedding"),
        default_headers=GATEWAY_HEADERS,
    )

    assert (success, message) == (True, "success")
    assert {key: value for key, value in probed[0].items() if key == "default_headers"} == live_kwarg


# ---------------------------------------------------------------------------
# (c) the headers go through the one contract validator
# ---------------------------------------------------------------------------


async def test_verify_refuses_a_header_value_carrying_crlf(monkeypatch):
    """A stored header has not been through the validator in this process."""
    module, _recorder, _rows = _load_service(monkeypatch)

    success, message, _result = await module.verify_api_key(
        PROVIDER,
        "sk-live",
        BASE_URL,
        "",
        _model_info("embedding"),
        default_headers={"X-Gateway-Token": "gw\r\nX-Injected: 1"},
    )

    assert success is False
    assert "default_headers" in message
    assert _ProbeClient.built == []


async def test_update_refuses_a_stored_header_value_carrying_crlf(monkeypatch):
    """An instance whose stored headers are unusable is reported, not probed and not written."""
    extra = json.dumps({"base_url": BASE_URL, "default_headers": {"X-Gateway-Token": "gw\r\nX-Injected: 1"}})
    module, recorder, _rows = _load_service(monkeypatch, instance_extra=extra)

    success, message = await module.update_provider_instance("tenant-1", PROVIDER, "instance-1", "primary", "sk-new", BASE_URL, "", _model_info("embedding"))

    assert success is False
    assert "default_headers" in message
    assert recorder.instance_updates == []


# ---------------------------------------------------------------------------
# (d) both call sites verify with the headers that will be in effect
# ---------------------------------------------------------------------------


async def test_create_verifies_with_the_headers_it_is_about_to_store(monkeypatch):
    module, recorder, _rows = _load_service(monkeypatch)
    _record_verification(monkeypatch, module, recorder)

    success, message = await module.create_provider_instance(
        "tenant-1",
        PROVIDER,
        "primary",
        "sk-new",
        BASE_URL,
        "",
        _model_info("embedding"),
        default_headers=dict(GATEWAY_HEADERS),
    )

    assert (success, message) == (True, "success")
    stored = json.loads(recorder.instances_created[0]["extra"])
    assert [call["default_headers"] for call in recorder.verify_calls] == [stored["default_headers"]]


async def test_create_probes_the_endpoint_with_the_submitted_headers(monkeypatch):
    """End to end through the real probe: the client the gateway sees carries the header."""
    module, recorder, _rows = _load_service(monkeypatch)

    success, message = await module.create_provider_instance(
        "tenant-1",
        PROVIDER,
        "primary",
        "sk-new",
        BASE_URL,
        "",
        _model_info("embedding"),
        default_headers=dict(GATEWAY_HEADERS),
    )

    assert (success, message) == (True, "success")
    stored = json.loads(recorder.instances_created[0]["extra"])
    assert [client.default_headers for client in _ProbeClient.built] == [stored["default_headers"]]


async def test_update_verifies_with_the_stored_headers_when_the_field_is_omitted(monkeypatch):
    """An update that does not resubmit the headers keeps them, so the probe must send them."""
    stored_headers = dict(GATEWAY_HEADERS)
    extra = json.dumps({"base_url": BASE_URL, "default_headers": stored_headers})
    module, recorder, _rows = _load_service(monkeypatch, instance_extra=extra)
    _record_verification(monkeypatch, module, recorder)

    success, _message = await module.update_provider_instance("tenant-1", PROVIDER, "instance-1", "primary", "sk-new", BASE_URL, "", _model_info("embedding"))

    assert success is True
    written = json.loads(dict(recorder.instance_updates)["instance-1"]["extra"])
    assert [call["default_headers"] for call in recorder.verify_calls] == [written["default_headers"]]
    assert written["default_headers"] == stored_headers


async def test_update_verifies_with_the_submitted_headers_not_the_stored_ones(monkeypatch):
    """Replacing the headers must be verified against the replacement."""
    extra = json.dumps({"base_url": BASE_URL, "default_headers": {"X-Gateway-Token": "stale"}})
    module, recorder, _rows = _load_service(monkeypatch, instance_extra=extra)
    _record_verification(monkeypatch, module, recorder)

    success, _message = await module.update_provider_instance(
        "tenant-1",
        PROVIDER,
        "instance-1",
        "primary",
        "sk-new",
        BASE_URL,
        "",
        _model_info("embedding"),
        True,
        default_headers=dict(GATEWAY_HEADERS),
    )

    assert success is True
    written = json.loads(dict(recorder.instance_updates)["instance-1"]["extra"])
    assert [call["default_headers"] for call in recorder.verify_calls] == [written["default_headers"]]
    assert written["default_headers"] == GATEWAY_HEADERS


async def test_update_verifies_without_headers_when_they_are_being_cleared(monkeypatch):
    """An empty object removes the stored headers, so the probe must not send them either."""
    extra = json.dumps({"base_url": BASE_URL, "default_headers": dict(GATEWAY_HEADERS)})
    module, recorder, _rows = _load_service(monkeypatch, instance_extra=extra)
    _record_verification(monkeypatch, module, recorder)

    success, _message = await module.update_provider_instance(
        "tenant-1",
        PROVIDER,
        "instance-1",
        "primary",
        "sk-new",
        BASE_URL,
        "",
        _model_info("embedding"),
        True,
        default_headers={},
    )

    assert success is True
    written = json.loads(dict(recorder.instance_updates)["instance-1"]["extra"])
    assert "default_headers" not in written
    assert [call["default_headers"] for call in recorder.verify_calls] == [{}]
