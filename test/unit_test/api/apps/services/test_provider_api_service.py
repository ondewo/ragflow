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
"""Tests for the provider instance and model endpoints in provider_api_service.py.

Four behaviours are pinned here:

* An update that skips the live probe (``verify=False``) must survive a factory model
  dictionary entry that carries no ``max_tokens``. 64 entries omit it, across every model
  type, and reading the key directly aborted the whole call with ``KeyError('max_tokens')``.
  Instance create walks the same branch whenever the request selects no model.
* A response never carries a usable API key. ``show_provider_instance`` reports a
  placeholder, which still distinguishes "a key is configured" from "none is", and an
  update that submits that placeholder back keeps the stored key.
* Deleting a model, an instance or a provider is refused while a chat assistant or a
  dataset still points at one of the models, and the refusal names the dependents.
  When nothing references them, the tenant's default-model settings are detached first,
  because a default pointing at a deleted model no longer resolves.
* ``default_headers`` submitted on instance create/update round-trips into the instance
  ``extra`` blob, which is where the model clients read it from.

The module's own imports are stubbed rather than installed: the real ones reach
PostgreSQL, OpenSearch and the model registry at import time.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from common.constants import ModelTypeBinary, StatusEnum

pytestmark = pytest.mark.p2


# ---------------------------------------------------------------------------
# A peewee stand-in: just enough of Model.select().where().limit()
# ---------------------------------------------------------------------------


class _Expression:
    """A row predicate that supports peewee's ``|`` composition."""

    def __init__(self, predicate):
        self.predicate = predicate

    def __or__(self, other):
        return _Expression(lambda row: self.predicate(row) or other.predicate(row))


class _Column:
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, value):
        return _Expression(lambda row, name=self.name, expected=value: getattr(row, name) == expected)

    def in_(self, values):
        candidates = list(values)
        return _Expression(lambda row, name=self.name, expected=candidates: getattr(row, name) in expected)

    def __hash__(self):
        return hash(self.name)


class _Query:
    def __init__(self, rows):
        self.rows = rows

    def where(self, *expressions):
        return _Query([row for row in self.rows if all(expression.predicate(row) for expression in expressions)])

    def limit(self, count):
        return self.rows[:count]


class _FakeTable:
    rows: ClassVar[list] = []
    id = _Column("id")
    name = _Column("name")
    tenant_id = _Column("tenant_id")
    status = _Column("status")

    @classmethod
    def select(cls, *_columns):
        return _Query(list(cls.rows))


class _FakeDialog(_FakeTable):
    llm_id = _Column("llm_id")
    tenant_llm_id = _Column("tenant_llm_id")
    rerank_id = _Column("rerank_id")
    tenant_rerank_id = _Column("tenant_rerank_id")


class _FakeKnowledgebase(_FakeTable):
    embd_id = _Column("embd_id")
    tenant_embd_id = _Column("tenant_embd_id")


def _dialog(dialog_id: str, name: str, *, llm_id: str = "", tenant_llm_id=None, rerank_id: str = "", tenant_rerank_id=None):
    return SimpleNamespace(
        id=dialog_id,
        name=name,
        tenant_id="tenant-1",
        status=StatusEnum.VALID.value,
        llm_id=llm_id,
        tenant_llm_id=tenant_llm_id,
        rerank_id=rerank_id,
        tenant_rerank_id=tenant_rerank_id,
    )


def _dataset(kb_id: str, name: str, *, embd_id: str = "", tenant_embd_id=None):
    return SimpleNamespace(
        id=kb_id,
        name=name,
        tenant_id="tenant-1",
        status=StatusEnum.VALID.value,
        embd_id=embd_id,
        tenant_embd_id=tenant_embd_id,
    )


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

# A VLLM-shaped provider with no static models, plus one whose dictionary entries are
# uneven: the TTS entry has no max_tokens, exactly like the real llm_factories.json.
_FACTORIES = [
    {"name": "VLLM", "url": "http://vllm.internal/v1", "llm": []},
    {
        "name": "Tongyi-Qianwen",
        "url": "http://dashscope.internal/v1",
        "llm": [
            {"llm_name": "qwen-chat", "model_type": "chat", "max_tokens": 4096},
            {"llm_name": "qwen3-tts-flash", "model_type": "tts"},
        ],
    },
]


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
    """The writes the service performed, for the tests to assert on."""

    def __init__(self):
        self.instance_updates: list[tuple[str, dict]] = []
        self.instances_created: list[dict] = []
        self.models_inserted: list[dict] = []
        self.models_updated: list[tuple[str, dict]] = []
        self.model_ids_deleted: list[str] = []
        self.instance_ids_deleted: list[str] = []
        self.tenant_updates: list[tuple[str, dict]] = []
        self.providers_deleted: list[tuple[str, str]] = []


def _load_service(monkeypatch, *, provider_name="VLLM", models=(), instance_extra="{}", instance_api_key="sk-live-abcdef", tenant=None):
    """Load provider_api_service against one provider holding one instance.

    Returns the module, the recorder of every write it made, and the fake rows so a
    test can read back the ids it should have acted on.
    """
    recorder = _Recorder()

    provider = SimpleNamespace(id="provider-1", provider_name=provider_name, tenant_id="tenant-1")
    instance = SimpleNamespace(
        id="instance-1",
        provider_id="provider-1",
        instance_name="primary",
        api_key=instance_api_key,
        extra=instance_extra,
        status="active",
        create_time=1,
    )
    model_rows = list(models)

    tenant_row = tenant
    if tenant_row is None:
        tenant_row = SimpleNamespace(
            id="tenant-1",
            llm_id="",
            tenant_llm_id=None,
            embd_id="",
            tenant_embd_id=None,
            asr_id="",
            tenant_asr_id=None,
            img2txt_id="",
            tenant_img2txt_id=None,
            rerank_id="",
            tenant_rerank_id=None,
            tts_id="",
            tenant_tts_id=None,
            ocr_id="",
            tenant_ocr_id=None,
        )

    def _insert_model(**kwargs):
        recorder.models_inserted.append(kwargs)
        return SimpleNamespace(id=f"model-{len(recorder.models_inserted)}")

    _stub(monkeypatch, "common.settings", FACTORY_LLM_INFOS=_FACTORIES)
    _stub(
        monkeypatch,
        "api.db.db_models",
        DB=SimpleNamespace(atomic=lambda: None),
        Dialog=_FakeDialog,
        Knowledgebase=_FakeKnowledgebase,
    )
    _stub(
        monkeypatch,
        "api.db.joint_services.tenant_model_service",
        resolve_model_config=lambda *args, **kwargs: {},
        delete_models_by_instance_ids=lambda instance_ids: recorder.model_ids_deleted.extend(model.id for model in model_rows),
        delete_instances_by_provider_ids=lambda provider_ids: None,
    )
    _stub(
        monkeypatch,
        "api.db.services.tenant_model_provider_service",
        TenantModelProviderService=SimpleNamespace(
            get_by_tenant_id_and_provider_id=lambda tenant_id, provider_id: provider if provider_id == provider.id else None,
            get_by_tenant_id_and_provider_name=lambda tenant_id, name: provider if name == provider.provider_name else None,
            get_by_id=lambda provider_id: (True, provider) if provider_id == provider.id else (False, None),
            delete_by_tenant_id_and_provider_name=lambda tenant_id, name: recorder.providers_deleted.append((tenant_id, name)),
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.tenant_model_instance_service",
        TenantModelInstanceService=SimpleNamespace(
            get_by_id=lambda instance_id: (True, instance) if instance_id == instance.id else (False, None),
            get_by_provider_id_and_instance_name=lambda provider_id, name: instance if name == instance.instance_name else None,
            get_all_by_provider_id=lambda provider_id: [instance],
            update_by_id=lambda instance_id, update: recorder.instance_updates.append((instance_id, update)),
            create_instance=lambda **kwargs: recorder.instances_created.append(kwargs) or instance,
            delete_by_ids=lambda instance_ids: recorder.instance_ids_deleted.extend(instance_ids),
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.tenant_model_service",
        TenantModelService=SimpleNamespace(
            get_models_by_instance_id=lambda instance_id: list(model_rows),
            get_by_provider_id_and_instance_id_and_model_name=lambda provider_id, instance_id, model_name: next((model for model in model_rows if model.model_name == model_name), None),
            insert=_insert_model,
            update_model=lambda model_id, update: recorder.models_updated.append((model_id, update)),
            delete_by_ids=lambda model_ids: recorder.model_ids_deleted.extend(model_ids),
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.user_service",
        TenantService=SimpleNamespace(
            get_or_none=lambda **kwargs: tenant_row if kwargs.get("id") == tenant_row.id else None,
            update_by_id=lambda tenant_id, update: recorder.tenant_updates.append((tenant_id, update)),
        ),
    )
    _stub(monkeypatch, "rag", __path__=[])
    _stub(monkeypatch, "rag.llm", ChatModel={}, CvModel={}, EmbeddingModel={}, ModelMeta={}, OcrModel={}, RerankModel={}, Seq2txtModel={}, TTSModel={})

    module_path = Path(__file__).resolve().parents[5] / "api" / "apps" / "services" / "provider_api_service.py"
    spec = importlib.util.spec_from_file_location("provider_api_service_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "provider_api_service_under_test", module)
    spec.loader.exec_module(module)
    return module, recorder, SimpleNamespace(provider=provider, instance=instance, tenant=tenant_row)


def _skip_verification(monkeypatch, module):
    """Create always probes the provider; these tests are not about the probe."""

    async def _verified(*_args, **_kwargs):
        return True, "success", {}

    monkeypatch.setattr(module, "verify_api_key", _verified)


def _model(model_id: str, model_name: str, model_type: int, extra: str = "{}"):
    return SimpleNamespace(id=model_id, model_name=model_name, model_type=model_type, extra=extra, status="active")


@pytest.fixture(autouse=True)
def _clear_dependent_tables():
    _FakeDialog.rows = []
    _FakeKnowledgebase.rows = []
    yield
    _FakeDialog.rows = []
    _FakeKnowledgebase.rows = []


# ---------------------------------------------------------------------------
# (a) updating without a live probe
# ---------------------------------------------------------------------------


async def test_update_instance_without_verify_tolerates_factory_model_without_max_tokens(monkeypatch):
    """A factory entry with no max_tokens must not abort the update with a KeyError."""
    module, recorder, _rows = _load_service(monkeypatch, provider_name="Tongyi-Qianwen")

    success, message = await module.update_provider_instance(
        "tenant-1",
        "Tongyi-Qianwen",
        "instance-1",
        "primary",
        "sk-new",
        "http://dashscope.internal/v1",
        "",
        None,
        False,
    )

    assert (success, message) == (True, "success")
    inserted = {entry["model_name"]: json.loads(entry["extra"]) for entry in recorder.models_inserted}
    assert inserted["qwen-chat"]["max_tokens"] == 4096
    assert inserted["qwen3-tts-flash"]["max_tokens"] == module._DEFAULT_MAX_TOKENS


async def test_update_instance_without_verify_tolerates_missing_max_tokens_on_existing_model(monkeypatch):
    """The same entry must not abort the update when the model is already stored either."""
    existing = _model("model-tts", "qwen3-tts-flash", ModelTypeBinary.TTS.value, extra=json.dumps({"max_tokens": 1}))
    module, recorder, _rows = _load_service(monkeypatch, provider_name="Tongyi-Qianwen", models=[existing])

    success, message = await module.update_provider_instance(
        "tenant-1",
        "Tongyi-Qianwen",
        "instance-1",
        "primary",
        "sk-new",
        "http://dashscope.internal/v1",
        "",
        None,
        False,
    )

    assert (success, message) == (True, "success")
    updated_extra = json.loads(dict(recorder.models_updated)["model-tts"]["extra"])
    assert updated_extra["max_tokens"] == module._DEFAULT_MAX_TOKENS


async def test_create_instance_tolerates_factory_model_without_max_tokens(monkeypatch):
    """Create takes the same factory-default branch and must tolerate the same entry.

    The create endpoint defaults ``model_info`` to an empty list, so a request that
    selects no model walks the whole factory dictionary for the provider.
    """
    module, recorder, _rows = _load_service(monkeypatch, provider_name="Tongyi-Qianwen")
    _skip_verification(monkeypatch, module)

    success, message = await module.create_provider_instance(
        "tenant-1",
        "Tongyi-Qianwen",
        "primary",
        "sk-new",
        "http://dashscope.internal/v1",
        "",
        [],
    )

    assert (success, message) == (True, "success")
    inserted = {entry["model_name"]: json.loads(entry["extra"]) for entry in recorder.models_inserted}
    assert inserted["qwen-chat"]["max_tokens"] == 4096
    assert inserted["qwen3-tts-flash"]["max_tokens"] == module._DEFAULT_MAX_TOKENS


# ---------------------------------------------------------------------------
# (b) the stored API key never reaches a response
# ---------------------------------------------------------------------------


def test_show_instance_masks_the_stored_api_key(monkeypatch):
    module, _recorder, _rows = _load_service(monkeypatch, instance_api_key="sk-live-abcdef")

    success, instance = module.show_provider_instance("tenant-1", "VLLM", "instance-1")

    assert success is True
    assert instance["api_key"] == module.API_KEY_MASK
    assert "sk-live-abcdef" not in json.dumps(instance)


def test_show_instance_reports_an_unset_api_key_as_empty(monkeypatch):
    """A masked value must still tell a caller whether a key is configured at all."""
    module, _recorder, _rows = _load_service(monkeypatch, instance_api_key="")

    _success, instance = module.show_provider_instance("tenant-1", "VLLM", "instance-1")

    assert instance["api_key"] == ""


def test_show_instance_never_returns_the_configured_headers(monkeypatch):
    """default_headers routinely carry an authorization header, so they stay server-side."""
    extra = json.dumps({"base_url": "http://vllm.internal/v1", "default_headers": {"Authorization": "Basic c2VjcmV0"}})
    module, _recorder, _rows = _load_service(monkeypatch, instance_extra=extra)

    _success, instance = module.show_provider_instance("tenant-1", "VLLM", "instance-1")

    assert "c2VjcmV0" not in json.dumps(instance)


async def test_update_instance_keeps_the_stored_api_key_when_the_mask_is_submitted(monkeypatch):
    """Echoing an instance back unchanged must not overwrite the key with the placeholder."""
    module, recorder, _rows = _load_service(monkeypatch, instance_api_key="sk-live-abcdef")

    success, _message = await module.update_provider_instance(
        "tenant-1",
        "VLLM",
        "instance-1",
        "primary",
        module.API_KEY_MASK,
        "http://vllm.internal/v1",
        "",
        [],
        False,
    )

    assert success is True
    assert dict(recorder.instance_updates)["instance-1"]["api_key"] == "sk-live-abcdef"


async def test_update_instance_stores_a_newly_submitted_api_key(monkeypatch):
    module, recorder, _rows = _load_service(monkeypatch, instance_api_key="sk-live-abcdef")

    await module.update_provider_instance(
        "tenant-1",
        "VLLM",
        "instance-1",
        "primary",
        "sk-rotated",
        "http://vllm.internal/v1",
        "",
        [],
        False,
    )

    assert dict(recorder.instance_updates)["instance-1"]["api_key"] == "sk-rotated"


# ---------------------------------------------------------------------------
# (c) nothing deletes a model something still points at
# ---------------------------------------------------------------------------


async def test_delete_model_refused_while_a_chat_assistant_names_it(monkeypatch):
    chat_model = _model("model-chat", "qwen-chat", ModelTypeBinary.CHAT.value)
    module, recorder, _rows = _load_service(monkeypatch, models=[chat_model])
    _FakeDialog.rows = [_dialog("dialog-1", "Support bot", llm_id="qwen-chat@primary@VLLM")]

    success, message = await module.delete_models_from_instance("tenant-1", "VLLM", "instance-1", ["qwen-chat"])

    assert success is False
    assert "chat assistant 'Support bot' (dialog-1)" in message
    assert "qwen-chat@primary@VLLM" in message
    assert recorder.model_ids_deleted == []


async def test_delete_model_refused_while_a_chat_assistant_holds_only_its_id(monkeypatch):
    """The tenant_*_id columns are the authoritative reference; the name may be blank."""
    chat_model = _model("model-chat", "qwen-chat", ModelTypeBinary.CHAT.value)
    module, recorder, _rows = _load_service(monkeypatch, models=[chat_model])
    _FakeDialog.rows = [_dialog("dialog-2", "Id only", llm_id="", tenant_llm_id="model-chat")]

    success, message = await module.delete_models_from_instance("tenant-1", "VLLM", "instance-1", ["qwen-chat"])

    assert success is False
    assert "chat assistant 'Id only' (dialog-2)" in message
    assert recorder.model_ids_deleted == []


async def test_delete_model_refused_while_a_chat_assistant_reranks_with_it(monkeypatch):
    rerank_model = _model("model-rerank", "bge-reranker", ModelTypeBinary.RERANK.value)
    module, recorder, _rows = _load_service(monkeypatch, models=[rerank_model])
    _FakeDialog.rows = [_dialog("dialog-3", "Reranked", rerank_id="bge-reranker@primary@VLLM")]

    success, message = await module.delete_models_from_instance("tenant-1", "VLLM", "instance-1", ["bge-reranker"])

    assert success is False
    assert "chat assistant 'Reranked' (dialog-3)" in message
    assert recorder.model_ids_deleted == []


async def test_delete_model_names_every_dependent_once(monkeypatch):
    """A model used as both chat and rerank model by one assistant is reported once."""
    dual_model = _model("model-dual", "qwen-chat", ModelTypeBinary.CHAT.value | ModelTypeBinary.RERANK.value)
    module, _recorder, _rows = _load_service(monkeypatch, models=[dual_model])
    _FakeDialog.rows = [_dialog("dialog-4", "Both", llm_id="qwen-chat@primary@VLLM", rerank_id="qwen-chat@primary@VLLM")]

    _success, message = await module.delete_models_from_instance("tenant-1", "VLLM", "instance-1", ["qwen-chat"])

    assert message.count("dialog-4") == 1


async def test_delete_model_ignores_an_assistant_of_another_tenant(monkeypatch):
    chat_model = _model("model-chat", "qwen-chat", ModelTypeBinary.CHAT.value)
    module, recorder, _rows = _load_service(monkeypatch, models=[chat_model])
    foreign = _dialog("dialog-9", "Someone else", llm_id="qwen-chat@primary@VLLM")
    foreign.tenant_id = "tenant-2"
    _FakeDialog.rows = [foreign]

    success, _message = await module.delete_models_from_instance("tenant-1", "VLLM", "instance-1", ["qwen-chat"])

    assert success is True
    assert recorder.model_ids_deleted == ["model-chat"]


async def test_delete_model_detaches_it_from_the_tenant_defaults(monkeypatch):
    """An unreferenced model is deleted, but only after the tenant stops defaulting to it."""
    chat_model = _model("model-chat", "qwen-chat", ModelTypeBinary.CHAT.value)
    tenant = SimpleNamespace(
        id="tenant-1",
        llm_id="qwen-chat@primary@VLLM",
        tenant_llm_id="model-chat",
        embd_id="",
        tenant_embd_id=None,
        asr_id="",
        tenant_asr_id=None,
        img2txt_id="",
        tenant_img2txt_id=None,
        rerank_id="",
        tenant_rerank_id=None,
        tts_id="",
        tenant_tts_id=None,
        ocr_id="",
        tenant_ocr_id=None,
    )
    module, recorder, _rows = _load_service(monkeypatch, models=[chat_model], tenant=tenant)

    success, _message = await module.delete_models_from_instance("tenant-1", "VLLM", "instance-1", ["qwen-chat"])

    assert success is True
    assert recorder.tenant_updates == [("tenant-1", {"llm_id": "", "tenant_llm_id": None})]
    assert recorder.model_ids_deleted == ["model-chat"]


async def test_delete_model_leaves_an_unrelated_tenant_default_alone(monkeypatch):
    chat_model = _model("model-chat", "qwen-chat", ModelTypeBinary.CHAT.value)
    tenant = SimpleNamespace(
        id="tenant-1",
        llm_id="other-model@primary@VLLM",
        tenant_llm_id="model-other",
        embd_id="",
        tenant_embd_id=None,
        asr_id="",
        tenant_asr_id=None,
        img2txt_id="",
        tenant_img2txt_id=None,
        rerank_id="",
        tenant_rerank_id=None,
        tts_id="",
        tenant_tts_id=None,
        ocr_id="",
        tenant_ocr_id=None,
    )
    module, recorder, _rows = _load_service(monkeypatch, models=[chat_model], tenant=tenant)

    await module.delete_models_from_instance("tenant-1", "VLLM", "instance-1", ["qwen-chat"])

    assert recorder.tenant_updates == []


def test_drop_instance_refused_while_a_dataset_uses_its_embedding_model(monkeypatch):
    embedding_model = _model("model-embd", "bge-m3", ModelTypeBinary.EMBEDDING.value)
    module, recorder, _rows = _load_service(monkeypatch, models=[embedding_model])
    _FakeKnowledgebase.rows = [_dataset("kb-1", "Handbook", embd_id="bge-m3@primary@VLLM")]

    success, message = module.drop_provider_instances("tenant-1", "VLLM", ["instance-1"])

    assert success is False
    assert "dataset 'Handbook' (kb-1)" in message
    assert recorder.instance_ids_deleted == []


def test_drop_instance_deletes_an_unreferenced_instance(monkeypatch):
    embedding_model = _model("model-embd", "bge-m3", ModelTypeBinary.EMBEDDING.value)
    module, recorder, _rows = _load_service(monkeypatch, models=[embedding_model])

    success, _message = module.drop_provider_instances("tenant-1", "VLLM", ["instance-1"])

    assert success is True
    assert recorder.instance_ids_deleted == ["instance-1"]


def test_delete_provider_refused_while_a_dataset_uses_one_of_its_models(monkeypatch):
    embedding_model = _model("model-embd", "bge-m3", ModelTypeBinary.EMBEDDING.value)
    module, recorder, _rows = _load_service(monkeypatch, models=[embedding_model])
    _FakeKnowledgebase.rows = [_dataset("kb-2", "Policies", tenant_embd_id="model-embd")]

    success, message = module.delete_provider("tenant-1", "VLLM")

    assert success is False
    assert "dataset 'Policies' (kb-2)" in message
    assert recorder.providers_deleted == []


def test_delete_provider_deletes_an_unreferenced_provider(monkeypatch):
    embedding_model = _model("model-embd", "bge-m3", ModelTypeBinary.EMBEDDING.value)
    module, recorder, _rows = _load_service(monkeypatch, models=[embedding_model])

    success, _message = module.delete_provider("tenant-1", "VLLM")

    assert success is True
    assert recorder.providers_deleted == [("tenant-1", "VLLM")]


# ---------------------------------------------------------------------------
# (d) custom HTTP headers reach the instance extra blob
# ---------------------------------------------------------------------------


async def test_create_instance_persists_default_headers(monkeypatch):
    module, recorder, _rows = _load_service(monkeypatch)
    _skip_verification(monkeypatch, module)

    success, _message = await module.create_provider_instance(
        "tenant-1",
        "VLLM",
        "primary-2",
        "sk-new",
        "http://vllm.internal/v1",
        "",
        [],
        default_headers={"X-Tenant": "oeamtc"},
    )

    assert success is True
    assert json.loads(recorder.instances_created[0]["extra"])["default_headers"] == {"X-Tenant": "oeamtc"}


async def test_update_instance_persists_default_headers(monkeypatch):
    module, recorder, _rows = _load_service(monkeypatch)

    success, _message = await module.update_provider_instance(
        "tenant-1",
        "VLLM",
        "instance-1",
        "primary",
        "sk-new",
        "http://vllm.internal/v1",
        "",
        [],
        False,
        default_headers={"X-Tenant": "oeamtc"},
    )

    assert success is True
    stored = json.loads(dict(recorder.instance_updates)["instance-1"]["extra"])
    assert stored["default_headers"] == {"X-Tenant": "oeamtc"}
    assert stored["base_url"] == "http://vllm.internal/v1"


async def test_update_instance_keeps_stored_headers_when_the_field_is_omitted(monkeypatch):
    extra = json.dumps({"base_url": "http://vllm.internal/v1", "default_headers": {"X-Tenant": "oeamtc"}})
    module, recorder, _rows = _load_service(monkeypatch, instance_extra=extra)

    await module.update_provider_instance(
        "tenant-1",
        "VLLM",
        "instance-1",
        "primary",
        "sk-new",
        "http://vllm.internal/v1",
        "",
        [],
        False,
    )

    stored = json.loads(dict(recorder.instance_updates)["instance-1"]["extra"])
    assert stored["default_headers"] == {"X-Tenant": "oeamtc"}


async def test_update_instance_removes_stored_headers_on_an_empty_object(monkeypatch):
    extra = json.dumps({"base_url": "http://vllm.internal/v1", "default_headers": {"X-Tenant": "oeamtc"}})
    module, recorder, _rows = _load_service(monkeypatch, instance_extra=extra)

    await module.update_provider_instance(
        "tenant-1",
        "VLLM",
        "instance-1",
        "primary",
        "sk-new",
        "http://vllm.internal/v1",
        "",
        [],
        False,
        default_headers={},
    )

    stored = json.loads(dict(recorder.instance_updates)["instance-1"]["extra"])
    assert "default_headers" not in stored


@pytest.mark.parametrize(
    "default_headers",
    [
        "Authorization: Basic x",
        {"Authorization": 1},
        {"": "value"},
    ],
)
async def test_update_instance_rejects_malformed_default_headers(monkeypatch, default_headers):
    module, recorder, _rows = _load_service(monkeypatch)

    success, message = await module.update_provider_instance(
        "tenant-1",
        "VLLM",
        "instance-1",
        "primary",
        "sk-new",
        "http://vllm.internal/v1",
        "",
        [],
        False,
        default_headers=default_headers,
    )

    assert success is False
    assert "default_headers" in message
    assert recorder.instance_updates == []


async def test_create_instance_rejects_malformed_default_headers(monkeypatch):
    module, recorder, _rows = _load_service(monkeypatch)

    success, message = await module.create_provider_instance(
        "tenant-1",
        "VLLM",
        "primary-2",
        "sk-new",
        "http://vllm.internal/v1",
        "",
        [],
        default_headers=["Authorization"],
    )

    assert success is False
    assert "default_headers" in message
    assert recorder.instances_created == []
