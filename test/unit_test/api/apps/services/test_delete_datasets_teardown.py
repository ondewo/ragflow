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
"""Tests for the storage teardown in delete_datasets().

Deleting a dataset has to remove its stored objects, and it has to do so before
anything that points at them: once the documents and the dataset record are gone,
nothing names those objects any more.
"""

import importlib.util
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.p2

TENANT_ID = "tenant-1"
KB_ID = "kb-1"


class _StubModelTypeBinary(IntEnum):
    CHAT = 1
    EMBEDDING = 2
    ASR = 4
    VISION = 8
    RERANK = 16
    TTS = 32
    OCR = 64


def _stub(monkeypatch, name, **attrs):
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    if "." in name:
        parent_name, _, child_name = name.rpartition(".")
        parent_mod = sys.modules.get(parent_name)
        if parent_mod is not None:
            monkeypatch.setattr(parent_mod, child_name, mod, raising=False)
    return mod


def _load_module(
    monkeypatch,
    *,
    remove_documents_error=None,
):
    """Import dataset_api_service with every service it touches replaced by a fake.

    Returns the module and a recorder carrying the ordered call log and the fakes
    the assertions look at.
    """
    calls = []
    kb = SimpleNamespace(id=KB_ID, tenant_id=TENANT_ID, name="test-kb")

    def _remove_documents_of_kb(kb_id, tenant_id):
        calls.append("remove_documents")
        if remove_documents_error is not None:
            raise remove_documents_error
        return 1

    # Storage teardown is the service's job, not the handler's: the handler must
    # reach it only through DocumentService.remove_documents_of_kb. A bare fake
    # here would let a handler-level remove_bucket call pass unnoticed.
    storage_impl = SimpleNamespace(rm=MagicMock())

    recorder = SimpleNamespace(
        calls=calls,
        kb=kb,
        storage_impl=storage_impl,
        remove_documents_of_kb=MagicMock(side_effect=_remove_documents_of_kb),
        kb_delete_by_id=MagicMock(side_effect=lambda kb_id: calls.append("delete_dataset") is None),
    )

    _stub(
        monkeypatch,
        "api.db.services.document_service",
        DocumentService=SimpleNamespace(
            remove_documents_of_kb=recorder.remove_documents_of_kb,
            filter_delete=MagicMock(return_value=0),
        ),
        queue_raptor_o_graphrag_tasks=MagicMock(),
    )
    _stub(
        monkeypatch,
        "api.db.services.file_service",
        FileService=SimpleNamespace(filter_delete=MagicMock(return_value=0)),
    )
    _stub(
        monkeypatch,
        "api.db.services.knowledgebase_service",
        KnowledgebaseService=SimpleNamespace(
            get_or_none=lambda **_kwargs: kb,
            delete_by_id=recorder.kb_delete_by_id,
            query=lambda **_kwargs: [],
        ),
        validate_dataset_embedding_models=lambda kbs: None,
    )
    _stub(
        monkeypatch,
        "api.db.services.connector_service",
        Connector2KbService=SimpleNamespace(filter_delete=MagicMock()),
        SyncLogsService=SimpleNamespace(
            filter_delete=MagicMock(),
            filter_update=MagicMock(side_effect=lambda *_a, **_k: calls.append("cancel_running_syncs")),
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.task_service",
        TaskService=SimpleNamespace(),
        GRAPH_RAPTOR_FAKE_DOC_ID="fake-doc",
    )
    _stub(
        monkeypatch,
        "api.db.services.user_service",
        TenantService=SimpleNamespace(),
        UserService=SimpleNamespace(),
        UserTenantService=SimpleNamespace(),
    )
    _stub(
        monkeypatch,
        "api.db.services.tenant_model_service",
        TenantModelService=SimpleNamespace(),
    )
    _stub(
        monkeypatch,
        "api.db.joint_services.tenant_model_service",
        get_composite_model_name_by_ids=MagicMock(),
        resolve_model_config=MagicMock(),
        resolve_model_id=MagicMock(),
    )
    _stub(
        monkeypatch,
        "api.utils.api_utils",
        deep_merge=MagicMock(),
        get_parser_config=MagicMock(),
        remap_dictionary_keys=MagicMock(),
        verify_embedding_availability=MagicMock(),
    )
    _stub(
        monkeypatch,
        "common.settings",
        docStoreConn=SimpleNamespace(delete_idx=lambda *_args, **_kwargs: None),
        STORAGE_IMPL=storage_impl,
    )
    _stub(
        monkeypatch,
        "api.db.db_models",
        DB=SimpleNamespace(connection_context=lambda: lambda func: func),
        TenantModel=SimpleNamespace(),
        Connector2Kb=SimpleNamespace(kb_id="kb_id"),
        Document=SimpleNamespace(kb_id="kb_id"),
        File=SimpleNamespace(source_type="source_type", id="id", type="type", name="name"),
        SyncLogs=SimpleNamespace(kb_id="kb_id", status=SimpleNamespace(in_=lambda _values: None)),
    )
    _stub(
        monkeypatch,
        "common.constants",
        PAGERANK_FLD="pagerank",
        TAG_FLD="tag",
        FileSource=SimpleNamespace(KNOWLEDGEBASE="knowledgebase"),
        PipelineTaskType=SimpleNamespace(
            PARSE="parse",
            DOWNLOAD="download",
            RAPTOR="raptor",
            GRAPH_RAG="graph_rag",
            MINDMAP="mindmap",
            ARTIFACT="artifact",
            SKILL="skill",
        ),
        StatusEnum=SimpleNamespace(),
        LLMType=SimpleNamespace(),
        RetCode=SimpleNamespace(),
        TaskStatus=SimpleNamespace(SCHEDULE="schedule", RUNNING="running", CANCEL="cancel"),
        ModelTypeBinary=_StubModelTypeBinary,
    )
    _stub(monkeypatch, "rag.advanced_rag", __path__=[])
    _stub(monkeypatch, "rag.advanced_rag.knowlege_compile", __path__=[])
    _stub(
        monkeypatch,
        "rag.advanced_rag.knowlege_compile.wiki",
        WIKI_PAGE_COMPILE_KWD="wiki",
        _chunk_hash=lambda content: "stub-hash",
    )
    _stub(monkeypatch, "rag.nlp.search", index_name=lambda tenant_id: f"idx-{tenant_id}")

    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "api" / "apps" / "services" / "dataset_api_service.py"
    spec = importlib.util.spec_from_file_location("test_delete_datasets_teardown_module", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "test_delete_datasets_teardown_module", module)
    spec.loader.exec_module(module)
    return module, recorder


@pytest.mark.asyncio
async def test_delete_datasets_runs_its_phases_in_order(monkeypatch):
    """Queued syncs are cancelled, then the documents go, then the dataset record.

    Storage teardown is deliberately absent here: it belongs to
    DocumentService.remove_documents_of_kb, so that no other caller of that
    service can leak objects. Its ordering against the row deletes is asserted
    in test_document_service_remove_documents_of_kb.py, and the single-owner
    rule itself in test_dataset_storage_teardown_ownership.py.
    """
    module, recorder = _load_module(monkeypatch)

    ok, result = await module.delete_datasets(TENANT_ID, ids=[KB_ID])

    assert ok is True
    assert result == {"success_count": 1}
    assert recorder.calls == [
        "cancel_running_syncs",
        "remove_documents",
        "delete_dataset",
    ]
    assert not hasattr(recorder.storage_impl, "remove_bucket"), "the handler must not tear down storage itself"


@pytest.mark.asyncio
async def test_delete_datasets_removes_documents_in_one_bulk_call(monkeypatch):
    """One call per dataset, taking the dataset id and the tenant that owns it."""
    module, recorder = _load_module(monkeypatch)

    ok, _result = await module.delete_datasets(TENANT_ID, ids=[KB_ID])

    assert ok is True
    recorder.remove_documents_of_kb.assert_called_once_with(kb_id=KB_ID, tenant_id=TENANT_ID)


@pytest.mark.asyncio
async def test_delete_datasets_keeps_the_dataset_when_document_removal_fails(monkeypatch):
    """A dataset whose documents could not be removed must not lose its record either."""
    module, recorder = _load_module(monkeypatch, remove_documents_error=RuntimeError("doc store down"))

    ok, message = await module.delete_datasets(TENANT_ID, ids=[KB_ID])

    assert ok is False
    assert "doc store down" in message
    recorder.kb_delete_by_id.assert_not_called()
