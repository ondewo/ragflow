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
"""Tests for the doc-store teardown in delete_user_data().

A tenant owns two indexes: the content index holding its chunks and the document
metadata index. Deleting the user has to drop both, or the deployment accumulates
one orphaned index per deleted tenant, each carrying shards and field mappings.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from api.db import UserTenantRole
from common.constants import ActiveEnum

pytestmark = pytest.mark.p2

USER_ID = "user-1"
TENANT_ID = "tenant-1"
KB_ID = "kb-1"
DOC_ID = "doc-1"


def _content_index_name(tenant_id: str) -> str:
    """Mirror of rag.nlp.search.index_name, which cannot be imported without a live settings module."""
    return f"ragflow_{tenant_id}"


def _metadata_index_name(tenant_id: str) -> str:
    """Mirror of DocMetadataService._get_doc_meta_index_name."""
    return f"ragflow_doc_meta_{tenant_id}"


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


def _load_module(monkeypatch, *, delete_idx_errors=None):
    """Import user_account_service with every service it touches replaced by a fake.

    Args:
        delete_idx_errors: optional {index_name: exception} raised by delete_idx for that index.

    Returns the module and a recorder carrying the ordered doc-store call log and the fakes.
    """
    delete_idx_errors = delete_idx_errors or {}
    calls = []

    def _delete(condition, index_name, kb_ids):
        calls.append(("delete", index_name))
        return 3

    def _delete_idx(index_name, kb_id):
        calls.append(("delete_idx", index_name))
        error = delete_idx_errors.get(index_name)
        if error is not None:
            raise error

    recorder = SimpleNamespace(
        calls=calls,
        doc_store=SimpleNamespace(delete=MagicMock(side_effect=_delete), delete_idx=MagicMock(side_effect=_delete_idx)),
        user_delete_by_id=MagicMock(return_value=1),
        remove_bucket=MagicMock(),
    )

    usr = SimpleNamespace(id=USER_ID, is_active=ActiveEnum.INACTIVE.value, is_superuser=False)

    _stub(
        monkeypatch,
        "api.db.services.user_service",
        TenantService=SimpleNamespace(delete_by_id=MagicMock(return_value=1)),
        UserService=SimpleNamespace(filter_by_id=lambda _user_id: usr, delete_by_id=recorder.user_delete_by_id),
        UserTenantService=SimpleNamespace(
            get_user_tenant_relation_by_user_id=lambda _user_id: [{"id": "ut-1", "tenant_id": TENANT_ID, "role": UserTenantRole.OWNER.value}],
            delete_by_ids=MagicMock(return_value=1),
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.knowledgebase_service",
        KnowledgebaseService=SimpleNamespace(
            get_kb_ids=lambda _user_id: [KB_ID],
            delete_by_ids=MagicMock(return_value=1),
            decrease_document_num_in_delete=MagicMock(),
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.document_service",
        DocumentService=SimpleNamespace(
            get_all_doc_ids_by_kb_ids=lambda _kb_ids: [{"id": DOC_ID, "kb_id": KB_ID}],
            delete_by_ids=MagicMock(return_value=1),
            get_all_docs_by_creator_id=lambda _user_id: [],
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.doc_metadata_service",
        DocMetadataService=SimpleNamespace(
            delete_document_metadata=MagicMock(),
            _get_doc_meta_index_name=_metadata_index_name,
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.file2document_service",
        File2DocumentService=SimpleNamespace(
            delete_by_document_ids_or_file_ids=MagicMock(return_value=1),
            get_by_document_ids=lambda _doc_ids: [],
        ),
    )
    _stub(
        monkeypatch,
        "api.db.services.file_service",
        FileService=SimpleNamespace(
            get_all_file_ids_by_tenant_id=lambda _tenant_id: [{"id": "file-1"}],
            delete_by_ids=MagicMock(return_value=1),
            get_by_ids=lambda _file_ids: [],
            insert=MagicMock(),
            delete_by_id=MagicMock(),
        ),
    )
    _stub(monkeypatch, "api.db.services.task_service", TaskService=SimpleNamespace(delete_by_doc_ids=MagicMock(return_value=1)))
    _stub(monkeypatch, "api.db.services.langfuse_service", TenantLangfuseService=SimpleNamespace(delete_ty_tenant_id=MagicMock(return_value=0)))
    _stub(monkeypatch, "api.db.services.mcp_server_service", MCPServerService=SimpleNamespace(delete_by_tenant_id=MagicMock(return_value=0)))
    _stub(monkeypatch, "api.db.services.search_service", SearchService=SimpleNamespace(delete_by_tenant_id=MagicMock(return_value=0)))
    _stub(monkeypatch, "api.db.services.memory_service", MemoryService=SimpleNamespace(get_by_tenant_id=lambda _tenant_id: [], delete_by_ids=MagicMock(return_value=0)))
    _stub(monkeypatch, "api.db.services.canvas_service", UserCanvasService=SimpleNamespace(get_all_agents_by_tenant_ids=lambda _tenant_ids, _user_id: []))
    _stub(monkeypatch, "api.db.services.user_canvas_version", UserCanvasVersionService=SimpleNamespace(get_all_canvas_version_by_canvas_ids=lambda _ids: []))
    _stub(monkeypatch, "api.db.services.dialog_service", DialogService=SimpleNamespace(get_all_dialogs_by_tenant_id=lambda _tenant_id: []))
    _stub(monkeypatch, "api.db.services.conversation_service", ConversationService=SimpleNamespace(get_all_conversation_by_dialog_ids=lambda _ids: []))
    _stub(
        monkeypatch,
        "api.db.services.api_service",
        APITokenService=SimpleNamespace(delete_by_tenant_id=MagicMock(return_value=0)),
        API4ConversationService=SimpleNamespace(delete_by_dialog_ids=MagicMock(return_value=0)),
    )
    # The real api.utils.api_utils pulls in the peewee models, which read a live settings module.
    _stub(monkeypatch, "api.utils.api_utils", group_by=MagicMock(return_value={}))
    _stub(monkeypatch, "memory.services.messages", MessageService=SimpleNamespace(has_index=lambda _uid, _memory_id: False, delete_index=MagicMock()))
    _stub(monkeypatch, "rag.nlp.search", index_name=_content_index_name)
    _stub(
        monkeypatch,
        "common.settings",
        docStoreConn=recorder.doc_store,
        STORAGE_IMPL=SimpleNamespace(bucket_exists=lambda _bucket: True, remove_bucket=recorder.remove_bucket),
    )

    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "api" / "db" / "joint_services" / "user_account_service.py"
    spec = importlib.util.spec_from_file_location("test_delete_user_data_module", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "test_delete_user_data_module", module)
    spec.loader.exec_module(module)
    return module, recorder


def test_delete_user_data_drops_the_content_and_metadata_indexes(monkeypatch):
    """Both of the tenant's indexes go, and only after its chunks were deleted from the content one."""
    module, recorder = _load_module(monkeypatch)

    result = module.delete_user_data(USER_ID)

    assert result["success"] is True
    assert recorder.calls == [
        ("delete", _content_index_name(TENANT_ID)),
        ("delete_idx", _content_index_name(TENANT_ID)),
        ("delete_idx", _metadata_index_name(TENANT_ID)),
    ]
    assert f"Deleted content index {_content_index_name(TENANT_ID)}." in result["message"]


def test_delete_user_data_drops_the_whole_content_index_not_one_dataset(monkeypatch):
    """An empty dataset id is what makes a doc-store backend drop the index itself.

    OSConnection.delete_idx returns without doing anything for a non-empty dataset
    id, because every dataset of a tenant shares the one index.
    """
    module, recorder = _load_module(monkeypatch)

    module.delete_user_data(USER_ID)

    recorder.doc_store.delete_idx.assert_any_call(_content_index_name(TENANT_ID), "")


def test_delete_user_data_continues_when_the_content_index_cannot_be_dropped(monkeypatch):
    """A doc-store that refuses the drop must not strand the rest of the deletion."""
    module, recorder = _load_module(monkeypatch, delete_idx_errors={_content_index_name(TENANT_ID): RuntimeError("opensearch down")})

    result = module.delete_user_data(USER_ID)

    assert result["success"] is True
    assert "Failed to delete content index (continuing)." in result["message"]
    recorder.doc_store.delete_idx.assert_any_call(_metadata_index_name(TENANT_ID), "")
    recorder.user_delete_by_id.assert_called_once_with(USER_ID)
