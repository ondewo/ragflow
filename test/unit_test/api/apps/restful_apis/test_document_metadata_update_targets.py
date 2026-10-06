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
"""Which documents a batch metadata update targets, and what an update reports.

Both batch handlers -- `POST /datasets/<id>/metadata/update` and
`PATCH /datasets/<id>/documents/metadatas` -- take a selector of
`document_ids` and/or `metadata_condition`. An absent `document_ids` means
"every document of the dataset", so a `metadata_condition` alone narrows that
set, and an empty selector targets all of them. The cases below pin all four
selector shapes against both handlers, plus the rejection of an id that
belongs to another dataset.

`update_document` reports metadata from the doc-meta index, which is the only
place it is stored; the document row carries no `meta_fields` column, so the
response would otherwise omit the key a caller reads back after writing it.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from common.constants import RetCode

DATASET_ID = "kb-1"
DOC_A = "doc-a"
DOC_B = "doc-b"
DOC_C = "doc-c"
KB_DOC_IDS = [DOC_A, DOC_B, DOC_C]
# {field_name: {value: [doc_ids]}}, the shape get_flatted_meta_by_kbs returns.
METAS = {"lang": {"de": [DOC_A, DOC_B], "en": [DOC_C]}}
LANG_IS_DE = {"logic": "and", "conditions": [{"name": "lang", "comparison_operator": "=", "value": "de"}]}
LANG_IS_RU = {"logic": "and", "conditions": [{"name": "lang", "comparison_operator": "=", "value": "ru"}]}
UPDATES = [{"key": "reviewed", "value": "yes"}]

BATCH_HANDLERS = ("metadata_batch_update", "update_metadata")


class _PassthroughManager:
    def route(self, *_args, **_kwargs):
        return lambda func: func


class _LenientModule(ModuleType):
    """A stub module that yields a harmless placeholder for any attribute that
    wasn't explicitly provided. document_api.py's top-level
    `from <mod> import a, b` only needs every imported name to exist; symbols
    that are not on the handler paths below are never called, so a no-op
    placeholder is safe. This keeps the test from rotting each time
    document_api.py grows an import.
    """

    def __getattr__(self, _name):
        return lambda *_a, **_k: None


class _FakeUpdateDocumentReq:
    """The validated-request stand-in for `update_document`.

    Only the attributes the handler reads are needed; the real pydantic model
    pulls in the whole validation module, and the behaviour under test is the
    response assembly rather than request validation.
    """

    def __init__(self, **req):
        self.meta_fields = req.get("meta_fields", {})
        self.parser_config = req.get("parser_config")
        self.chunk_method = req.get("chunk_method")
        self.pipeline_id = req.get("pipeline_id")


def _stub(monkeypatch, name, **attrs):
    mod = _LenientModule(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _load_document_api(monkeypatch, *, request_json, kb_doc_ids=KB_DOC_IDS, metas=METAS, stored_metadata=None, doc_row=None):
    """Load document_api.py with the minimum stubs to exercise the metadata handlers.

    `common.constants` and `common.metadata_utils` are the real modules, so
    RetCode and the `metadata_condition` evaluation are the ones the routes
    actually use.

    Returns the module plus a `calls` dict recording what the handler asked the
    services to do.
    """
    calls = {"batch_update": [], "metadata_written": []}

    def _batch_update_metadata(kb_id, doc_ids, updates=None, deletes=None):
        calls["batch_update"].append((kb_id, sorted(doc_ids), updates, deletes))
        return len(doc_ids)

    def _update_document_metadata(doc_id, meta_fields):
        calls["metadata_written"].append((doc_id, meta_fields))
        return True

    doc = doc_row if doc_row is not None else SimpleNamespace(id=DOC_A, name="a.md", type="text", kb_id=DATASET_ID)

    _stub(
        monkeypatch,
        "api.apps",
        AUTH_JWT="jwt",
        AUTH_API="api",
        AUTH_BETA="beta",
        current_user=SimpleNamespace(id="tenant-1"),
        login_required=lambda func=None, **_kwargs: (lambda f: f) if func is None else func,
    )
    _stub(
        monkeypatch,
        "api.apps.services.document_api_service",
        validate_document_update_fields=lambda *_a, **_k: (None, None),
        map_doc_keys=lambda d: {"id": d.id, "name": d.name, "dataset_id": d.kb_id},
    )
    _stub(
        monkeypatch,
        "api.db.db_models",
        API4Conversation=SimpleNamespace(),
        # Applied as `@DB.connection_context()` at module level, so it has to
        # hand back an identity decorator.
        DB=SimpleNamespace(connection_context=lambda *_a, **_k: lambda func: func),
        Task=SimpleNamespace(),
    )
    _stub(monkeypatch, "api.db.services", duplicate_name=lambda *_a, **_k: "")
    _stub(
        monkeypatch,
        "api.db.services.doc_metadata_service",
        DocMetadataService=SimpleNamespace(
            get_flatted_meta_by_kbs=lambda _kb_ids: metas,
            batch_update_metadata=_batch_update_metadata,
            update_document_metadata=_update_document_metadata,
            get_document_metadata=lambda _doc_id: stored_metadata if stored_metadata is not None else {},
        ),
    )
    _stub(monkeypatch, "api.db.services.document_counter_service", release_reparse_counters=lambda *_a, **_k: None)
    _stub(
        monkeypatch,
        "api.db.services.document_service",
        DocumentService=SimpleNamespace(
            query=lambda **_k: [doc],
            get_by_id=lambda _doc_id: (True, doc),
            get_by_kb_id=lambda *_a, **_k: ([], 0),
        ),
    )
    _stub(monkeypatch, "api.db.services.file2document_service", File2DocumentService=SimpleNamespace())
    _stub(monkeypatch, "api.db.services.file_service", FileService=SimpleNamespace())
    _stub(
        monkeypatch,
        "api.db.services.knowledgebase_service",
        KnowledgebaseService=SimpleNamespace(
            accessible=lambda *_a, **_k: True,
            query=lambda **_k: [SimpleNamespace(id=DATASET_ID)],
            get_by_id=lambda _kb_id: (True, SimpleNamespace(id=DATASET_ID)),
            list_documents_by_ids=lambda _kb_ids: list(kb_doc_ids),
        ),
    )
    _stub(monkeypatch, "api.db.services.canvas_service", UserCanvasService=SimpleNamespace())
    _stub(monkeypatch, "api.common.check_team_permission", check_kb_team_permission=lambda *_a, **_k: True)
    _stub(monkeypatch, "api.db.services.task_service", TaskService=SimpleNamespace(), cancel_all_task_of=lambda *_a, **_k: None)

    async def _get_request_json():
        return request_json

    _stub(
        monkeypatch,
        "api.utils.api_utils",
        get_data_error_result=lambda message="", code=RetCode.DATA_ERROR, data=False: {"code": code, "message": message},
        get_error_data_result=lambda message="", **_k: {"code": RetCode.DATA_ERROR, "message": message},
        get_result=lambda *_a, **kwargs: {"code": RetCode.SUCCESS, "data": kwargs.get("data")},
        get_json_result=lambda *_a, **kwargs: {"code": RetCode.SUCCESS, "data": kwargs.get("data")},
        server_error_response=lambda e: {"code": RetCode.EXCEPTION_ERROR, "message": repr(e)},
        add_tenant_id_to_kwargs=lambda func: func,
        get_request_json=_get_request_json,
        # Used as `@validate_request(...)` at module level, so it must return an
        # identity decorator (the lenient fallback returns None and `@None`
        # raises TypeError during import).
        validate_request=lambda *_a, **_k: lambda func: func,
    )
    _stub(
        monkeypatch,
        "api.utils.validation_utils",
        UpdateDocumentReq=_FakeUpdateDocumentReq,
        format_validation_error_message=lambda e: str(e),
        validate_and_parse_json_request=lambda *_a, **_k: ({}, None),
    )
    _stub(monkeypatch, "common.settings", retriever=SimpleNamespace(), docStoreConn=SimpleNamespace(), STORAGE_IMPL=SimpleNamespace())
    _stub(monkeypatch, "common.misc_utils", get_uuid=lambda: "uuid")
    _stub(monkeypatch, "common.ssrf_guard", assert_url_is_safe=lambda *_a, **_k: None)
    _stub(monkeypatch, "api.utils.file_utils", filename_type=lambda *_a, **_k: None, thumbnail=lambda *_a, **_k: None)
    _stub(monkeypatch, "api.utils.file_response", apply_preview_file_response_headers=lambda *_a, **_k: None)
    _stub(monkeypatch, "api.utils.web_utils", CONTENT_TYPE_MAP={}, is_valid_url=lambda *_a, **_k: True)

    monkeypatch.setitem(sys.modules, "quart", _LenientModule("quart"))

    # parents[5] = repo root from test/unit_test/api/apps/restful_apis/<file>
    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "api" / "apps" / "restful_apis" / "document_api.py"
    spec = importlib.util.spec_from_file_location("test_document_metadata_api_module", module_path)
    module = importlib.util.module_from_spec(spec)
    # `manager` must exist before exec so the @manager.route decorators run.
    module.manager = _PassthroughManager()
    monkeypatch.setitem(sys.modules, "test_document_metadata_api_module", module)
    spec.loader.exec_module(module)

    # Pin the module globals the handlers resolve at call time. The sys.modules
    # stubs only guarantee the import succeeds: in a full environment, where a
    # real module is already loaded, `from x import y` binds the real name and
    # the stub is bypassed. Rebinding here makes the run identical in the
    # bare-stub and full-dependency cases.
    module.KnowledgebaseService = sys.modules["api.db.services.knowledgebase_service"].KnowledgebaseService
    module.DocumentService = sys.modules["api.db.services.document_service"].DocumentService
    module.DocMetadataService = sys.modules["api.db.services.doc_metadata_service"].DocMetadataService
    module.UpdateDocumentReq = _FakeUpdateDocumentReq
    module.validate_document_update_fields = lambda *_a, **_k: (None, None)
    module.map_doc_keys = sys.modules["api.apps.services.document_api_service"].map_doc_keys
    module.get_request_json = _get_request_json
    api_utils = sys.modules["api.utils.api_utils"]
    module.get_error_data_result = api_utils.get_error_data_result
    module.get_result = api_utils.get_result
    module.server_error_response = api_utils.server_error_response
    return module, calls


async def _run_batch_handler(module, handler_name):
    """Call one of the two batch handlers with the arguments its route supplies."""
    return await getattr(module, handler_name)(dataset_id=DATASET_ID, tenant_id="tenant-1")


@pytest.mark.p1
@pytest.mark.parametrize("handler_name", BATCH_HANDLERS)
class TestBatchMetadataUpdateTargets:
    """Both batch handlers select the same documents for the same selector."""

    async def test_empty_selector_targets_every_document(self, monkeypatch: pytest.MonkeyPatch, handler_name: str) -> None:
        """No selector means the whole dataset, not nothing at all."""
        module, calls = _load_document_api(monkeypatch, request_json={"updates": UPDATES})

        response = await _run_batch_handler(module, handler_name)

        assert response == {"code": RetCode.SUCCESS, "data": {"updated": len(KB_DOC_IDS), "matched_docs": len(KB_DOC_IDS)}}
        assert calls["batch_update"] == [(DATASET_ID, sorted(KB_DOC_IDS), UPDATES, [])]

    async def test_metadata_condition_alone_targets_the_matching_documents(self, monkeypatch: pytest.MonkeyPatch, handler_name: str) -> None:
        """A condition without document_ids narrows the dataset, not an empty set."""
        module, calls = _load_document_api(monkeypatch, request_json={"selector": {"metadata_condition": LANG_IS_DE}, "updates": UPDATES})

        response = await _run_batch_handler(module, handler_name)

        matched = sorted(METAS["lang"]["de"])
        assert response["data"] == {"updated": len(matched), "matched_docs": len(matched)}
        assert calls["batch_update"] == [(DATASET_ID, matched, UPDATES, [])]

    async def test_document_ids_alone_targets_exactly_those(self, monkeypatch: pytest.MonkeyPatch, handler_name: str) -> None:
        """An explicit id list is used as given."""
        module, calls = _load_document_api(monkeypatch, request_json={"selector": {"document_ids": [DOC_C]}, "updates": UPDATES})

        response = await _run_batch_handler(module, handler_name)

        assert response["data"] == {"updated": 1, "matched_docs": 1}
        assert calls["batch_update"] == [(DATASET_ID, [DOC_C], UPDATES, [])]

    async def test_document_ids_and_condition_intersect(self, monkeypatch: pytest.MonkeyPatch, handler_name: str) -> None:
        """Given both, only the ids that also match the condition are updated."""
        module, calls = _load_document_api(monkeypatch, request_json={"selector": {"document_ids": [DOC_A, DOC_C], "metadata_condition": LANG_IS_DE}, "updates": UPDATES})

        response = await _run_batch_handler(module, handler_name)

        assert response["data"] == {"updated": 1, "matched_docs": 1}
        assert calls["batch_update"] == [(DATASET_ID, [DOC_A], UPDATES, [])]

    async def test_condition_matching_nothing_updates_nothing(self, monkeypatch: pytest.MonkeyPatch, handler_name: str) -> None:
        """A condition no document satisfies leaves every document alone."""
        module, calls = _load_document_api(monkeypatch, request_json={"selector": {"metadata_condition": LANG_IS_RU}, "updates": UPDATES})

        response = await _run_batch_handler(module, handler_name)

        assert response["data"] == {"updated": 0, "matched_docs": 0}
        assert calls["batch_update"] == []

    async def test_foreign_document_id_is_rejected(self, monkeypatch: pytest.MonkeyPatch, handler_name: str) -> None:
        """An id outside the dataset is still an error, and nothing is updated."""
        foreign_id = "doc-of-another-dataset"
        module, calls = _load_document_api(monkeypatch, request_json={"selector": {"document_ids": [DOC_A, foreign_id]}, "updates": UPDATES})

        response = await _run_batch_handler(module, handler_name)

        assert response["code"] == RetCode.DATA_ERROR
        assert foreign_id in response["message"]
        assert calls["batch_update"] == []


@pytest.mark.p1
class TestUpdateDocumentReportsStoredMetadata:
    """`update_document` answers with the metadata the doc-meta index holds."""

    async def test_response_carries_the_stored_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The response reports the stored metadata, not the request's echo.

        The document row has no `meta_fields` column, so `map_doc_keys` cannot
        supply the key; a caller that writes crawl metadata and reads the
        response back needs the value a later `metadata_condition` will see.
        """
        requested = {"source": "crawler"}
        stored = {"source": "crawler", "crawled_at": "2026-01-01"}
        module, calls = _load_document_api(monkeypatch, request_json={"meta_fields": requested}, stored_metadata=stored)

        response = await module.update_document(tenant_id="tenant-1", dataset_id=DATASET_ID, document_id=DOC_A)

        assert calls["metadata_written"] == [(DOC_A, requested)]
        assert response["code"] == RetCode.SUCCESS
        assert response["data"]["meta_fields"] == stored
        assert response["data"]["meta_fields"] != requested
