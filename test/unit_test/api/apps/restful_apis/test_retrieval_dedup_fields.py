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
"""What `POST /api/v1/retrieval` accepts for near-duplicate suppression.

`dedup_threshold` is a Jaccard similarity over word shingles, so only a value in
[0, 1] carries meaning: 0 disables suppression, 1 only groups chunks with an
identical shingle set. A value outside that range is rejected rather than
clamped, the same way the handler already rejects an out-of-range `knn_top_k` or
`rerank_candidates_count`, so a caller who sends a percentage hears about it
instead of silently getting every chunk suppressed.

Both knobs are forwarded to `Dealer.retrieval` under the names it declares, and
are also honoured when they come from a saved search configuration rather than
from the request body.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from common.constants import RetCode

DATASET_ID = "kb-1"
TENANT_ID = "tenant-1"
QUESTION = "Wie melde ich einen Schaden?"
_KB = SimpleNamespace(id=DATASET_ID, tenant_id=TENANT_ID, embd_id="bge-m3@VLLM")
_MODEL_CONFIG = {"llm_name": "bge-m3", "llm_factory": "VLLM", "model_type": "embedding"}


class _PassthroughManager:
    def route(self, *_args, **_kwargs):
        return lambda func: func


class _LenientModule(ModuleType):
    """A stub module that yields a harmless placeholder for any attribute that
    wasn't explicitly provided. chunk_api.py's top-level
    `from <mod> import a, b` only needs every imported name to exist; symbols
    that are not on the retrieval path below are never called, so a no-op
    placeholder is safe. This keeps the test from rotting each time
    chunk_api.py grows an import.
    """

    def __getattr__(self, _name):
        return lambda *_a, **_k: None


class _RecordingRetriever:
    """Stands in for `settings.retriever`, recording what each retrieval got."""

    def __init__(self):
        self.calls: list[dict] = []

    async def retrieval(self, *_args, **kwargs):
        self.calls.append(kwargs)
        return {"total": 0, "chunks": [], "doc_aggs": {}}

    def retrieval_by_children(self, chunks, _tenant_ids):
        return chunks


def _stub(monkeypatch, name, **attrs):
    mod = _LenientModule(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _load_chunk_api(monkeypatch, *, request_json, search_detail=None):
    """Load chunk_api.py with the minimum stubs to exercise the retrieval route.

    `common.constants` and `api.utils.pagination_utils` are the real modules, so
    RetCode and the page/page_size defaults are the ones the route actually uses.

    Returns the module plus the retriever recording the retrieval kwargs.
    """
    retriever = _RecordingRetriever()

    async def _get_request_json():
        return request_json

    _stub(
        monkeypatch,
        "api.apps",
        login_required=lambda func=None, **_kwargs: (lambda f: f) if func is None else func,
    )
    _stub(monkeypatch, "api.apps.services")
    _stub(monkeypatch, "api.apps.services.structure_graph_common")
    _stub(monkeypatch, "api.db.db_models", Document=SimpleNamespace(), Task=SimpleNamespace())
    _stub(
        monkeypatch,
        "api.db.joint_services.tenant_model_service",
        get_tenant_default_model_by_type=lambda *_a, **_k: _MODEL_CONFIG,
        resolve_model_config=lambda *_a, **_k: _MODEL_CONFIG,
    )
    _stub(
        monkeypatch,
        "api.db.services.doc_metadata_service",
        DocMetadataService=SimpleNamespace(get_flatted_meta_by_kbs=lambda _kb_ids: {}),
    )
    _stub(monkeypatch, "api.db.services.document_counter_service", release_reparse_counters=lambda *_a, **_k: None)
    _stub(monkeypatch, "api.db.services.document_service", DocumentService=SimpleNamespace())
    _stub(
        monkeypatch,
        "api.db.services.knowledgebase_service",
        KnowledgebaseService=SimpleNamespace(
            accessible=lambda **_k: True,
            get_by_ids=lambda _kb_ids: [_KB],
            get_by_id=lambda _kb_id: (True, _KB),
            list_documents_by_ids=lambda _kb_ids: [],
        ),
        validate_dataset_embedding_models=lambda _kbs: None,
    )
    _stub(monkeypatch, "api.db.services.llm_service", LLMBundle=lambda *_a, **_k: SimpleNamespace())
    _stub(
        monkeypatch,
        "api.db.services.search_service",
        SearchService=SimpleNamespace(get_detail=lambda _search_id: search_detail),
    )
    _stub(monkeypatch, "api.db.services.task_service", TaskService=SimpleNamespace(), cancel_all_task_of=lambda *_a, **_k: None)
    _stub(monkeypatch, "api.db.services.tenant_llm_service", TenantLLMService=SimpleNamespace())
    _stub(
        monkeypatch,
        "api.utils.api_utils",
        add_tenant_id_to_kwargs=lambda func: func,
        construct_json_result=lambda **kwargs: dict(kwargs),
        get_error_data_result=lambda message="", **_k: {"code": RetCode.DATA_ERROR, "message": message},
        get_request_json=_get_request_json,
        get_result=lambda *_a, **kwargs: {"code": RetCode.SUCCESS, **kwargs},
        server_error_response=lambda e: {"code": RetCode.EXCEPTION_ERROR, "message": repr(e)},
    )
    _stub(
        monkeypatch,
        "api.utils.reference_metadata_utils",
        enrich_chunks_with_document_metadata=lambda *_a, **_k: None,
        resolve_reference_metadata_preferences=lambda _req, _search_config=None: (False, None),
    )
    _stub(monkeypatch, "common.doc_store")
    _stub(monkeypatch, "common.doc_store.doc_store_base", OrderByExpr=SimpleNamespace)
    _stub(monkeypatch, "common.settings", retriever=retriever, kg_retriever=SimpleNamespace(), docStoreConn=SimpleNamespace())
    _stub(monkeypatch, "rag")
    _stub(monkeypatch, "rag.app")
    _stub(monkeypatch, "rag.app.tag", label_question=lambda _question, _kbs: {})
    _stub(monkeypatch, "rag.nlp", search=SimpleNamespace())
    _stub(monkeypatch, "rag.prompts")
    _stub(monkeypatch, "rag.prompts.generator")

    monkeypatch.setitem(sys.modules, "quart", _LenientModule("quart"))

    # parents[5] = repo root from test/unit_test/api/apps/restful_apis/<file>
    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "api" / "apps" / "restful_apis" / "chunk_api.py"
    spec = importlib.util.spec_from_file_location("test_chunk_retrieval_api_module", module_path)
    module = importlib.util.module_from_spec(spec)
    # `manager` must exist before exec so the @manager.route decorators run.
    module.manager = _PassthroughManager()
    monkeypatch.setitem(sys.modules, "test_chunk_retrieval_api_module", module)
    spec.loader.exec_module(module)

    # Pin the module globals the handler resolves at call time. The sys.modules
    # stubs only guarantee the import succeeds: in a full environment, where a
    # real module is already loaded, `from x import y` binds the real name and
    # the stub is bypassed. Rebinding here makes the run identical in the
    # bare-stub and full-dependency cases.
    module.settings = sys.modules["common.settings"]
    module.KnowledgebaseService = sys.modules["api.db.services.knowledgebase_service"].KnowledgebaseService
    module.validate_dataset_embedding_models = sys.modules["api.db.services.knowledgebase_service"].validate_dataset_embedding_models
    module.SearchService = sys.modules["api.db.services.search_service"].SearchService
    module.DocMetadataService = sys.modules["api.db.services.doc_metadata_service"].DocMetadataService
    module.LLMBundle = sys.modules["api.db.services.llm_service"].LLMBundle
    module.resolve_model_config = sys.modules["api.db.joint_services.tenant_model_service"].resolve_model_config
    module.get_tenant_default_model_by_type = sys.modules["api.db.joint_services.tenant_model_service"].get_tenant_default_model_by_type
    module.label_question = sys.modules["rag.app.tag"].label_question
    module.resolve_reference_metadata_preferences = sys.modules["api.utils.reference_metadata_utils"].resolve_reference_metadata_preferences
    api_utils = sys.modules["api.utils.api_utils"]
    module.get_request_json = _get_request_json
    module.get_error_data_result = api_utils.get_error_data_result
    module.get_result = api_utils.get_result
    module.server_error_response = api_utils.server_error_response
    return module, retriever


async def _retrieve(monkeypatch, request_body, *, search_detail=None):
    """Run the retrieval route for one request body; return its response and the retriever."""
    request_json = {"dataset_ids": [DATASET_ID], "question": QUESTION, **request_body}
    module, retriever = _load_chunk_api(monkeypatch, request_json=request_json, search_detail=search_detail)

    response = await module.retrieval_test(tenant_id=TENANT_ID)

    return response, retriever


def _assert_rejected(response, retriever, field):
    assert response["code"] == RetCode.DATA_ERROR, response
    assert field in response["message"], response["message"]
    assert retriever.calls == [], "a rejected request must not reach the retriever"


@pytest.mark.p1
class TestRetrievalDedupFields:
    """The two dedup request fields: defaults, forwarding, and validation."""

    async def test_absent_fields_disable_suppression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A request that says nothing about dedup retrieves exactly as before."""
        response, retriever = await _retrieve(monkeypatch, {})

        assert response["code"] == RetCode.SUCCESS, response
        assert len(retriever.calls) == 1
        kwargs = retriever.calls[0]
        assert kwargs["dedup_threshold"] == 0.0
        assert kwargs["dedup_before_rerank"] is False

    async def test_fields_reach_the_retriever(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both knobs arrive under the names Dealer.retrieval declares."""
        response, retriever = await _retrieve(monkeypatch, {"dedup_threshold": 0.85, "dedup_before_rerank": True})

        assert response["code"] == RetCode.SUCCESS, response
        kwargs = retriever.calls[0]
        assert kwargs["dedup_threshold"] == 0.85
        assert kwargs["dedup_before_rerank"] is True

    @pytest.mark.parametrize("threshold", [0, 0.5, 1, "0.75"])
    async def test_threshold_is_coerced_to_a_float_inside_the_range(self, monkeypatch: pytest.MonkeyPatch, threshold) -> None:
        """An int or a numeric string is accepted and handed on as a float."""
        response, retriever = await _retrieve(monkeypatch, {"dedup_threshold": threshold})

        assert response["code"] == RetCode.SUCCESS, response
        passed = retriever.calls[0]["dedup_threshold"]
        assert passed == float(threshold)
        assert isinstance(passed, float)

    @pytest.mark.parametrize("threshold", [-0.1, 1.01, 85, 100])
    async def test_threshold_outside_the_unit_interval_is_rejected(self, monkeypatch: pytest.MonkeyPatch, threshold) -> None:
        """A similarity is not a percentage; out of range is an error, not a clamp."""
        response, retriever = await _retrieve(monkeypatch, {"dedup_threshold": threshold})

        _assert_rejected(response, retriever, "dedup_threshold")

    @pytest.mark.parametrize("threshold", ["aggressive", None, [0.5]])
    async def test_non_numeric_threshold_is_rejected(self, monkeypatch: pytest.MonkeyPatch, threshold) -> None:
        response, retriever = await _retrieve(monkeypatch, {"dedup_threshold": threshold})

        _assert_rejected(response, retriever, "dedup_threshold")

    @pytest.mark.parametrize("before_rerank", ["true", 1, None])
    async def test_non_boolean_dedup_before_rerank_is_rejected(self, monkeypatch: pytest.MonkeyPatch, before_rerank) -> None:
        """The flag decides where suppression happens, so a truthy string is not a yes."""
        response, retriever = await _retrieve(monkeypatch, {"dedup_before_rerank": before_rerank})

        _assert_rejected(response, retriever, "dedup_before_rerank")

    async def test_saved_search_config_supplies_the_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stored search configuration carries the dedup settings too."""
        search_detail = {
            "tenant_id": TENANT_ID,
            "search_config": {"dedup_threshold": 0.7, "dedup_before_rerank": True},
        }

        response, retriever = await _retrieve(monkeypatch, {"search_id": "search-1"}, search_detail=search_detail)

        assert response["code"] == RetCode.SUCCESS, response
        kwargs = retriever.calls[0]
        assert kwargs["dedup_threshold"] == 0.7
        assert kwargs["dedup_before_rerank"] is True

    async def test_request_body_overrides_the_saved_search_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit request field wins over the stored one, as for every other knob."""
        search_detail = {
            "tenant_id": TENANT_ID,
            "search_config": {"dedup_threshold": 0.7, "dedup_before_rerank": True},
        }

        response, retriever = await _retrieve(
            monkeypatch,
            {"search_id": "search-1", "dedup_threshold": 0.0, "dedup_before_rerank": False},
            search_detail=search_detail,
        )

        assert response["code"] == RetCode.SUCCESS, response
        kwargs = retriever.calls[0]
        assert kwargs["dedup_threshold"] == 0.0
        assert kwargs["dedup_before_rerank"] is False
