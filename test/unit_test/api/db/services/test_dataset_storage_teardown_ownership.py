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
"""Tests for who tears down a dataset's object storage, and what a failed teardown does.

``DocumentService.remove_documents_of_kb`` is the single owner: it drops the
dataset's stored objects ahead of every row that names them, and every caller
reaches storage through it. The failure policy follows from that order and
from the scope that failed - a backend that reports the one dataset-wide
teardown as failed stops the removal, because the rows below are the only thing
that could still find those objects, while a backend that has nothing it can
scope returns normally and the removal carries on.
"""

import ast
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="\\[Errno 13\\] Permission denied\\.  joblib will operate in serial mode",
    category=UserWarning,
)

from api.db.db_models import Document, Task
from api.db.services import document_service

pytestmark = pytest.mark.p2

KB_ID = "kb-7"
TENANT_ID = "tenant-7"
REPO_ROOT = Path(__file__).resolve().parents[5]

# Every phase of the removal that deletes a row pointing at the dataset's stored objects.
ROW_PHASES = ("doc_store_delete", "delete_kb_metadata", "delete_files", "delete_file2documents", "delete_documents")


class _FakeQuery:
    """Lazy peewee-like query that yields preset rows instead of hitting a database."""

    def __init__(self, rows):
        self._rows = rows

    def where(self, *_args, **_kwargs):
        return self

    def __iter__(self):
        return iter(self._rows)


class _FakeDelete:
    def __init__(self, calls, name, rowcount):
        self._calls = calls
        self._name = name
        self._rowcount = rowcount

    def where(self, *_args, **_kwargs):
        self._calls.append(self._name)
        return self

    def execute(self):
        return self._rowcount


def _stub_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _document(doc_id, thumbnail=""):
    return SimpleNamespace(id=doc_id, kb_id=KB_ID, thumbnail=thumbnail)


def _install(monkeypatch, *, docs, has_remove_bucket=True, remove_bucket_error=None, thumbnail_rm_error=None):
    """Replace every collaborator of remove_documents_of_kb with a recorder.

    Returns the ordered list of phase names the removal asked for, so a phase
    that did or did not run after a failed teardown can be read off it.
    """
    calls = []

    def _record(name, error=None):
        def _call(*_args, **_kwargs):
            calls.append(name)
            if error is not None:
                raise error

        return _call

    monkeypatch.setattr(document_service.DB, "connect", lambda *_a, **_k: None)
    monkeypatch.setattr(document_service.DB, "close", lambda *_a, **_k: None)
    monkeypatch.setattr(document_service.DB, "atomic", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(
        document_service.DocumentService,
        "model",
        SimpleNamespace(
            id=Document.id,
            kb_id=Document.kb_id,
            thumbnail=Document.thumbnail,
            select=lambda *_fields: _FakeQuery(list(docs)),
            delete=lambda: _FakeDelete(calls, "delete_documents", len(docs)),
        ),
    )
    monkeypatch.setattr(
        document_service.DocumentService,
        "delete_chunk_images",
        classmethod(lambda cls, doc, tenant_id: calls.append("delete_chunk_images")),
    )
    monkeypatch.setattr(
        document_service,
        "Knowledgebase",
        SimpleNamespace(id=Document.kb_id, update=lambda **_kwargs: _FakeDelete(calls, "reset_kb_counters", 1)),
    )
    monkeypatch.setattr(document_service, "REDIS_CONN", SimpleNamespace(set=lambda *_a, **_k: None))

    _stub_module(
        monkeypatch,
        "api.db.services.task_service",
        TaskService=SimpleNamespace(
            model=SimpleNamespace(id=Task.id, doc_id=Task.doc_id, progress=Task.progress, select=lambda *_fields: _FakeQuery([])),
            filter_delete=_record("delete_tasks"),
        ),
        abort_doc_chunking_counter=lambda doc_id: None,
    )
    _stub_module(monkeypatch, "api.db.services.file_service", FileService=SimpleNamespace(filter_delete=_record("delete_files")))
    _stub_module(monkeypatch, "api.db.services.file2document_service", File2DocumentService=SimpleNamespace(filter_delete=_record("delete_file2documents")))

    monkeypatch.setattr(
        document_service.DocMetadataService,
        "delete_kb_metadata",
        classmethod(lambda cls, kb_id, tenant_id: calls.append("delete_kb_metadata")),
    )
    monkeypatch.setattr(
        document_service.settings,
        "docStoreConn",
        SimpleNamespace(delete=_record("doc_store_delete"), index_exist=lambda index_name, kb_id: True),
        raising=False,
    )

    storage_attrs = {"obj_exist": lambda bucket, key: True, "rm": _record("storage_rm", thumbnail_rm_error)}
    if has_remove_bucket:
        storage_attrs["remove_bucket"] = _record("remove_bucket", remove_bucket_error)
    monkeypatch.setattr(document_service.settings, "STORAGE_IMPL", SimpleNamespace(**storage_attrs), raising=False)

    return calls


class _RemoveBucketCallSites(ast.NodeVisitor):
    """Collects the name of the function enclosing every ``…remove_bucket(…)`` call."""

    def __init__(self):
        self.enclosing_functions = []
        self.sites = []

    def visit_FunctionDef(self, node):
        self.enclosing_functions.append(node.name)
        self.generic_visit(node)
        self.enclosing_functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "remove_bucket":
            self.sites.append(self.enclosing_functions[-1] if self.enclosing_functions else "<module>")
        self.generic_visit(node)


def _remove_bucket_call_sites(relative_path):
    visitor = _RemoveBucketCallSites()
    visitor.visit(ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8")))
    return visitor.sites


def test_the_dataset_delete_path_tears_down_storage_from_exactly_one_place():
    """The handler delegates the teardown; a second call site there would split the failure policy in two.

    Two layers tearing down the same objects means the outcome of a failed
    teardown depends on which call happened to fail - one of them aborts the
    dataset and the other logs and carries on.
    """
    assert _remove_bucket_call_sites("api/apps/services/dataset_api_service.py") == []
    assert _remove_bucket_call_sites("api/db/services/document_service.py") == ["remove_documents_of_kb"]


def test_a_reported_teardown_failure_stops_the_removal_before_any_row_is_deleted(monkeypatch):
    """The rows that name the objects are the only way back to them, so none of them may go."""
    calls = _install(monkeypatch, docs=[_document("doc-1")], remove_bucket_error=RuntimeError("seaweedfs is unreachable"))

    with pytest.raises(RuntimeError, match="seaweedfs is unreachable"):
        document_service.DocumentService.remove_documents_of_kb(kb_id=KB_ID, tenant_id=TENANT_ID)

    assert "remove_bucket" in calls
    for row_phase in ROW_PHASES:
        assert row_phase not in calls, f"{row_phase} ran although the storage teardown had reported a failure"


def test_a_failing_per_document_teardown_does_not_stop_the_removal(monkeypatch):
    """A backend without remove_bucket tears the objects down document by document, and one object is not the dataset.

    The fatal policy follows from the scope the teardown failed on, not from
    the phase: losing the one dataset-wide call strands every object of the
    dataset at once, while losing one document's thumbnail strands that one
    object - which ``remove_document`` deletes under an explicit "non-critical,
    log and continue". Aborting here would make the dataset undeletable for as
    long as that object stays unremovable, and would answer differently for the
    same backend depending on whether the user deleted the documents or the
    dataset. The proportionality is pinned in
    test_remove_documents_of_kb_storage_failure_policy.py.
    """
    calls = _install(
        monkeypatch,
        docs=[_document("doc-1", thumbnail="thumbnail-1.png")],
        has_remove_bucket=False,
        thumbnail_rm_error=RuntimeError("blob service is unreachable"),
    )

    deleted = document_service.DocumentService.remove_documents_of_kb(kb_id=KB_ID, tenant_id=TENANT_ID)

    assert deleted == 1
    assert "storage_rm" in calls
    for row_phase in ROW_PHASES:
        assert row_phase in calls, f"{row_phase} was skipped although only one object could not be removed"


def test_a_backend_that_refuses_the_teardown_without_raising_does_not_block_the_removal(monkeypatch):
    """RAGFlowS3 refuses a shared bucket carrying no prefix_path by logging and returning.

    That is a deployment with nothing it can scope rather than a broken one, so
    the dataset still has to become deletable.
    """
    calls = _install(monkeypatch, docs=[_document("doc-1")])

    deleted = document_service.DocumentService.remove_documents_of_kb(kb_id=KB_ID, tenant_id=TENANT_ID)

    assert deleted == 1
    for row_phase in ROW_PHASES:
        assert row_phase in calls, f"{row_phase} was skipped although the storage teardown had succeeded"
    assert calls.index("remove_bucket") < min(calls.index(row_phase) for row_phase in ROW_PHASES), "the stored objects go before the rows that name them"
