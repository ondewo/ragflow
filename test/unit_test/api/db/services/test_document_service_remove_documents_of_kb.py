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
"""Tests for DocumentService.remove_documents_of_kb().

The dataset teardown must stay set-based: deleting a crawl dataset of tens of
thousands of documents runs synchronously inside the request handler, so a
per-document round trip stalls the worker for its whole duration. The tests
below assert the call pattern - one delete per table whatever the document
count - and that every table and store the per-document ``remove_document``
path touches is still covered.
"""

import sys
import warnings
from contextlib import nullcontext
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
from common.constants import FileSource

pytestmark = pytest.mark.p2

KB_ID = "kb-1"
TENANT_ID = "tenant-1"


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

    def where(self, *args, **_kwargs):
        self._calls.append((self._name, args))
        return self

    def execute(self):
        return self._rowcount


def _fake_document_model(calls, docs):
    model = SimpleNamespace(
        id=Document.id,
        kb_id=Document.kb_id,
        thumbnail=Document.thumbnail,
        select=lambda *_fields: _FakeQuery(docs),
        delete=lambda: _FakeDelete(calls, "delete_documents", len(docs)),
    )
    return model


def _fake_task_model(tasks):
    return SimpleNamespace(
        id=Task.id,
        doc_id=Task.doc_id,
        progress=Task.progress,
        select=lambda *_fields: _FakeQuery(tasks),
    )


def _stub_module(monkeypatch, name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _document(doc_id, thumbnail=""):
    return SimpleNamespace(id=doc_id, kb_id=KB_ID, thumbnail=thumbnail)


class _Recorder:
    """Ordered record of everything the teardown asked a collaborator to do."""

    def __init__(self):
        self.calls = []
        self.cancelled_tasks = []
        self.aborted_docs = []
        self.chunk_image_docs = []
        self.removed_objects = []

    def names(self):
        return [name for name, _args in self.calls]

    def args_of(self, name):
        return [args for call_name, args in self.calls if call_name == name]


def _install(monkeypatch, *, docs, tasks=(), remove_bucket=True, wrapped_backend_has_remove_bucket=True, failing=None):
    recorder = _Recorder()
    calls = recorder.calls
    failing = failing or set()

    def _record(name):
        def _call(*args, **_kwargs):
            calls.append((name, args))
            if name in failing:
                raise RuntimeError(f"{name} is unavailable")

        return _call

    monkeypatch.setattr(document_service.DB, "connect", lambda *_a, **_k: None)
    monkeypatch.setattr(document_service.DB, "close", lambda *_a, **_k: None)
    monkeypatch.setattr(document_service.DB, "atomic", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(document_service.DocumentService, "model", _fake_document_model(calls, list(docs)))

    def _delete_chunk_images(cls, doc, tenant_id):
        calls.append(("delete_chunk_images", (doc.id, tenant_id)))
        recorder.chunk_image_docs.append(doc.id)

    monkeypatch.setattr(document_service.DocumentService, "delete_chunk_images", classmethod(_delete_chunk_images))

    monkeypatch.setattr(
        document_service,
        "Knowledgebase",
        SimpleNamespace(id=Document.kb_id, update=lambda **kwargs: _FakeDelete(calls, f"reset_kb_counters:{sorted(kwargs.items())}", 1)),
        raising=True,
    )

    def _redis_set(key, value, *_args, **_kwargs):
        calls.append(("redis_set", (key, value)))
        recorder.cancelled_tasks.append(key)

    monkeypatch.setattr(document_service, "REDIS_CONN", SimpleNamespace(set=_redis_set), raising=True)

    def _abort(doc_id):
        calls.append(("abort_doc_chunking_counter", (doc_id,)))
        recorder.aborted_docs.append(doc_id)

    _stub_module(
        monkeypatch,
        "api.db.services.task_service",
        TaskService=SimpleNamespace(model=_fake_task_model(list(tasks)), filter_delete=_record("delete_tasks")),
        abort_doc_chunking_counter=_abort,
    )
    _stub_module(
        monkeypatch,
        "api.db.services.file_service",
        FileService=SimpleNamespace(filter_delete=_record("delete_files")),
    )
    _stub_module(
        monkeypatch,
        "api.db.services.file2document_service",
        File2DocumentService=SimpleNamespace(filter_delete=_record("delete_file2documents")),
    )

    delete_kb_metadata = _record("delete_kb_metadata")
    monkeypatch.setattr(
        document_service.DocMetadataService,
        "delete_kb_metadata",
        classmethod(lambda cls, kb_id, tenant_id: delete_kb_metadata(kb_id, tenant_id)),
    )

    doc_store = SimpleNamespace(
        delete=_record("doc_store_delete"),
        index_exist=lambda index_name, kb_id: True,
    )
    monkeypatch.setattr(document_service.settings, "docStoreConn", doc_store, raising=False)

    def _rm(bucket, key):
        calls.append(("storage_rm", (bucket, key)))
        recorder.removed_objects.append(key)

    storage_attrs = {
        "rm": _rm,
        "obj_exist": lambda bucket, key: True,
    }
    if remove_bucket:
        storage_attrs["remove_bucket"] = _record("remove_bucket")
    if not wrapped_backend_has_remove_bucket:
        # Mimic EncryptedStorage: it carries a remove_bucket whatever it wraps and delegates to a
        # backend that may have none of its own, in which case it deletes nothing.
        storage_attrs["storage_impl"] = SimpleNamespace(rm=_rm, obj_exist=lambda bucket, key: True)
    monkeypatch.setattr(document_service.settings, "STORAGE_IMPL", SimpleNamespace(**storage_attrs), raising=False)

    return recorder


@pytest.mark.parametrize("document_count", [1, 50])
def test_remove_documents_of_kb_is_set_based(monkeypatch, document_count):
    """The teardown's call pattern must not grow with the number of documents."""
    docs = [_document(f"doc-{i}") for i in range(document_count)]
    recorder = _install(monkeypatch, docs=docs)

    deleted = document_service.DocumentService.remove_documents_of_kb(KB_ID, TENANT_ID)

    assert deleted == document_count
    assert recorder.names() == [
        "delete_tasks",
        "remove_bucket",
        "doc_store_delete",
        "delete_kb_metadata",
        "delete_files",
        "delete_file2documents",
        "delete_documents",
        "reset_kb_counters:[('chunk_num', 0), ('doc_num', 0), ('token_num', 0)]",
    ]


def test_remove_documents_of_kb_covers_every_store_the_per_document_path_touched(monkeypatch):
    """Nothing the per-document path cleaned up may survive the bulk teardown."""
    recorder = _install(monkeypatch, docs=[_document("doc-1"), _document("doc-2")])

    document_service.DocumentService.remove_documents_of_kb(KB_ID, TENANT_ID)

    names = recorder.names()
    for expected in ("delete_tasks", "remove_bucket", "doc_store_delete", "delete_kb_metadata", "delete_files", "delete_file2documents", "delete_documents"):
        assert expected in names, f"{expected} missing from {names}"

    # One kb_id-scoped sweep stands in for the per-document chunk, nav, wiki-product and graph cleanups.
    condition, index_name, index_kb_id = recorder.args_of("doc_store_delete")[0]
    assert condition == {"kb_id": KB_ID}
    assert index_name == document_service.search.index_name(TENANT_ID)
    assert index_kb_id == KB_ID

    assert recorder.args_of("delete_kb_metadata")[0] == (KB_ID, TENANT_ID)
    assert recorder.args_of("remove_bucket")[0] == (KB_ID,)

    # The document-row delete is scoped to this one dataset, not to every dataset of the tenant.
    (document_filters,) = recorder.args_of("delete_documents")
    (document_scope,) = document_filters
    assert document_scope.lhs is Document.kb_id
    assert document_scope.op == "="
    assert document_scope.rhs == KB_ID


def test_remove_documents_of_kb_deletes_the_document_rows_last(monkeypatch):
    """The id-scoped deletes read a subquery over the document rows, so those rows go last."""
    recorder = _install(monkeypatch, docs=[_document("doc-1")])

    document_service.DocumentService.remove_documents_of_kb(KB_ID, TENANT_ID)

    names = recorder.names()
    for dependent in ("delete_tasks", "delete_files", "delete_file2documents"):
        assert names.index(dependent) < names.index("delete_documents")


def test_remove_documents_of_kb_cancels_unfinished_tasks_once_per_document(monkeypatch):
    """Cancellation is bounded by the in-flight tasks, and a document with several of them is aborted once."""
    tasks = [
        SimpleNamespace(id="task-1", doc_id="doc-1"),
        SimpleNamespace(id="task-2", doc_id="doc-1"),
        SimpleNamespace(id="task-3", doc_id="doc-2"),
    ]
    recorder = _install(monkeypatch, docs=[_document("doc-1"), _document("doc-2")], tasks=tasks)

    document_service.DocumentService.remove_documents_of_kb(KB_ID, TENANT_ID)

    assert recorder.cancelled_tasks == ["task-1-cancel", "task-2-cancel", "task-3-cancel"]
    assert sorted(recorder.aborted_docs) == ["doc-1", "doc-2"]
    assert recorder.names().index("abort_doc_chunking_counter") < recorder.names().index("delete_tasks")


def test_remove_documents_of_kb_falls_back_to_per_document_storage_cleanup(monkeypatch):
    """A storage backend without remove_bucket keeps the chunk-image and thumbnail cleanup it had."""
    docs = [_document("doc-1", thumbnail="thumbnail-1.png"), _document("doc-2", thumbnail=f"{document_service.IMG_BASE64_PREFIX}inline"), _document("doc-3")]
    recorder = _install(monkeypatch, docs=docs, remove_bucket=False)

    document_service.DocumentService.remove_documents_of_kb(KB_ID, TENANT_ID)

    assert "remove_bucket" not in recorder.names()
    assert recorder.chunk_image_docs == ["doc-1", "doc-2", "doc-3"]
    # Only the stored thumbnail is an object; an inline base64 one and an empty one are not.
    assert recorder.removed_objects == ["thumbnail-1.png"]


def test_remove_documents_of_kb_falls_back_when_the_wrapped_backend_cannot_drop_a_bucket(monkeypatch):
    """A storage wrapper only forwards remove_bucket, so the capability belongs to the backend it wraps."""
    docs = [_document("doc-1", thumbnail="thumbnail-1.png"), _document("doc-2")]
    recorder = _install(monkeypatch, docs=docs, wrapped_backend_has_remove_bucket=False)

    document_service.DocumentService.remove_documents_of_kb(KB_ID, TENANT_ID)

    assert "remove_bucket" not in recorder.names()
    assert recorder.chunk_image_docs == ["doc-1", "doc-2"]
    assert recorder.removed_objects == ["thumbnail-1.png"]


def test_remove_documents_of_kb_removes_the_document_rows_when_a_side_store_fails(monkeypatch):
    """A failing non-critical phase must not leave the dataset's rows behind."""
    recorder = _install(monkeypatch, docs=[_document("doc-1")], failing={"doc_store_delete", "delete_kb_metadata"})

    deleted = document_service.DocumentService.remove_documents_of_kb(KB_ID, TENANT_ID)

    assert deleted == 1
    assert "delete_documents" in recorder.names()


def test_remove_documents_of_kb_scopes_the_file_deletes(monkeypatch):
    """The destructive file deletes must not be able to reach another dataset's rows.

    The per-document path resolved one File2Document row at a time, so it could
    only ever delete a file the document actually pointed at. The set-based form
    deletes by subquery, which is only safe while both conditions stay in place:
    ``source_type`` pinned to the knowledge-base source, and the id constrained
    by the dataset's own documents. An unscoped delete here would take every
    tenant's uploads with it.
    """
    recorder = _install(monkeypatch, docs=[_document("doc-1"), _document("doc-2")])

    document_service.DocumentService.remove_documents_of_kb(KB_ID, TENANT_ID)

    file_filters = recorder.args_of("delete_files")
    assert len(file_filters) == 1, "the file delete must be issued exactly once, not per document"
    conditions = file_filters[0][0]
    assert len(conditions) == 2, f"expected a source_type and an id condition, got {conditions}"

    rendered = [str(condition.lhs.name) + " " + str(condition.op) for condition in conditions]
    assert "source_type =" in rendered, f"source_type is not pinned: {rendered}"
    assert any(entry.startswith("id ") and "IN" in entry.upper() for entry in rendered), f"the file id is not constrained to a subquery: {rendered}"

    source_type_condition = next(condition for condition in conditions if condition.lhs.name == "source_type")
    assert source_type_condition.rhs == FileSource.KNOWLEDGEBASE

    f2d_filters = recorder.args_of("delete_file2documents")
    assert len(f2d_filters) == 1
    f2d_conditions = f2d_filters[0][0]
    assert len(f2d_conditions) == 1
    assert f2d_conditions[0].lhs.name == "document_id"
    assert "IN" in str(f2d_conditions[0].op).upper(), "the mapping delete must be scoped to this dataset's documents"
