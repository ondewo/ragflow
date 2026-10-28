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
"""Tests for how a failed object-storage teardown is weighed against the scope it failed on.

``DocumentService.remove_documents_of_kb`` tears a dataset's stored objects
down either in one ``remove_bucket`` call covering the whole dataset, or - on a
backend that offers none - object by object. The two failures are not the same
size, so they do not share a policy:

* the dataset-wide teardown failing means every object of the dataset is about
  to be stranded with nothing left to find it by, so the removal stops and the
  dataset stays deletable;
* one document's chunk images or thumbnail failing strands that one object, and
  ``remove_document`` labels both of those deletes "non-critical, log and
  continue", so this path does the same. Giving up on the dataset instead would
  make it undeletable for as long as the single object stays unremovable, and
  would hand the same backend two answers depending on whether the caller
  deleted the documents or the dataset.
"""

import logging
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

KB_ID = "kb-9"
TENANT_ID = "tenant-9"
REPO_ROOT = Path(__file__).resolve().parents[5]

# Every phase of the removal that deletes a row by which the dataset's stored objects could still be found.
ROW_PHASES = ("doc_store_delete", "delete_kb_metadata", "delete_files", "delete_file2documents", "delete_documents")


class _FakeQuery:
    """Lazy peewee-like query yielding preset rows instead of hitting a database."""

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


def _document(doc_id, thumbnail=""):
    return SimpleNamespace(id=doc_id, kb_id=KB_ID, thumbnail=thumbnail)


def _install(
    monkeypatch,
    *,
    docs,
    has_remove_bucket=True,
    remove_bucket_error=None,
    failing_thumbnails=(),
    failing_chunk_images=(),
):
    """Replace every collaborator of remove_documents_of_kb with a recorder.

    Returns the ordered list of phase names the removal asked for; a per-document
    teardown is recorded as ``"storage_rm:<doc id>"`` so the documents reached
    after a failing one can be read off it.
    """
    calls = []

    def _record(name, error=None):
        def _call(*_args, **_kwargs):
            calls.append(name)
            if error is not None:
                raise error

        return _call

    def _rm(_bucket, key):
        doc_id = key.removesuffix(".png")
        calls.append(f"storage_rm:{doc_id}")
        if doc_id in failing_thumbnails:
            raise RuntimeError(f"blob service refused {key}")

    def _delete_chunk_images(_cls, doc, _tenant_id):
        calls.append(f"delete_chunk_images:{doc.id}")
        if doc.id in failing_chunk_images:
            raise RuntimeError(f"chunk image store refused {doc.id}")

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
    monkeypatch.setattr(document_service.DocumentService, "delete_chunk_images", classmethod(_delete_chunk_images))
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

    storage_attrs = {"obj_exist": lambda bucket, key: True, "rm": _rm}
    if has_remove_bucket:
        storage_attrs["remove_bucket"] = _record("remove_bucket", remove_bucket_error)
    monkeypatch.setattr(document_service.settings, "STORAGE_IMPL", SimpleNamespace(**storage_attrs), raising=False)

    return calls


def test_a_failing_thumbnail_teardown_does_not_make_the_dataset_undeletable(monkeypatch, caplog):
    """One unremovable object must not cost the dataset its deletion, only a warning.

    ``remove_document`` deletes the same thumbnail under an explicit
    "non-critical, log and continue", so a dataset delete that aborted here
    would answer differently for the same backend and the same object depending
    only on which call the user made.
    """
    calls = _install(
        monkeypatch,
        docs=[_document("doc-1", thumbnail="doc-1.png")],
        has_remove_bucket=False,
        failing_thumbnails=("doc-1",),
    )

    with caplog.at_level(logging.WARNING):
        deleted = document_service.DocumentService.remove_documents_of_kb(kb_id=KB_ID, tenant_id=TENANT_ID)

    assert deleted == 1
    assert "storage_rm:doc-1" in calls
    for row_phase in ROW_PHASES:
        assert row_phase in calls, f"{row_phase} was skipped although only one object could not be removed"
    assert any("doc-1" in record.message for record in caplog.records if record.levelno == logging.WARNING), "the unremovable object was not reported"


def test_a_failing_chunk_image_teardown_does_not_make_the_dataset_undeletable(monkeypatch):
    """Chunk images carry the same "non-critical" label as the thumbnail."""
    calls = _install(
        monkeypatch,
        docs=[_document("doc-1", thumbnail="doc-1.png")],
        has_remove_bucket=False,
        failing_chunk_images=("doc-1",),
    )

    deleted = document_service.DocumentService.remove_documents_of_kb(kb_id=KB_ID, tenant_id=TENANT_ID)

    assert deleted == 1
    for row_phase in ROW_PHASES:
        assert row_phase in calls, f"{row_phase} was skipped although only one document's chunk images could not be removed"


def test_one_failing_document_does_not_strand_the_objects_of_the_documents_behind_it(monkeypatch):
    """Aborting on the first failure would leave every later document's objects untouched."""
    docs = [_document(f"doc-{i}", thumbnail=f"doc-{i}.png") for i in (1, 2, 3)]
    calls = _install(monkeypatch, docs=docs, has_remove_bucket=False, failing_thumbnails=("doc-1",))

    deleted = document_service.DocumentService.remove_documents_of_kb(kb_id=KB_ID, tenant_id=TENANT_ID)

    assert deleted == 3
    for doc in docs:
        assert f"storage_rm:{doc.id}" in calls, f"the teardown stopped before {doc.id}"


def test_a_failing_dataset_wide_teardown_still_stops_the_removal(monkeypatch):
    """The whole-dataset scope keeps the fatal policy: everything would be stranded at once."""
    calls = _install(monkeypatch, docs=[_document("doc-1")], remove_bucket_error=RuntimeError("seaweedfs is unreachable"))

    with pytest.raises(RuntimeError, match="seaweedfs is unreachable"):
        document_service.DocumentService.remove_documents_of_kb(kb_id=KB_ID, tenant_id=TENANT_ID)

    assert "remove_bucket" in calls
    for row_phase in ROW_PHASES:
        assert row_phase not in calls, f"{row_phase} ran although the dataset-wide teardown had reported a failure"
