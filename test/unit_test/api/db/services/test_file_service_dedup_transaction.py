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
"""Tests for the duplicate-folder cleanup in FileService.

``new_a_file_from_kb``, ``init_skills_folder`` and ``init_knowledgebase_docs``
merge duplicate folders inside ``DB.atomic()``. Every statement in that block
must run on the connection the transaction holds: a call into a method
decorated with ``@DB.connection_context()`` closes the database on exit, and
peewee refuses to close while a transaction is open
("Attempting to close database while transaction is open."). The tests drive
the cleanup with peewee's own ``close`` guard and a fake transaction, and
assert that the duplicates are deleted and re-parented without tripping it.
"""

import sys
import types
import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)


def _install_xgboost_stub_if_unavailable():
    if "xgboost" in sys.modules:
        return
    try:
        import xgboost  # noqa: F401
    except Exception:
        sys.modules["xgboost"] = types.ModuleType("xgboost")


_install_xgboost_stub_if_unavailable()

from api.db.db_models import File  # noqa: E402
from api.db.services import file_service as file_service_module  # noqa: E402
from api.db.services.file_service import FileService  # noqa: E402

pytestmark = pytest.mark.p2

TENANT_ID = "tenant-1"
ROOT_ID = "root-1"


class _FakeSelect:
    def __init__(self, rows):
        self._rows = rows

    def where(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def __iter__(self):
        return iter(self._rows)


class _FakeWrite:
    """Records ``(kind, values, rhs of the where clause)`` when executed."""

    def __init__(self, calls, kind, values=None):
        self._calls = calls
        self._kind = kind
        self._values = values
        self._target = None

    def where(self, expression):
        self._target = expression.rhs
        return self

    def execute(self):
        self._calls.append((self._kind, self._values, self._target))
        return 1


def _row(row_id):
    return SimpleNamespace(id=row_id, to_dict=lambda: {"id": row_id})


@pytest.fixture
def duplicate_folders(monkeypatch):
    """Three rows for the same folder, with the transaction state tracked on the real ``DB``."""
    db = file_service_module.DB
    transactions = []
    calls = []

    @contextmanager
    def _atomic():
        transactions.append(object())
        try:
            yield
        finally:
            transactions.pop()

    # Keep peewee's own ``close`` (and with it the open-transaction guard); only the
    # connection itself and the transaction bookkeeping are faked.
    monkeypatch.setattr(db, "atomic", _atomic)
    monkeypatch.setattr(db, "in_transaction", lambda: bool(transactions))
    monkeypatch.setattr(db, "is_closed", lambda: False)

    monkeypatch.setattr(File, "select", classmethod(lambda _cls, *_a: _FakeSelect([_row("keep"), _row("dup-1"), _row("dup-2")])))
    monkeypatch.setattr(File, "update", classmethod(lambda _cls, **values: _FakeWrite(calls, "update", values)))
    monkeypatch.setattr(File, "delete", classmethod(lambda _cls: _FakeWrite(calls, "delete")))
    return calls


@pytest.mark.parametrize(
    "cleanup",
    [
        pytest.param(lambda: FileService.new_a_file_from_kb(TENANT_ID, "folder", ROOT_ID), id="new_a_file_from_kb"),
        pytest.param(lambda: FileService.init_skills_folder(ROOT_ID, TENANT_ID), id="init_skills_folder"),
        pytest.param(lambda: FileService.init_knowledgebase_docs(ROOT_ID, TENANT_ID), id="init_knowledgebase_docs"),
    ],
)
def test_duplicate_folders_are_merged_inside_one_transaction(duplicate_folders, cleanup):
    cleanup()

    assert duplicate_folders == [
        ("update", {"parent_id": "keep"}, "dup-1"),
        ("delete", None, "dup-1"),
        ("update", {"parent_id": "keep"}, "dup-2"),
        ("delete", None, "dup-2"),
    ]
