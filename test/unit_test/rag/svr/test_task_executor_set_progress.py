#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

"""Unit tests for ``rag.svr.task_executor.set_progress``.

``set_progress`` is the only place the executor reports progress from, so it has to return its peewee
connection on every path and it has to give up on a task whose row no longer exists. Both are covered here
with fakes; nothing in this module talks to Postgres, Redis or the document engine.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from peewee import DoesNotExist

_PARAMETRIC_UMAP_KEY = "umap.parametric_umap"
_PDF_PARSER_KEY = "deepdoc.parser.pdf_parser"

# Transitive imports of the task executor that are not installed in every test environment. They are only
# stubbed when genuinely unimportable, so a full environment still exercises the real packages.
_OPTIONAL_IMPORTS: tuple[str, ...] = ("rapidfuzz", "rapidfuzz.distance", "spacy")


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str) -> MagicMock:
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock()


def _make_stub(name: str) -> _StubModule:
    stub = _StubModule(name)
    stub.__path__ = []  # type: ignore[attr-defined]
    return stub


def _stub_parametric_umap() -> None:
    """Satisfy ``umap``'s optional TensorFlow import before anything pulls ``umap`` in.

    Without TensorFlow, importing ``umap`` emits an ``ImportWarning`` that the project's warning filters
    turn into a collection error. ``rag.svr.task_executor_refactor`` reaches ``umap`` through graspologic.
    """
    sys.modules.setdefault(_PARAMETRIC_UMAP_KEY, _make_stub(_PARAMETRIC_UMAP_KEY))


def _stub_unimportable_optional_imports() -> None:
    for name in _OPTIONAL_IMPORTS:
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        sys.modules[name] = _make_stub(name)


def _restore_pdf_parser_module() -> None:
    """Put the real ``deepdoc.parser.pdf_parser`` back before importing the task executor.

    ``test/unit_test/rag/conftest.py`` registers a stub under that key which carries ``RAGFlowPdfParser``
    only, so ``deepdoc.parser`` fails on ``PlainParser``. The task executor imports ``rag.app.*`` and with it
    the real deepdoc parsers, so an environment that can import it at all can import them too.
    """
    module = sys.modules.get(_PDF_PARSER_KEY)
    if module is None or getattr(module, "__file__", None) is not None:
        return
    del sys.modules[_PDF_PARSER_KEY]
    importlib.invalidate_caches()
    importlib.import_module(_PDF_PARSER_KEY)


_stub_parametric_umap()
_stub_unimportable_optional_imports()
_restore_pdf_parser_module()

from common.exceptions import TaskCanceledException  # noqa: E402
from rag.svr import task_executor  # noqa: E402

TASK_ID = "task-6f1b"


@pytest.fixture
def executor(monkeypatch) -> SimpleNamespace:
    """Replace the three collaborators of ``set_progress`` with recording fakes.

    The returned namespace both configures the fakes (``task_row``, ``canceled``, ``update_error``,
    ``canceled_error``) and records what they saw (``lookups``, ``updates``, ``closed``).
    """
    env = SimpleNamespace(
        task_row=object(),
        canceled=False,
        update_error=None,
        canceled_error=None,
        lookups=[],
        updates=[],
        closed=0,
    )

    class _FakeTaskService:
        @staticmethod
        def get_or_none(**kwargs):
            env.lookups.append(kwargs)
            return env.task_row

        @staticmethod
        def update_progress(task_id, info):
            env.updates.append((task_id, info))
            if env.update_error is not None:
                raise env.update_error

    def _has_canceled(task_id):
        if env.canceled_error is not None:
            raise env.canceled_error
        return env.canceled

    def _close_connection():
        env.closed += 1

    monkeypatch.setattr(task_executor, "TaskService", _FakeTaskService)
    monkeypatch.setattr(task_executor, "has_canceled", _has_canceled)
    monkeypatch.setattr(task_executor, "close_connection", _close_connection)

    return env


# ---------------------------------------------------------------------------
# The connection is returned on every path
# ---------------------------------------------------------------------------


def test_reports_progress_and_closes_the_connection(executor):
    task_executor.set_progress(TASK_ID, prog=0.5, msg="halfway")

    assert executor.lookups == [{"id": TASK_ID}]
    assert len(executor.updates) == 1
    task_id, info = executor.updates[0]
    assert task_id == TASK_ID
    assert info["progress"] == 0.5
    assert info["progress_msg"].endswith("halfway")
    assert executor.closed == 1


def test_closes_the_connection_when_the_progress_update_fails(executor):
    executor.update_error = RuntimeError("deadlock detected")

    task_executor.set_progress(TASK_ID, prog=0.5, msg="halfway")

    assert executor.updates != []
    assert executor.closed == 1


def test_closes_the_connection_when_the_cancel_check_fails(executor):
    executor.canceled_error = RuntimeError("redis is unreachable")

    task_executor.set_progress(TASK_ID, msg="halfway")

    assert executor.updates == []
    assert executor.closed == 1


def test_closes_the_connection_when_the_task_was_canceled(executor):
    executor.canceled = True

    with pytest.raises(TaskCanceledException):
        task_executor.set_progress(TASK_ID, prog=0.5, msg="halfway")

    _, info = executor.updates[0]
    assert info["progress"] == -1
    assert "[Canceled]" in info["progress_msg"]
    assert executor.closed == 1


# ---------------------------------------------------------------------------
# A task whose row is gone is abandoned
# ---------------------------------------------------------------------------


def test_abandons_a_task_whose_row_was_deleted(executor):
    """``TaskService.update_progress`` skips a deleted task silently, so the row probe has to abandon it."""
    executor.task_row = None

    with pytest.raises(TaskCanceledException) as excinfo:
        task_executor.set_progress(TASK_ID, msg="halfway")

    assert excinfo.value.msg == "halfway"
    assert executor.updates == []
    assert executor.closed == 1


def test_abandons_a_task_whose_row_was_deleted_while_reporting_a_failure(executor):
    """The error path reports with ``prog=-1``; a deleted row still has to win over writing that failure."""
    executor.task_row = None

    with pytest.raises(TaskCanceledException) as excinfo:
        task_executor.set_progress(TASK_ID, prog=-1, msg="boom")

    assert excinfo.value.msg == "[ERROR]boom"
    assert executor.updates == []


def test_abandons_a_task_when_the_progress_update_raises_does_not_exist(executor):
    executor.update_error = DoesNotExist()

    with pytest.raises(TaskCanceledException):
        task_executor.set_progress(TASK_ID, prog=0.5, msg="halfway")

    assert executor.closed == 1


def test_a_live_task_is_never_abandoned(executor):
    for prog in (None, 0.0, 0.5, 1.0):
        task_executor.set_progress(TASK_ID, prog=prog, msg="halfway")

    assert len(executor.updates) == 4
    assert executor.closed == 4
