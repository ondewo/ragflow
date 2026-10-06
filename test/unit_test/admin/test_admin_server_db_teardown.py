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
"""The admin server must return every request's database connection to the pool.

``admin/server/admin_server.py`` is a separate Flask app from the API server and
is driven continuously by API clients (user create/delete/activate, API-key
generation). A connection kept past the end of a request is a connection lost
for the lifetime of the worker thread, so the ``teardown_request`` hook has to
run ``close_connection()`` for every request and every metadata database, not
only for the one whose driver pools most aggressively.
"""

import ast
import logging
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
from flask import Blueprint, Flask

ROOT = Path(__file__).resolve().parents[3]
ADMIN_SERVER = ROOT / "admin" / "server" / "admin_server.py"


def _stub_modules() -> dict[str, types.ModuleType]:
    """
    Build stand-ins for the admin_server imports the teardown hook does not touch.

    Importing them for real pulls in the whole API dependency graph (routes -> services -> the
    dialog service -> the agentic RAG stack), so the stubs keep the import to the module under test.

    Returns:
        dict[str, types.ModuleType]: The stub modules, keyed by the name they are imported under.
    """
    routes: types.ModuleType = types.ModuleType("routes")
    routes.admin_bp = Blueprint("admin_bp_stub", "admin_bp_stub")

    auth: types.ModuleType = types.ModuleType("auth")
    auth.init_default_admin = lambda: None
    auth.setup_auth = lambda _login_manager: None

    config: types.ModuleType = types.ModuleType("config")
    config.load_configurations = lambda _path: {}
    config.SERVICE_CONFIGS = types.SimpleNamespace(configs={})

    db_models: types.ModuleType = types.ModuleType("api.db.db_models")
    db_models.close_connection = lambda: None

    return {"routes": routes, "auth": auth, "config": config, "api.db.db_models": db_models}


@pytest.fixture
def admin_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Import ``admin_server`` with its heavy dependencies stubbed out."""
    for name, module in _stub_modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "admin_server", raising=False)

    import admin_server as module

    assert module.__file__ == str(ADMIN_SERVER), f"unexpected admin_server resolved: {module.__file__}"

    try:
        yield module
    finally:
        sys.modules.pop("admin_server", None)


@pytest.fixture
def closed(admin_server: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """Record every ``close_connection()`` the teardown hook performs."""
    calls: list[None] = []
    monkeypatch.setattr(admin_server, "close_connection", lambda: calls.append(None))

    return calls


@pytest.mark.parametrize("database_type", ["postgres", "mysql", "gaussdb", "opengauss"])
def test_connection_is_returned_for_every_database_type(
    admin_server: types.ModuleType,
    closed: list[None],
    monkeypatch: pytest.MonkeyPatch,
    database_type: str,
) -> None:
    monkeypatch.setattr(admin_server.settings, "DATABASE_TYPE", database_type, raising=False)

    admin_server._db_close(None)

    assert closed == [None], f"DATABASE_TYPE={database_type} left the connection checked out"


def test_failed_request_is_logged_and_still_returns_the_connection(
    admin_server: types.ModuleType,
    closed: list[None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError("admin boom")

    with caplog.at_level(logging.ERROR):
        admin_server._db_close(failure)

    assert closed == [None]
    records: list[logging.LogRecord] = [
        record for record in caplog.records if record.getMessage() == f"Admin request failed: {failure}"
    ]
    assert len(records) == 1, f"expected the request failure to be logged once, got {caplog.messages}"
    assert records[0].exc_info is not None and records[0].exc_info[1] is failure


def test_successful_request_is_not_logged_as_a_failure(
    admin_server: types.ModuleType,
    closed: list[None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR):
        admin_server._db_close(None)

    assert closed == [None]
    assert caplog.records == []


def test_flask_teardown_runs_for_successful_and_failing_requests(
    admin_server: types.ModuleType,
    closed: list[None],
) -> None:
    app = Flask(__name__)
    app.teardown_request(admin_server._db_close)

    @app.get("/ok")
    def ok() -> str:
        return "ok"

    @app.get("/boom")
    def boom() -> str:
        raise RuntimeError("admin boom")

    client = app.test_client()
    assert client.get("/ok").status_code == 200
    assert client.get("/boom").status_code == 500

    assert len(closed) == 2, "the teardown hook must return the connection after every request"


def test_teardown_registration_is_not_guarded(admin_server: types.ModuleType) -> None:
    """A registration nested in a condition is the shape that leaked connections."""
    module: ast.Module = ast.parse(ADMIN_SERVER.read_text(encoding="utf-8"))
    main_blocks: list[ast.If] = [
        node for node in module.body if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
    ]
    assert len(main_blocks) == 1, f"expected one __main__ block, found {len(main_blocks)}"

    registrations: list[str] = [
        ast.unparse(statement)
        for statement in main_blocks[0].body
        if isinstance(statement, ast.Expr) and "teardown_request" in ast.unparse(statement)
    ]

    assert registrations == [f"app.teardown_request({admin_server._db_close.__name__})"]
