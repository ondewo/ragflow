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
"""Tests for `_validate_dataset_ids` in `api/apps/restful_apis/chat_api.py`.

The helper authorizes the datasets a chat assistant may be attached to, and
`POST /chats`, `PUT /chats/<id>` and `PATCH /chats/<id>` all route through it.
Its contract is ownership-only: a dataset is rejected when it does not exist or
is not accessible to the caller, and accepted when it exists but holds no parsed
chunks yet. A dataset and its assistant are routinely provisioned in one step,
before any document has been uploaded, and the chunk count only becomes non-zero
once parsing finishes asynchronously — so an empty dataset is a transient state,
not an invalid argument.

The helper's only module-level dependencies are `thread_pool_exec`,
`KnowledgebaseService` and `validate_dataset_embedding_models`, so the tests
extract its definition from the source via `ast` and exec it against fakes for
those three names. That mirrors `test_chat_completions_sanitize_floats.py` and
avoids stubbing the dozens of heavy imports `chat_api.py` pulls in at module
load (quart, peewee models, services).
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_CHAT_API_PATH = Path(__file__).resolve().parents[5] / "api" / "apps" / "restful_apis" / "chat_api.py"
_CHAT_API_SOURCE = _CHAT_API_PATH.read_text()


def _extract(name):
    """Return the module-level function or async function node called `name`."""
    tree = ast.parse(_CHAT_API_SOURCE)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_CHAT_API_PATH}")


def _make_kb(kb_id, *, chunk_num=0, embd_id="embd-1"):
    return SimpleNamespace(id=kb_id, chunk_num=chunk_num, embd_id=embd_id, tenant_embd_id="")


def _load_validator(*, accessible, datasets, embedding_error=None):
    """Exec `_validate_dataset_ids` against fakes and return it plus a call log.

    `accessible` maps dataset id -> whether `KnowledgebaseService.accessible`
    says the caller owns it. `datasets` maps dataset id -> the knowledgebase row
    `KnowledgebaseService.query` returns, or `None` for "no such row".
    `embedding_error` is what `validate_dataset_embedding_models` returns.
    """
    calls = {"accessible": [], "query": [], "embedding_check": []}

    def _accessible(kb_id, user_id):
        calls["accessible"].append((kb_id, user_id))
        return accessible.get(kb_id, False)

    def _query(**kwargs):
        dataset_id = kwargs["id"]
        calls["query"].append(dataset_id)
        kb = datasets.get(dataset_id)
        return [kb] if kb is not None else []

    async def _thread_pool_exec(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def _validate_dataset_embedding_models(kbs):
        calls["embedding_check"].append(list(kbs))
        return embedding_error

    namespace = {
        "thread_pool_exec": _thread_pool_exec,
        "KnowledgebaseService": SimpleNamespace(accessible=_accessible, query=_query),
        "validate_dataset_embedding_models": _validate_dataset_embedding_models,
    }
    extracted = ast.Module(body=[_extract("_validate_dataset_ids")], type_ignores=[])
    exec(compile(extracted, str(_CHAT_API_PATH), "exec"), namespace)
    return namespace["_validate_dataset_ids"], calls


@pytest.mark.p1
class TestUnparsedDatasetIsAcceptable:
    """A dataset without parsed chunks may be attached to a chat assistant."""

    async def test_dataset_with_no_chunks_is_accepted(self):
        validate, _ = _load_validator(accessible={"kb-fresh": True}, datasets={"kb-fresh": _make_kb("kb-fresh", chunk_num=0)})
        assert await validate(["kb-fresh"], "tenant-1") == ["kb-fresh"]

    async def test_parsed_and_unparsed_datasets_mix(self):
        validate, _ = _load_validator(
            accessible={"kb-fresh": True, "kb-parsed": True},
            datasets={"kb-fresh": _make_kb("kb-fresh", chunk_num=0), "kb-parsed": _make_kb("kb-parsed", chunk_num=12)},
        )
        assert await validate(["kb-fresh", "kb-parsed"], "tenant-1") == ["kb-fresh", "kb-parsed"]

    async def test_unparsed_dataset_still_reaches_the_embedding_model_check(self):
        """An accepted dataset must be handed to the embedding-model check.

        Accepting an empty dataset is only correct if it is still compared
        against its siblings — otherwise a fresh dataset could smuggle a
        mismatched embedding model into the assistant.
        """
        validate, calls = _load_validator(
            accessible={"kb-fresh": True, "kb-parsed": True},
            datasets={"kb-fresh": _make_kb("kb-fresh", chunk_num=0), "kb-parsed": _make_kb("kb-parsed", chunk_num=12)},
        )
        await validate(["kb-fresh", "kb-parsed"], "tenant-1")
        assert [kb.id for kb in calls["embedding_check"][0]] == ["kb-fresh", "kb-parsed"]

    async def test_embedding_model_mismatch_is_still_rejected(self):
        validate, _ = _load_validator(
            accessible={"kb-a": True, "kb-b": True},
            datasets={"kb-a": _make_kb("kb-a"), "kb-b": _make_kb("kb-b", embd_id="embd-2")},
            embedding_error="Datasets use different embedding models.",
        )
        assert await validate(["kb-a", "kb-b"], "tenant-1") == "Datasets use different embedding models."

    def test_the_unparsed_rejection_message_is_gone(self):
        """No handler may re-introduce its own inline "not parsed yet" gate."""
        assert "doesn't own parsed file" not in _CHAT_API_SOURCE


@pytest.mark.p1
class TestOwnershipIsStillEnforced:
    """Only the "not parsed yet" condition became acceptable."""

    async def test_inaccessible_dataset_is_rejected(self):
        validate, calls = _load_validator(accessible={"kb-foreign": False}, datasets={"kb-foreign": _make_kb("kb-foreign", chunk_num=42)})
        assert await validate(["kb-foreign"], "tenant-1") == "You don't own the dataset kb-foreign"
        assert calls["query"] == []
        assert calls["embedding_check"] == []

    async def test_unknown_dataset_is_rejected(self):
        validate, calls = _load_validator(accessible={"kb-gone": True}, datasets={})
        assert await validate(["kb-gone"], "tenant-1") == "You don't own the dataset kb-gone"
        assert calls["embedding_check"] == []

    async def test_accessibility_is_checked_for_the_calling_tenant(self):
        validate, calls = _load_validator(accessible={"kb-1": True}, datasets={"kb-1": _make_kb("kb-1")})
        await validate(["kb-1"], "tenant-7")
        assert calls["accessible"] == [("kb-1", "tenant-7")]

    async def test_rejection_short_circuits_before_later_datasets(self):
        validate, calls = _load_validator(accessible={"kb-foreign": False, "kb-own": True}, datasets={"kb-own": _make_kb("kb-own")})
        assert await validate(["kb-foreign", "kb-own"], "tenant-1") == "You don't own the dataset kb-foreign"
        assert calls["accessible"] == [("kb-foreign", "tenant-1")]


@pytest.mark.p2
class TestDatasetIdsShape:
    """Shape handling around the ownership checks is unchanged."""

    async def test_missing_dataset_ids_resolves_to_no_datasets(self):
        validate, calls = _load_validator(accessible={}, datasets={})
        assert await validate(None, "tenant-1") == []
        assert calls["embedding_check"] == []

    async def test_empty_list_is_accepted(self):
        validate, _ = _load_validator(accessible={}, datasets={})
        assert await validate([], "tenant-1") == []

    async def test_non_list_is_rejected(self):
        validate, _ = _load_validator(accessible={}, datasets={})
        assert await validate("kb-1", "tenant-1") == "`dataset_ids` should be a list."

    async def test_falsy_ids_are_dropped(self):
        validate, calls = _load_validator(accessible={"kb-1": True}, datasets={"kb-1": _make_kb("kb-1")})
        assert await validate(["", None, "kb-1"], "tenant-1") == ["kb-1"]
        assert calls["accessible"] == [("kb-1", "tenant-1")]


@pytest.mark.p1
class TestEveryWriteHandlerUsesTheSharedValidator:
    """create, update and patch must not grow their own dataset validation."""

    @pytest.mark.parametrize("handler", ["create", "update_chat", "patch_chat"])
    def test_handler_awaits_validate_dataset_ids(self, handler):
        called = {node.func.id for node in ast.walk(_extract(handler)) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert "_validate_dataset_ids" in called
