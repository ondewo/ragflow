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
"""A chat assistant's near-duplicate suppression settings reach retrieval.

`dedup_threshold` and `dedup_before_rerank` are columns on the dialog, so every
turn of the chat-completion path has to hand them to `Dealer.retrieval`;
otherwise the assistant stores a setting it never applies. The dialog object
`async_chat` receives is not always a full row -- a column projection omits what
it did not select, and a row written before a column existed reads back as
NULL -- so both shapes must still retrieve, with suppression off.

Tests drive the real `dialog_service.async_chat()` with the heavy dependencies
stubbed, reusing the stand-ins from ``test_dialog_service_final_answer``.
"""

import pytest

from test_dialog_service_final_answer import (  # noqa: E402
    _KB,
    _LLM_CONFIG,
    _StreamingChatModel,
    _StubRetriever,
    _collect,
    _make_dialog,
)

from api.db.services import dialog_service  # noqa: E402

QUESTION = "Wie melde ich einen Schaden?"


class _RecordingRetriever(_StubRetriever):
    """A `_StubRetriever` that also records the kwargs of every retrieval."""

    def __init__(self):
        self.calls: list[dict] = []

    async def retrieval(self, *args, **kwargs):
        self.calls.append(kwargs)
        return await super().retrieval(*args, **kwargs)


def _drive_async_chat(monkeypatch, dialog):
    """Run one non-streaming turn of async_chat for `dialog`; return the retriever."""
    chat_mdl = _StreamingChatModel("Melden Sie den Schaden telefonisch.")
    retriever = _RecordingRetriever()

    monkeypatch.setattr(dialog_service, "resolve_model_type", lambda _tid, _llm_id: ["chat"])
    monkeypatch.setattr(dialog_service, "resolve_model_config", lambda _tid, _type, _llm_id: _LLM_CONFIG)
    monkeypatch.setattr(dialog_service.TenantLangfuseService, "filter_by_tenant", lambda tenant_id: None)
    # get_models returns (kbs, embd_mdl, rerank_mdl, chat_mdl, tts_mdl)
    monkeypatch.setattr(dialog_service, "get_models", lambda _dialog, **_kwargs: ([_KB], chat_mdl, None, chat_mdl, None))
    monkeypatch.setattr(dialog_service.KnowledgebaseService, "get_field_map", lambda _kb_ids: {})
    monkeypatch.setattr(dialog_service.KnowledgebaseService, "get_by_ids", lambda _ids: [_KB])
    monkeypatch.setattr(dialog_service.settings, "retriever", retriever, raising=False)
    monkeypatch.setattr(dialog_service, "label_question", lambda _q, _kbs: "")
    # kb_prompt calls DocumentService.get_by_ids which needs a live DB; stub it out.
    monkeypatch.setattr(dialog_service, "kb_prompt", lambda _kbinfos, _max_tokens, **_kw: ["RAGFlow ist eine RAG-Engine."])

    events = _collect(dialog_service.async_chat(dialog, [{"role": "user", "content": QUESTION}], stream=False))

    assert events, "async_chat must yield at least one event"
    assert len(retriever.calls) == 1, f"expected exactly one retrieval, got {len(retriever.calls)}"
    return retriever.calls[0]


@pytest.mark.p1
class TestDialogDedupSettingsReachRetrieval:
    """What the chat-completion path passes on for the two dialog columns."""

    def test_configured_settings_are_passed_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dialog configured to suppress near-duplicates says so on every turn."""
        dialog = _make_dialog(None)
        dialog.dedup_threshold = 0.9
        dialog.dedup_before_rerank = True

        kwargs = _drive_async_chat(monkeypatch, dialog)

        assert kwargs["dedup_threshold"] == 0.9
        assert kwargs["dedup_before_rerank"] is True

    def test_default_settings_disable_suppression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The column defaults retrieve exactly as retrieval did without the feature."""
        dialog = _make_dialog(None)
        dialog.dedup_threshold = 0.0
        dialog.dedup_before_rerank = False

        kwargs = _drive_async_chat(monkeypatch, dialog)

        assert kwargs["dedup_threshold"] == 0.0
        assert kwargs["dedup_before_rerank"] is False

    def test_dialog_without_the_columns_still_retrieves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A projection that did not select the columns must not break the turn."""
        dialog = _make_dialog(None)
        assert not hasattr(dialog, "dedup_threshold")
        assert not hasattr(dialog, "dedup_before_rerank")

        kwargs = _drive_async_chat(monkeypatch, dialog)

        assert kwargs["dedup_threshold"] == 0.0
        assert kwargs["dedup_before_rerank"] is False

    def test_null_columns_are_read_as_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A row written before the columns existed reads back NULL, which means off."""
        dialog = _make_dialog(None)
        dialog.dedup_threshold = None
        dialog.dedup_before_rerank = None

        kwargs = _drive_async_chat(monkeypatch, dialog)

        assert kwargs["dedup_threshold"] == 0.0
        assert kwargs["dedup_before_rerank"] is False
