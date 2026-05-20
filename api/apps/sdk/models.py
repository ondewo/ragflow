#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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

import asyncio
import logging
import os

from peewee import OperationalError
from quart import request

from api.db.db_models import Dialog, Knowledgebase, TenantLLM
from api.db.services.tenant_llm_service import TenantLLMService
from api.db.services.user_service import TenantService
from api.utils.api_utils import (
    get_error_argument_result,
    get_error_data_result,
    get_result,
    token_required,
)
from api.utils.validation_utils import (
    AddModelReq,
    DeleteModelReq,
    SetDefaultModelsReq,
    UpdateModelReq,
    validate_and_parse_json_request,
)
from common.constants import LLMType, StatusEnum
from rag.llm import ChatModel, EmbeddingModel, RerankModel


_FACTORY = "OpenAI-API-Compatible"
_NAME_SUFFIX = "___OpenAI-API"

_DEFAULT_COLUMN = {
    "chat": ("llm_id", LLMType.CHAT.value),
    "embedding": ("embd_id", LLMType.EMBEDDING.value),
    "rerank": ("rerank_id", LLMType.RERANK.value),
}


def _full_name(name: str) -> str:
    return f"{name}{_NAME_SUFFIX}"


def _canonical_reference(name: str) -> str:
    return f"{name}@{_FACTORY}"


def _parse_model(value: str) -> tuple[str, str] | None:
    """Split '<llm_name>@<llm_factory>' into (llm_name, llm_factory). Returns None if invalid."""
    if "@" not in value:
        return None
    name, _, factory = value.rpartition("@")
    if not name or not factory:
        return None
    return name, factory


def _serialize_row(row: TenantLLM) -> dict:
    return {
        "model_type": row.model_type,
        "model_name": row.llm_name,
        "model_factory": row.llm_factory,
        "base_url": row.api_base or "",
        "max_tokens": row.max_tokens,
        "default_headers": row.default_headers or {},
    }


def _read_tenant_defaults(tenant_id: str) -> dict | None:
    ok, tenant = TenantService.get_by_id(tenant_id)
    if not ok:
        return None
    return {
        "llm": tenant.llm_id,
        "embedding": tenant.embd_id,
        "rerank": tenant.rerank_id,
    }


async def _verify_model(model_type: str, name: str, base_url: str, api_key: str | None, default_headers: dict | None = None) -> str | None:
    """Probe the configured model. Returns None on success, an error string on failure."""
    timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", 10))
    key = api_key or "x"
    headers_kwarg = {"default_headers": default_headers} if default_headers else {}
    try:
        if model_type == LLMType.EMBEDDING:
            mdl = EmbeddingModel[_FACTORY](key=key, model_name=name, base_url=base_url, **headers_kwarg)
            arr, _tc = await asyncio.wait_for(asyncio.to_thread(mdl.encode, ["ping"]), timeout=timeout)
            if arr is None or len(arr) == 0 or len(arr[0]) == 0:
                return "Embedding model returned an empty vector."
        elif model_type == LLMType.CHAT:
            mdl = ChatModel[_FACTORY](key=key, model_name=name, base_url=base_url, provider=_FACTORY, **headers_kwarg)
            answer, tc = await asyncio.wait_for(
                mdl.async_chat(None, [{"role": "user", "content": "ping"}], {"temperature": 0}),
                timeout=timeout,
            )
            if not tc and "**ERROR**:" in answer:
                return answer
        elif model_type == LLMType.RERANK:
            mdl = RerankModel[_FACTORY](key=key, model_name=name, base_url=base_url, **headers_kwarg)
            arr, _tc = await asyncio.wait_for(
                asyncio.to_thread(mdl.similarity, "q", ["a", "b"]),
                timeout=timeout,
            )
            if not len(arr):
                return "Rerank model returned no scores."
    except asyncio.TimeoutError:
        return f"Verification timed out after {timeout}s."
    except Exception as e:
        return f"Verification failed: {e}"
    return None


@manager.route("/models", methods=["POST"])  # noqa: F821
@token_required
async def add_model(tenant_id):
    """Add an OpenAI-API-Compatible model. Verifies reachability before saving."""
    req, err = await validate_and_parse_json_request(request, AddModelReq)
    if err is not None or req is None:
        return get_error_argument_result(err)

    name = req["model_name"] + _NAME_SUFFIX

    existing = TenantLLMService.get_or_none(tenant_id=tenant_id, llm_factory=_FACTORY, llm_name=name)
    if existing is not None:
        return get_error_data_result(message=f"Model '{req["model_name"]}' already exists. Use PUT to update it.")

    headers = req.get("default_headers") or {}
    verify_err = await _verify_model(req["model_type"], name, req["base_url"], req.get("api_key"), headers)
    if verify_err is not None:
        return get_error_data_result(message=verify_err)

    fields = {
        "tenant_id": tenant_id,
        "llm_factory": _FACTORY,
        "model_type": req["model_type"],
        "llm_name": name,
        "api_base": req["base_url"],
        "api_key": req.get("api_key") or "",
        "default_headers": headers,
    }
    if req.get("max_tokens") is not None:
        fields["max_tokens"] = req["max_tokens"]

    try:
        TenantLLMService.save(**fields)
    except OperationalError as e:
        logging.exception(e)
        return get_error_data_result(message="Database operation failed.")

    saved = TenantLLMService.get_or_none(tenant_id=tenant_id, llm_factory=_FACTORY, llm_name=name)
    return get_result(data=_serialize_row(saved))


@manager.route("/models", methods=["PUT"])  # noqa: F821
@token_required
async def update_model(tenant_id):
    """Update fields of an existing model. Re-verifies before saving."""
    req, err = await validate_and_parse_json_request(request, UpdateModelReq, exclude_unset=True)
    if err is not None:
        return get_error_argument_result(err)

    parsed = _parse_model(req.pop("model"))
    if parsed is None:
        return get_error_argument_result(message="model must be in '<model_name>@<model_factory>' format.")
    name, factory = parsed

    if not req:
        return get_error_argument_result(message="No fields provided to update.")

    if "base_url" in req and req["base_url"] is None:
        return get_error_argument_result(message="base_url cannot be cleared.")
    if "max_tokens" in req and req["max_tokens"] is None:
        return get_error_argument_result(message="max_tokens cannot be cleared.")

    row = TenantLLMService.get_or_none(tenant_id=tenant_id, llm_factory=factory, llm_name=name)
    if row is None:
        return get_error_data_result(message=f"Model '{name}@{factory}' not found.")

    merged_base_url = req["base_url"] if "base_url" in req else row.api_base
    merged_max_tokens = req["max_tokens"] if "max_tokens" in req else row.max_tokens
    merged_api_key = (req["api_key"] or "") if "api_key" in req else (row.api_key or "")
    # default_headers: present + null clears (to {}); present + dict replaces; absent keeps existing.
    if "default_headers" in req:
        merged_default_headers = req["default_headers"] or {}
    else:
        merged_default_headers = row.default_headers or {}

    verify_err = await _verify_model(row.model_type, name, merged_base_url, merged_api_key or None, merged_default_headers)
    if verify_err is not None:
        return get_error_data_result(message=verify_err)

    update = {
        "api_base": merged_base_url,
        "api_key": merged_api_key,
        "max_tokens": merged_max_tokens,
        "default_headers": merged_default_headers,
    }
    try:
        TenantLLMService.filter_update(
            [TenantLLM.tenant_id == tenant_id, TenantLLM.llm_factory == factory, TenantLLM.llm_name == name],
            update,
        )
    except OperationalError as e:
        logging.exception(e)
        return get_error_data_result(message="Database operation failed.")

    updated = TenantLLMService.get_or_none(tenant_id=tenant_id, llm_factory=factory, llm_name=name)
    return get_result(data=_serialize_row(updated))


@manager.route("/models", methods=["DELETE"])  # noqa: F821
@token_required
async def delete_model(tenant_id):
    """Remove a model. Refuses if it's referenced by a chat assistant or dataset; clears it from defaults if needed."""
    req, err = await validate_and_parse_json_request(request, DeleteModelReq)
    if err is not None or req is None:
        return get_error_argument_result(err)

    parsed = _parse_model(req["model"])
    if parsed is None:
        return get_error_argument_result(message="model must be in '<model_name>@<model_factory>' format.")
    name, factory = parsed

    row = TenantLLMService.get_or_none(tenant_id=tenant_id, llm_factory=factory, llm_name=name)
    if row is None:
        return get_error_data_result(message=f"Model '{name}@{factory}' not found.")

    display_name = name[:name.find("___")] if "___" in name else name
    refs = [
        display_name,
        name,
        _canonical_reference(name),
    ]
    try:
        if row.model_type == LLMType.CHAT:
            users = list(Dialog.select(Dialog.id).where(Dialog.tenant_id == tenant_id, Dialog.status == StatusEnum.VALID.value, Dialog.llm_id.in_(refs)).limit(5))
            if users:
                ids = ", ".join(d.id for d in users)
                return get_error_data_result(message=f"Model '{name}@{factory}' is used as the LLM by chat assistant(s): {ids}.")
        elif row.model_type == LLMType.EMBEDDING:
            users = list(Knowledgebase.select(Knowledgebase.id).where(Knowledgebase.tenant_id == tenant_id, Knowledgebase.status == StatusEnum.VALID.value, Knowledgebase.embd_id.in_(refs)).limit(5))
            if users:
                ids = ", ".join(k.id for k in users)
                return get_error_data_result(message=f"Model '{name}@{factory}' is used as the embedding model by dataset(s): {ids}.")
        elif row.model_type == LLMType.RERANK:
            users = list(Dialog.select(Dialog.id).where(Dialog.tenant_id == tenant_id, Dialog.status == StatusEnum.VALID.value, Dialog.rerank_id.in_(refs)).limit(5))
            if users:
                ids = ", ".join(d.id for d in users)
                return get_error_data_result(message=f"Model '{name}@{factory}' is used as the rerank model by chat assistant(s): {ids}.")

        tenant = TenantService.get_or_none(id=tenant_id)
        if tenant is not None:
            clear = {}
            if row.model_type == LLMType.CHAT and tenant.llm_id in refs:
                clear["llm_id"] = ""
            elif row.model_type == LLMType.EMBEDDING and tenant.embd_id in refs:
                clear["embd_id"] = ""
            elif row.model_type == LLMType.RERANK and tenant.rerank_id in refs:
                clear["rerank_id"] = ""
            if clear:
                TenantService.update_by_id(tenant_id, clear)

        TenantLLMService.filter_delete(
            [TenantLLM.tenant_id == tenant_id, TenantLLM.llm_factory == factory, TenantLLM.llm_name == name],
        )
    except OperationalError as e:
        logging.exception(e)
        return get_error_data_result(message="Database operation failed.")

    return get_result()


@manager.route("/models", methods=["GET"])  # noqa: F821
@token_required
def list_models(tenant_id):
    """List all models configured for the current tenant. API keys are never returned."""
    try:
        rows = TenantLLMService.query(tenant_id=tenant_id, llm_factory=_FACTORY)
    except OperationalError as e:
        logging.exception(e)
        return get_error_data_result(message="Database operation failed.")
    return get_result(data=[_serialize_row(r) for r in rows])


@manager.route("/models/defaults", methods=["GET"])  # noqa: F821
@token_required
def get_default_models(tenant_id):
    """Return the tenant's current default LLM, embedding, and rerank models."""
    defaults = _read_tenant_defaults(tenant_id)
    if defaults is None:
        return get_error_data_result(message="Tenant not found.")
    return get_result(data=defaults)


@manager.route("/models/defaults", methods=["PUT"])  # noqa: F821
@token_required
async def set_default_models(tenant_id):
    """Set one or more default models. Missing fields stay unchanged; null/"" clears."""
    req, err = await validate_and_parse_json_request(request, SetDefaultModelsReq, exclude_unset=True)
    if err is not None:
        return get_error_argument_result(err)
    if not req:
        return get_error_argument_result(message="No fields provided to update.")

    update = {}
    for api_field, value in req.items():
        column, internal_type = _DEFAULT_COLUMN[api_field]
        if not value:
            update[column] = ""
            continue

        parsed = _parse_model(req["model"])
        if parsed is None:
            return get_error_argument_result(message="model must be in '<model_name>@<model_factory>' format.")
        name, factory = parsed

        target = TenantLLMService.get_or_none(tenant_id=tenant_id, llm_factory=factory, llm_name=name)
        if target is None:
            return get_error_data_result(message=f"Model '{value}' not found for default '{api_field}'.")

        if target.model_type != internal_type:
            return get_error_data_result(message=f"Model '{value}' is not a {api_field} model.")

        update[column] = _canonical_reference(value)

    try:
        TenantService.update_by_id(tenant_id, update)
    except OperationalError as e:
        logging.exception(e)
        return get_error_data_result(message="Database operation failed.")

    defaults = _read_tenant_defaults(tenant_id)
    if defaults is None:
        return get_error_data_result(message="Tenant not found.")
    return get_result(data=defaults)
