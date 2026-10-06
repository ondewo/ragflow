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
import os
import json
import logging
import asyncio
from urllib.parse import urlparse, urlunparse

from common.constants import LLMType, ActiveStatusEnum, ModelTypeBinary, ModelVerifyStatusEnum, StatusEnum
from common.settings import FACTORY_LLM_INFOS
from api.db.db_models import DB
from api.db.joint_services.tenant_model_service import resolve_model_config, delete_models_by_instance_ids, delete_instances_by_provider_ids
from api.db.services.tenant_model_provider_service import TenantModelProviderService
from api.db.services.tenant_model_instance_service import TenantModelInstanceService
from api.db.services.tenant_model_service import TenantModelService
from api.utils import model_utils
from rag.llm import ChatModel, CvModel, EmbeddingModel, ModelMeta, OcrModel, RerankModel, Seq2txtModel, TTSModel


def _to_int(v, default=500):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _factory_model_types(llm: dict) -> list[str]:
    model_type = llm.get("model_type")
    return model_utils.normalize_model_types(model_type) if model_type else []


def _normalize_provider_base_url(provider_name: str, base_url: str | None):
    if provider_name != "VLLM" or not base_url:
        return base_url
    base_url = base_url.strip().rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def _redact_url(url: str | None) -> str:
    """Drop credentials from a user-supplied URL before it reaches a log or a response.

    Provider base URLs are typed by the user and routinely carry secrets, either as
    userinfo (`https://user:token@host/v1`) or as a query token
    (`https://host/v1?api-key=...`). Keep the scheme, host, port and path, which is
    what makes a discovery failure diagnosable, and drop the rest.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        # A base URL with no authority is malformed for our purposes, and `.port` only
        # validates on access, so an unparsable port raises here rather than at parse
        # time. Either way the input is unsafe to echo back, so give up on it entirely.
        if not parsed.netloc:
            return "<unparsable url>"
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
    except ValueError:
        return "<unparsable url>"
    if parsed.username or parsed.password:
        netloc = f"***@{netloc}"
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


def _scrub_url_secrets(text: str, url: str | None) -> str:
    """Remove every secret-bearing fragment of *url* from provider-produced text.

    A client echoes back the URL it was handed, whole or in part, so replacing only
    the exact string it was given is not enough to keep userinfo and query tokens out
    of a message built from an exception.
    """
    if not text or not url:
        return text
    text = text.replace(url, _redact_url(url))
    try:
        parsed = urlparse(url)
    except ValueError:
        return text
    fragments = [parsed.password, parsed.username, parsed.query, parsed.fragment]
    for fragment in sorted((f for f in fragments if f), key=len, reverse=True):
        text = text.replace(fragment, "***")
    return text


API_KEY_MASK: str = "***"


def _mask_api_key(api_key: str | None) -> str:
    """Return a placeholder in place of a stored API key, keeping only whether one is set.

    Callers (and the provider form in the UI) need to know that an instance has a
    credential; none of them need its value, and responses are routinely logged.
    An unset key stays an empty string so "no key configured" remains distinguishable.
    """
    return API_KEY_MASK if api_key else ""


def _is_masked_api_key(api_key: str | dict | None) -> bool:
    """Whether *api_key* is the placeholder a previous response handed out.

    A client that echoes an instance back unchanged submits the mask rather than the
    real credential, which must leave the stored key alone instead of overwriting it.
    """
    return isinstance(api_key, str) and api_key == API_KEY_MASK


def _validate_default_headers(default_headers: dict | None) -> dict[str, str] | None:
    """Validate the custom HTTP headers an instance adds to every model request.

    Args:
        default_headers (dict | None): header mapping as submitted, or None when the request omits it.

    Returns:
        dict[str, str] | None: the validated mapping, or None when nothing was submitted.

    Raises:
        ValueError: when the value is not an object of string keys and string values.
    """
    if default_headers is None:
        return None
    if not isinstance(default_headers, dict):
        raise ValueError("default_headers must be an object of header name to header value")
    for name, value in default_headers.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("default_headers keys must be non-empty header names")
        if not isinstance(value, str):
            raise ValueError(f"default_headers['{name}'] must be a string")

    # One rule set governs both sides. The read path raises on any header it
    # will not put on the wire, so validating against it here answers a bad
    # header with a 400 at save time rather than storing a configuration that
    # fails every model request made with the instance.
    return _contract_validate_default_headers({name.strip(): value for name, value in default_headers.items()})


# `api.db.services.tenant_llm_service` pulls the user/tenant model layer in behind it, and
# this module is imported by every provider call, so the two header helpers it owns are
# resolved at call time -- as `_find_model_dependents` resolves its dependent tables below.
def _contract_validate_default_headers(default_headers: dict[str, str] | None) -> dict[str, str] | None:
    """Apply the one header rule set shared by the write path, the read path and the probe."""
    from api.db.services.tenant_llm_service import validate_default_headers

    return validate_default_headers(default_headers)


def _probe_model(registry: dict, provider_name: str, *args, default_headers: dict[str, str] | None = None, **kwargs):
    """Build the client the verification probe talks to, carrying the instance's headers.

    A gateway in front of a self-hosted endpoint rejects any request that arrives
    without its header, so a probe built without the configured headers reports a
    correct configuration as a bad credential -- and verification runs before the
    write, so the instance is never saved.

    Whether the client sends the headers is decided by the same `default_headers_kwarg`
    the live request is built with, so the probe and the request that follows it either
    both carry the headers or both do not. Deciding it here a second time would let
    verification pass a configuration the live path cannot construct, or probe with a
    header the live path silently drops.
    """
    from api.db.services.tenant_llm_service import default_headers_kwarg

    model_cls = registry[provider_name]
    headers_kwarg = default_headers_kwarg(client_cls=model_cls, default_headers=default_headers, factory_name=provider_name)
    return model_cls(*args, **headers_kwarg, **kwargs)


def _normalize_provider_api_key(provider_name: str, api_key: str | dict | None):
    if provider_name == "VLLM" and not api_key:
        return "x"
    return api_key


def _bedrock_api_key_config(api_key: str | dict | None) -> dict | None:
    if isinstance(api_key, dict):
        config = api_key
    elif isinstance(api_key, str):
        try:
            config = json.loads(api_key)
        except json.JSONDecodeError:
            return None
    else:
        return None
    if isinstance(config, dict) and config.get("auth_mode") == "bedrock_api_key":
        return config
    return None


def _validate_bedrock_api_key_config(api_key: str | dict | None) -> dict[str, object] | None:
    config = _bedrock_api_key_config(api_key)
    if config is None:
        return None

    bedrock_api_key = config.get("bedrock_api_key")
    if not isinstance(bedrock_api_key, str) or not bedrock_api_key.strip():
        raise ValueError("Bedrock API key must be provided")
    bedrock_region = config.get("bedrock_region")
    if not isinstance(bedrock_region, str) or not bedrock_region.strip():
        raise ValueError("AWS region must be provided")

    return {
        **config,
        "bedrock_api_key": bedrock_api_key.strip(),
        "bedrock_region": bedrock_region.strip(),
    }


# Not every entry in the factory model dictionary carries a context length: 64 of them omit
# it, across every model type (DeepInfra chat and embedding entries make up most of them).
# Fall back to the same default the rest of the API uses.
_DEFAULT_MAX_TOKENS: int = 8192


def _factory_llm_name(llm: dict) -> str:
    return llm.get("name") or llm.get("llm_name", "")


# Tenant-wide default models, as the (model-name column, tenant_model id column) pair per type.
_TENANT_DEFAULT_COLUMNS: tuple[tuple[int, str, str], ...] = (
    (ModelTypeBinary.CHAT.value, "llm_id", "tenant_llm_id"),
    (ModelTypeBinary.EMBEDDING.value, "embd_id", "tenant_embd_id"),
    (ModelTypeBinary.ASR.value, "asr_id", "tenant_asr_id"),
    (ModelTypeBinary.VISION.value, "img2txt_id", "tenant_img2txt_id"),
    (ModelTypeBinary.RERANK.value, "rerank_id", "tenant_rerank_id"),
    (ModelTypeBinary.TTS.value, "tts_id", "tenant_tts_id"),
    (ModelTypeBinary.OCR.value, "ocr_id", "tenant_ocr_id"),
)

# How many dependents to name per model. Enough to act on, short enough to read.
_DEPENDENT_SAMPLE_SIZE: int = 5


def _model_name_references(model_name: str, instance_name: str, provider_name: str) -> list[str]:
    """Every spelling a model-name column can use to point at one model."""
    return [model_name, f"{model_name}@{provider_name}", f"{model_name}@{instance_name}@{provider_name}"]


def _find_model_dependents(tenant_id: str, provider_name: str, instance_name: str, model_obj) -> list[str]:
    """Name the chat assistants and datasets that still use *model_obj*.

    Args:
        tenant_id (str): tenant that owns the dependents.
        provider_name (str): provider/factory name the model belongs to.
        instance_name (str): instance name the model belongs to.
        model_obj (TenantModel): the model row about to be deleted.

    Returns:
        list[str]: one human-readable entry per dependent, empty when nothing references the model.
    """
    # The dependent tables are read only when a deletion is checked, so they are resolved
    # here rather than widening the import surface of a module every provider call loads.
    from api.db.db_models import Dialog, Knowledgebase

    # A stored reference is either the model's tenant_model id (the tenant_*_id columns) or
    # its composite name (the *_id name columns). Resolution prefers the id and falls back
    # to the name, so a reference check has to look at both to be conclusive.
    dependent_tables = (
        (ModelTypeBinary.CHAT.value, "chat assistant", Dialog, Dialog.llm_id, Dialog.tenant_llm_id),
        (ModelTypeBinary.RERANK.value, "chat assistant", Dialog, Dialog.rerank_id, Dialog.tenant_rerank_id),
        (ModelTypeBinary.EMBEDDING.value, "dataset", Knowledgebase, Knowledgebase.embd_id, Knowledgebase.tenant_embd_id),
    )

    references = _model_name_references(model_obj.model_name, instance_name, provider_name)
    dependents: dict[tuple[str, str], str] = {}

    for type_bit, label, table, name_column, id_column in dependent_tables:
        if not model_obj.model_type & type_bit:
            continue
        rows = (
            table.select(table.id, table.name)
            .where(
                table.tenant_id == tenant_id,
                table.status == StatusEnum.VALID.value,
                (name_column.in_(references)) | (id_column == model_obj.id),
            )
            .limit(_DEPENDENT_SAMPLE_SIZE)
        )
        for row in rows:
            dependents[(label, row.id)] = f"{label} '{row.name}' ({row.id})"

    return list(dependents.values())


def _model_deletion_blockers(tenant_id: str, provider_name: str, models: list[tuple[str, object]]) -> list[str]:
    """One message per still-referenced model, naming its dependents.

    Args:
        tenant_id (str): tenant that owns the models and their dependents.
        provider_name (str): provider/factory name the models belong to.
        models (list[tuple[str, object]]): (instance_name, model row) pairs about to be deleted.

    Returns:
        list[str]: a message per blocked model, empty when the deletion may go ahead.
    """
    blockers: list[str] = []
    for instance_name, model_obj in models:
        dependents = _find_model_dependents(tenant_id, provider_name, instance_name, model_obj)
        if dependents:
            composite_name = f"{model_obj.model_name}@{instance_name}@{provider_name}"
            blockers.append(f"Model '{composite_name}' is still used by {', '.join(dependents)}.")
    return blockers


def _clear_tenant_default_models(tenant_id: str, provider_name: str, models: list[tuple[str, object]]) -> None:
    """Drop *models* from the tenant's default-model settings.

    A default pointing at a deleted model resolves to nothing and fails the next request
    that needs that model type, so the tenant row is detached as part of the deletion.

    Args:
        tenant_id (str): tenant whose defaults to clean up.
        provider_name (str): provider/factory name the models belong to.
        models (list[tuple[str, object]]): (instance_name, model row) pairs being deleted.
    """
    # Resolved at call time, as in _find_model_dependents: the tenant row is read only when
    # something is actually being deleted.
    from api.db.services.user_service import TenantService

    tenant = TenantService.get_or_none(id=tenant_id)
    if tenant is None:
        return

    cleared: dict[str, str | None] = {}
    for instance_name, model_obj in models:
        references = _model_name_references(model_obj.model_name, instance_name, provider_name)
        for type_bit, name_column, id_column in _TENANT_DEFAULT_COLUMNS:
            if not model_obj.model_type & type_bit:
                continue
            if getattr(tenant, name_column) in references:
                cleared[name_column] = ""
            if getattr(tenant, id_column) == model_obj.id:
                cleared[id_column] = None

    if cleared:
        TenantService.update_by_id(tenant_id, cleared)


def _instance_models(instance_objs: list) -> list[tuple[str, object]]:
    """Pair every model of every instance with its instance name."""
    return [(instance_obj.instance_name, model_obj) for instance_obj in instance_objs for model_obj in TenantModelService.get_models_by_instance_id(instance_obj.id)]


def list_providers(tenant_id: str, all_available: bool = False):
    """
    List providers for a tenant.

    If available_only is True, list all system-wide providers (pool providers).
    Otherwise, list providers that the tenant has configured, with a has_instance flag.

    :param tenant_id: tenant ID
    :param all_available: whether to list all available providers
    :return: (success, result)
    """
    if not FACTORY_LLM_INFOS:
        return False, []

    factory_rank_mapping = {factory["name"]: -_to_int(factory.get("rank", "500")) for factory in FACTORY_LLM_INFOS}
    factory_info_map = {f["name"]: f for f in FACTORY_LLM_INFOS}
    if all_available:
        providers = []
        for factory_info in FACTORY_LLM_INFOS:
            if factory_info["name"] in ["Youdao", "FastEmbed", "BAAI", "Builtin", "siliconflow_intl"]:
                continue
            model_types = sorted(set(model_type for llm in factory_info.get("llm", []) for model_type in _factory_model_types(llm))) if factory_info.get("llm", []) else []
            if factory_info["name"] in ["MinerU", "PaddleOCR", "OpenDataLoader", "Mistral OCR"]:
                model_types.append("ocr")
            provider = {"model_types": model_types, "name": factory_info["name"], "url": {"default": factory_info.get("url", "")}}
            if factory_info["name"].lower() == "siliconflow":
                provider["url"]["intl"] = factory_info_map.get("siliconflow_intl", {}).get("url", "https://api.siliconflow.com/v1")
            elif factory_info["name"] == "Tongyi-Qianwen":
                provider["url"]["intl"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            providers.append(provider)
        providers.sort(key=lambda x: (factory_rank_mapping.get(x["name"]), x["name"]))
        return True, providers

    # List tenant-configured providers
    factory_names = TenantModelProviderService.list_provider_names_by_tenant_id(tenant_id)

    providers = []
    factory_info_mapping = {f["name"]: f for f in FACTORY_LLM_INFOS}
    for name in factory_names:
        if name not in ["Youdao", "FastEmbed", "BAAI", "Builtin", "siliconflow_intl"] and factory_info_mapping.get(name):
            factory_info = factory_info_mapping[name]
            provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, name)
            has_instance = bool(provider_obj and TenantModelInstanceService.get_all_by_provider_id(provider_obj.id))
            model_types = sorted(set(model_type for llm in factory_info.get("llm", []) for model_type in _factory_model_types(llm))) if factory_info.get("llm", []) else []
            if name in ["MinerU", "PaddleOCR", "OpenDataLoader", "Mistral OCR"]:
                model_types.append("ocr")

            provider = {"has_instance": has_instance, "model_types": model_types, "name": factory_info["name"], "url": {"default": factory_info.get("url", "")}}
            if factory_info["name"].lower() == "siliconflow":
                provider["url"]["intl"] = factory_info_map.get("siliconflow_intl", {}).get("url", "https://api.siliconflow.com/v1")
            elif factory_info["name"] == "Tongyi-Qianwen":
                provider["url"]["intl"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            providers.append(provider)
    providers.sort(key=lambda x: (factory_rank_mapping.get(x["name"]), x["name"]))
    return True, providers


def add_provider(tenant_id: str, provider_name: str):
    """
    Add a provider (factory) for a tenant.

    :param tenant_id: tenant ID
    :param provider_name: provider/factory name
    :return: (success, result_or_error_message)
    """
    if not FACTORY_LLM_INFOS:
        return False, "No providers found"
    # Check if factory is allowed
    allowed_factories = [f["name"] for f in FACTORY_LLM_INFOS]
    if provider_name not in allowed_factories:
        return False, f"Provider '{provider_name}' is not allowed"

    existing = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if existing:
        return False, f"Provider {provider_name} already exists"

    TenantModelProviderService.insert(tenant_id=tenant_id, provider_name=provider_name)
    return True, "success"


def delete_provider(tenant_id: str, provider_id_or_name: str):
    """
    Delete all instances and models for a provider.

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider ID or provider/factory name
    :return: (success, result_or_error_message)
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"Provider {provider_id_or_name} not found"
    instance_objs = TenantModelInstanceService.get_all_by_provider_id(provider_obj.id)
    if instance_objs:
        models = _instance_models(instance_objs)
        blockers = _model_deletion_blockers(tenant_id, provider_obj.provider_name, models)
        if blockers:
            return False, " ".join(blockers + ["Repoint or delete the dependents before deleting the provider."])
        _clear_tenant_default_models(tenant_id, provider_obj.provider_name, models)

        instance_ids = [instance_obj.id for instance_obj in instance_objs]
        delete_models_by_instance_ids(instance_ids)
        delete_instances_by_provider_ids([provider_obj.id])
    TenantModelProviderService.delete_by_tenant_id_and_provider_name(tenant_id, provider_obj.provider_name)
    return True, "success"


def show_provider(provider_id_or_name: str):
    """
    Show provider details from LLMFactories.

    :param provider_id_or_name: provider/factory ID or name
    :return: (success, result_or_error_message)
    """
    provider_obj = None
    if provider_id_or_name:
        _, provider_obj = TenantModelProviderService.get_by_id(provider_id_or_name)
    provider_name = provider_obj.provider_name if provider_obj else provider_id_or_name
    fac_list = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
    if not fac_list:
        return False, f"Provider '{provider_id_or_name}' not found"
    factory_info = fac_list[0]
    return True, {"base_url": {"default": factory_info.get("url", "")}, "name": factory_info["name"], "total_models": len(factory_info.get("llm", []))}


async def list_provider_models(
    provider_id_or_name: str,
    api_key: str | dict | None = None,
    base_url: str | None = None,
):
    """
    List all models for a provider from the LLM dictionary.

    :param provider_id_or_name: provider ID or provider/factory name
    :param api_key: api key
    :param base_url: base url
    :return: (success, result_or_error_message)
    """
    provider_obj = None
    if provider_id_or_name:
        _, provider_obj = TenantModelProviderService.get_by_id(provider_id_or_name)
    provider_name = provider_obj.provider_name if provider_obj else provider_id_or_name
    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
    if not factory_info:
        return False, f"Provider '{provider_id_or_name}' not found"
    api_key = _normalize_provider_api_key(provider_name, api_key)
    static_llms = [
        {
            "name": _factory_llm_name(llm),
            "max_tokens": llm.get("max_tokens", 8192),
            "model_types": _factory_model_types(llm),
            "features": (llm.get("features") if llm.get("features") is not None else ((["is_tools"] if llm.get("is_tools") else []) + (["thinking"] if llm.get("thinking") else []))),
        }
        for llm in factory_info[0]["llm"]
    ]

    model_base_url = _normalize_provider_base_url(provider_name, base_url) or factory_info[0].get("url", "")
    remote_models = []
    bedrock_api_key_config = _bedrock_api_key_config(api_key)
    should_fetch_remote = provider_name in ModelMeta and (provider_name != "Bedrock" or bedrock_api_key_config is not None)
    if should_fetch_remote:
        try:
            remote_models = await ModelMeta[provider_name](api_key, model_base_url).get_model_list()
        except ValueError as error:
            if provider_name == "Bedrock":
                return False, str(error)
            raise

    if provider_name == "Bedrock" and bedrock_api_key_config is not None and not remote_models:
        return False, "No Bedrock models were discovered"

    if not static_llms and not remote_models:
        return True, []

    if provider_name == "Bedrock" and bedrock_api_key_config is not None:
        static_models = {model["name"]: model for model in static_llms}
        models = [
            {
                **model,
                "max_tokens": static_models.get(model["name"], {}).get("max_tokens", model.get("max_tokens", 8192)),
            }
            for model in remote_models
        ]
    else:
        # Merge static and remote models, preferring remote_models on name conflicts
        merged = {m["name"]: m for m in static_llms}
        merged.update({m["name"]: m for m in remote_models})
        models = list(merged.values())

    models.sort(key=lambda x: x["name"])
    return True, models


def show_provider_model(provider_id_or_name: str, model_name: str):
    """
    Show a specific model for a provider.

    :param provider_id_or_name: provider/factory ID or name
    :param model_name: model name
    :return: (success, result_or_error_message)
    """
    provider_obj = None
    if provider_id_or_name:
        _, provider_obj = TenantModelProviderService.get_by_id(provider_id_or_name)
    provider_name = provider_obj.provider_name if provider_obj else provider_id_or_name
    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
    if not factory_info:
        return False, f"Provider '{provider_id_or_name}' not found"
    llms = factory_info[0]["llm"]
    if not llms:
        return False, f"No models found for provider '{provider_id_or_name}'"
    target_llm = [llm for llm in llms if _factory_llm_name(llm) == model_name]
    if not target_llm:
        return False, f"Model '{model_name}' not found"
    llm_info = target_llm[0]

    return True, {
        "name": _factory_llm_name(llm_info),
        "max_tokens": llm_info["max_tokens"],
        "model_types": _factory_model_types(llm_info),
        "thinking": None,
        "model_type_map": {model_type: True for model_type in _factory_model_types(llm_info)},
    }


async def update_provider_instance(
    tenant_id: str,
    provider_id_or_name: str,
    instance_id_or_name: str,
    instance_name: str,
    api_key: str | dict,
    base_url: str,
    region: str,
    model_info: list[dict] = None,
    verify: bool = True,
    default_headers: dict | None = None,
):
    """
    Update a provider instance.

    Updates the instance's api_key, base_url, region, and re-creates all models
    based on the provided model_info list.

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :param instance_id_or_name: instance ID or name
    :param instance_name: instance name (used as a logical identifier)
    :param api_key: API key
    :param base_url: base url
    :param region: region
    :param model_info: model info, [{
        "model_type": ["chat"],  # support multiple
        "model_name": "name",
        "max_tokens": 4096,
        "extra": {
            "is_tools": True
        }
    }]
    :param verify: verify api_key
    :param default_headers: custom HTTP headers to add to every request to this instance;
        None leaves the stored headers untouched, {} removes them
    :return: (success, result_or_error_message)
    """
    if not provider_id_or_name:
        return False, "Provider ID or name is required"

    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"Provider '{provider_id_or_name}' does not exist"

    provider_name = provider_obj.provider_name

    # Find the instance
    instance_obj = None
    if instance_id_or_name:
        _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
    if instance_obj and instance_obj.provider_id != provider_obj.id:
        instance_obj = None
    if not instance_obj:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_id_or_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    base_url = _normalize_provider_base_url(provider_name, base_url)
    api_key = _normalize_provider_api_key(provider_name, api_key)
    region = (region or "").strip()

    # Responses hand out a placeholder instead of the stored key, so a client that submits
    # the instance back unchanged means "keep the key I was never shown".
    if _is_masked_api_key(api_key):
        api_key = instance_obj.api_key

    try:
        bedrock_api_key_config = _validate_bedrock_api_key_config(api_key) if provider_name == "Bedrock" else None
        default_headers = _validate_default_headers(default_headers)
    except ValueError as error:
        return False, str(error)
    bedrock_api_key_auth = bedrock_api_key_config is not None
    if bedrock_api_key_auth:
        api_key = bedrock_api_key_config

    api_key_str = ""
    if api_key:
        api_key_str = api_key if isinstance(api_key, str) else json.dumps(api_key)

    existing_extra = json.loads(instance_obj.extra) if instance_obj.extra else {}
    # The probe has to use the headers the instance will carry once this update lands:
    # the submitted set, the stored set when the request omits the field, and none at
    # all when it submits {} to clear them. Any other set either rejects a correct
    # configuration or accepts one that cannot reach the endpoint.
    effective_headers = default_headers if default_headers is not None else existing_extra.get("default_headers")

    # Verify api_key
    model_verify_result = {}
    runtime_verify = verify and not bedrock_api_key_auth
    if runtime_verify:
        success, msg, model_verify_result = await verify_api_key(provider_name, api_key, base_url, region, model_info, default_headers=effective_headers)
        if not success:
            return False, msg

    # Update instance record
    update_dict = {
        "api_key": api_key_str,
    }
    if instance_name != instance_obj.instance_name:
        update_dict["instance_name"] = instance_name

    extra_fields = {}
    if base_url:
        extra_fields["base_url"] = base_url
    if region:
        extra_fields["region"] = region
    # Preserve existing extra fields not overwritten
    existing_extra.update(extra_fields)
    if default_headers is not None:
        # An omitted default_headers keeps the stored headers; an empty object removes them.
        if default_headers:
            existing_extra["default_headers"] = default_headers
        else:
            existing_extra.pop("default_headers", None)
    update_dict["extra"] = json.dumps(existing_extra)
    TenantModelInstanceService.update_by_id(instance_obj.id, update_dict)

    # Use the (possibly updated) instance_name for model recreation
    effective_instance_name = instance_name

    # Upsert models: add new ones, update existing ones, remove ones no longer selected
    existing_model_objs = TenantModelService.get_models_by_instance_id(instance_obj.id)
    existing_model_names = {model_obj.model_name: model_obj for model_obj in existing_model_objs}

    # Delete models that are no longer in the submitted model_info
    submitted_model_names = set()
    if model_info:
        submitted_model_names = {m.get("model_name") for m in model_info if m.get("model_name")}
    elif model_info is not None:
        # model_info is explicitly an empty list — remove all models
        submitted_model_names = set()
    models_to_remove = set(existing_model_names.keys()) - submitted_model_names
    if models_to_remove:
        TenantModelService.delete_by_ids([existing_model_names[n].id for n in models_to_remove])

    msg = ""
    if model_info:
        for model in model_info:
            model_name = model.get("model_name")
            if not model_name:
                continue
            if runtime_verify:
                verify_status = model_verify_result.get(model_name, ModelVerifyStatusEnum.UNKNOWN.value)
                if model.get("extra"):
                    model["extra"].update({"verify": verify_status})
                else:
                    model["extra"] = {"verify": verify_status}

            if model_name in existing_model_names:
                # Update existing model
                update_dict = {}
                if isinstance(model.get("model_type"), (str, list)):
                    target_model_type = model_utils.calculate_model_type(model["model_type"])
                    if target_model_type != existing_model_names[model_name].model_type:
                        update_dict["model_type"] = target_model_type
                merged_extra = json.loads(existing_model_names[model_name].extra) if existing_model_names[model_name].extra else {}
                merged_extra.update(model.get("extra") or {})
                if "max_tokens" in model:
                    merged_extra.update({"max_tokens": model["max_tokens"]})
                update_dict["extra"] = json.dumps(merged_extra)
                if update_dict:
                    TenantModelService.update_model(existing_model_names[model_name].id, update_dict)
            else:
                # Add new model
                success, _msg = add_model_to_instance(tenant_id, provider_name, effective_instance_name, **model)
                if not success:
                    msg += _msg
    else:
        if model_info is None:
            # model_info not provided — add all factory default models (same as create)
            factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_name]
            factory_llms = factory_info[0]["llm"]
            for llm in factory_llms:
                llm_name = _factory_llm_name(llm)
                if llm_name in existing_model_names:
                    # Update existing
                    update_dict = {}
                    target_model_type = model_utils.calculate_model_type(_factory_model_types(llm))
                    if target_model_type != existing_model_names[llm_name].model_type:
                        update_dict["model_type"] = target_model_type
                    db_extra = json.loads(existing_model_names[llm_name].extra) if existing_model_names[llm_name].extra else {}
                    db_extra_fields = {
                        "max_tokens": llm.get("max_tokens", _DEFAULT_MAX_TOKENS),
                        "is_tools": llm.get("is_tools", False),
                        "thinking": "thinking" in llm.get("features", []),
                    }
                    if runtime_verify:
                        verify_status = model_verify_result.get(llm_name, ModelVerifyStatusEnum.UNKNOWN.value)
                        db_extra_fields["verify"] = verify_status
                    db_extra.update(db_extra_fields)
                    update_dict["extra"] = json.dumps(db_extra)
                    if update_dict:
                        TenantModelService.update_model(existing_model_names[llm_name].id, update_dict)
                else:
                    extra_fields = {
                        "is_tools": llm.get("is_tools", False),
                        "thinking": "thinking" in llm.get("features", []),
                    }
                    if runtime_verify:
                        verify_status = model_verify_result.get(llm_name, ModelVerifyStatusEnum.UNKNOWN.value)
                        extra_fields["verify"] = verify_status
                    max_tokens = llm.get("max_tokens", _DEFAULT_MAX_TOKENS)
                    success, _msg = add_model_to_instance(
                        tenant_id, provider_name, effective_instance_name, **{"model_type": _factory_model_types(llm), "model_name": llm_name, "max_tokens": max_tokens, "extra": extra_fields}
                    )
                    if not success:
                        msg += _msg
    if msg:
        return False, msg
    return True, "success"


async def create_provider_instance(
    tenant_id: str,
    provider_id_or_name: str,
    instance_name: str,
    api_key: str | dict,
    base_url: str,
    region: str,
    model_info: list[dict] = None,
    default_headers: dict | None = None,
):
    """
    Create a provider instance.

    The instance_name parameter is accepted for API compatibility but in the old
    model all records under a factory share the same API key configuration.

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :param instance_name: instance name (used as a logical identifier)
    :param api_key: API key
    :param base_url: base url
    :param region: region
    :param model_info: model info, [{
        "model_type": ["chat"],  # support multiple
        "model_name": "name",
        "max_tokens": 4096,
        "extra": {
            "field1": "value1",
            "field2": "'value2"
        }
    }]
    :param default_headers: custom HTTP headers to add to every request to this instance
    :return: (success, result_or_error_message)
    """
    if not provider_id_or_name:
        return False, "Provider ID or name is required"

    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"Provider '{provider_id_or_name}' does not exist"

    provider_name = provider_obj.provider_name

    base_url = _normalize_provider_base_url(provider_name, base_url)
    api_key = _normalize_provider_api_key(provider_name, api_key)
    region = (region or "").strip()

    if instance_name == "default":
        return False, "Instance name cannot be 'default'"

    # Check if provider exists in the system
    allowed_factories = [f["name"] for f in FACTORY_LLM_INFOS]
    if provider_name not in allowed_factories:
        return False, f"Provider '{provider_name}' is not allowed"

    try:
        bedrock_api_key_config = _validate_bedrock_api_key_config(api_key) if provider_name == "Bedrock" else None
        default_headers = _validate_default_headers(default_headers)
    except ValueError as error:
        return False, str(error)
    bedrock_api_key_auth = bedrock_api_key_config is not None
    if bedrock_api_key_auth:
        api_key = bedrock_api_key_config

    api_key_str = ""
    if api_key:
        api_key_str = api_key if isinstance(api_key, str) else json.dumps(api_key)

    if bedrock_api_key_auth:
        if not model_info:
            return False, "At least one Bedrock model must be selected"
        model_verify_result = {}
    else:
        success, verify_msg, model_verify_result = await verify_api_key(provider_name, api_key, base_url, region, model_info, default_headers=default_headers)
        if not success:
            return False, verify_msg

    extra_fields = {}
    if base_url:
        extra_fields["base_url"] = base_url
    if region:
        extra_fields["region"] = region
    if default_headers:
        extra_fields["default_headers"] = default_headers
    TenantModelInstanceService.create_instance(provider_id=provider_obj.id, instance_name=instance_name, api_key=api_key_str, extra=json.dumps(extra_fields))
    if model_info:
        msg = ""
        for model in model_info:
            if model.get("extra"):
                model["extra"].update({"verify": model_verify_result.get(model["model_name"], ModelVerifyStatusEnum.UNKNOWN.value)})
            else:
                model["extra"] = {"verify": model_verify_result.get(model["model_name"], ModelVerifyStatusEnum.UNKNOWN.value)}
            success, _msg = add_model_to_instance(tenant_id, provider_name, instance_name, **model)
            if not success:
                msg += _msg
        if msg:
            return False, msg
    else:
        msg = ""
        target_factory_name = "siliconflow_intl" if provider_name.lower() == "siliconflow" and region == "intl" else provider_name
        factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == target_factory_name]
        factory_llms = factory_info[0]["llm"]
        for llm in factory_llms:
            llm_name = _factory_llm_name(llm)
            success, _msg = add_model_to_instance(
                tenant_id,
                provider_name,
                instance_name,
                **{
                    "model_type": _factory_model_types(llm),
                    "model_name": llm_name,
                    "max_tokens": llm.get("max_tokens", _DEFAULT_MAX_TOKENS),
                    "extra": {
                        "is_tools": llm.get("is_tools", False),
                        "thinking": "thinking" in llm.get("features", []),
                        "verify": model_verify_result.get(llm_name, ModelVerifyStatusEnum.UNKNOWN.value),
                    },
                },
            )
            if not success:
                msg += _msg
        if msg:
            return False, msg

    return True, "success"


async def create_name_only_provider_instance(tenant_id: str, provider_name: str, instance_name: str):
    """
    Create a provider instance with only a name (no api_key/base_url validation).

    :param tenant_id: tenant ID
    :param provider_name: provider/factory name
    :param instance_name: instance name (used as a logical identifier)
    :return: (success, result_or_error_message)
    """
    if not provider_name:
        return False, "Provider name is required"

    if instance_name == "default":
        return False, "Instance name cannot be 'default'"

    allowed_factories = [f["name"] for f in FACTORY_LLM_INFOS]
    if provider_name not in allowed_factories:
        return False, f"Provider '{provider_name}' is not allowed"

    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_name)
    if not provider_obj:
        return False, f"Provider '{provider_name}' does not exist"

    TenantModelInstanceService.create_instance(provider_id=provider_obj.id, instance_name=instance_name, api_key="", extra=json.dumps({}))
    return True, "success"


def list_provider_instances(tenant_id: str, provider_id_or_name: str):
    """
    List provider instances for a tenant.

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :return: (success, result_or_error_message)
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"
    provider_id = provider_obj.id
    instance_objs = TenantModelInstanceService.get_all_by_provider_id(provider_id)
    if not instance_objs:
        return True, []
    instances = []
    instance_objs.sort(key=lambda x: x.create_time, reverse=True)
    for instance_obj in instance_objs:
        extra_fields = json.loads(instance_obj.extra) if instance_obj.extra else {}
        instances.append(
            {
                "id": instance_obj.id,
                "instance_name": instance_obj.instance_name,
                "provider_id": provider_id,
                "region": extra_fields.get("region", ""),
                "status": instance_obj.status,
            }
        )

    return True, instances


async def _run_verification(label: str, coro, timeout_seconds: int):
    """
    Run a verification coroutine with timeout and uniform error handling.

    Returns (True, result) on success, or (False, error_message) on failure.
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return True, result
    except asyncio.TimeoutError:
        logging.exception("Timeout accessing %s", label)
        return False, f"\nTimeout accessing {label}."
    except asyncio.CancelledError:
        logging.exception("Verification cancelled for %s", label)
        return False, f"\n{label} verification aborted."
    except Exception as e:
        logging.exception("Fail to access %s", label)
        return False, f"\nFail to access {label}.{str(e)}"


async def verify_api_key(
    provider_id_or_name: str,
    api_key: str | dict,
    base_url: str = None,
    region: str = None,
    model_info: list[dict] = None,
    default_headers: dict[str, str] | None = None,
):
    """
    Verify API key for a provider.

    :param provider_id_or_name: provider/factory ID or name
    :param api_key: API key
    :param base_url: base url
    :param region: region
    :param model_info: model info, [{
        "model_type": ["chat"],  # support multiple
        "model_name": "name",
        "max_tokens": 4096,
        "extra": {
            "field1": "value1",
            "field2": "'value2"
        }
    }]
    :param default_headers: custom HTTP headers the instance adds to every model request;
        the probe sends them so the endpoint is reached exactly as a live request will reach it
    :return: (success, result_or_error_message)
    """
    if not provider_id_or_name:
        return False, "Provider ID or name is required", {}

    # One rule set governs the write path, the read path and this probe. Headers read
    # back from a stored instance have not been through it in this process, and a value
    # carrying CR/LF would inject further headers into the probe request. Nothing to
    # check when no headers are configured, which is also every provider but a
    # gateway-fronted one.
    if default_headers:
        try:
            default_headers = _contract_validate_default_headers(default_headers)
        except ValueError as error:
            return False, str(error), {}

    provider_obj = None
    if provider_id_or_name:
        _, provider_obj = TenantModelProviderService.get_by_id(provider_id_or_name)
    provider_name = provider_obj.provider_name if provider_obj else provider_id_or_name

    base_url = _normalize_provider_base_url(provider_name, base_url)
    api_key = _normalize_provider_api_key(provider_name, api_key)

    if region and region == "intl" and provider_name.lower() == "siliconflow":
        target_factory_name = "siliconflow_intl"
    else:
        target_factory_name = provider_name

    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == target_factory_name]
    if not factory_info:
        return False, f"Provider '{provider_id_or_name}' not found", {}

    if model_info:
        factory_llms = [
            {
                "model_type": _type,
                "llm_name": model.get("model_name", ""),
            }
            for model in model_info
            if model
            for _type in model.get("model_type", [])
        ]
        if not factory_llms:
            return False, f"No valid models found for provider '{provider_id_or_name}'", {}
    else:
        factory_llms = factory_info[0]["llm"]
        if not factory_llms:
            model_base_url = base_url or factory_info[0].get("url", "")
            discovery_error = ""
            try:
                if provider_name in ModelMeta:
                    remote_models = await ModelMeta[provider_name](api_key, model_base_url).get_model_list()
                    if remote_models:
                        factory_llms = [
                            {
                                "model_type": mt,
                                "llm_name": m["name"],
                            }
                            for m in remote_models
                            for mt in m.get("model_types", [])
                        ]
            except Exception as e:
                # Discovery reaches a user-supplied base URL, so it fails for mundane reasons:
                # the host is unreachable from inside the container, TLS is wrong, the port is
                # closed. Reporting only "no models found" sends people hunting for a bad key.
                safe_url = _redact_url(model_base_url)
                reason = _scrub_url_secrets(str(e), model_base_url)
                # logging.exception would write the raw `e` and its traceback, and clients
                # echo the URL they were handed, so the credentials would land in the log
                # even though the caller-visible message is clean. Log the scrubbed reason.
                logging.error("Model discovery failed for provider %s at %s: %s: %s", provider_name, safe_url, type(e).__name__, reason)
                discovery_error = f" Discovery against {safe_url} failed: {reason}"
            if not factory_llms:
                return False, f"No models found for provider '{provider_id_or_name}'.{discovery_error}", {}

    model_verify_result = {}
    # test if api key works
    timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", 10))
    extra = {"provider": provider_name}
    msg = ""
    api_key_str = api_key if isinstance(api_key, str) else json.dumps(api_key)
    # check passed types
    passed_types = set()
    for llm in factory_llms:
        model_types = _factory_model_types(llm)
        any_passed = False
        for mt_value in model_types:
            if mt_value in passed_types:
                continue
            passed = False

            if mt_value == LLMType.EMBEDDING.value:
                if provider_name not in EmbeddingModel:
                    msg += f"\nEmbedding model from {provider_name} is not supported yet."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                label = f"embedding model({llm['llm_name']})"
                try:
                    mdl = _probe_model(EmbeddingModel, provider_name, api_key_str, llm["llm_name"], base_url=base_url, default_headers=default_headers)
                except Exception as e:
                    logging.exception("Fail to init %s", label)
                    msg += f"\nFail to access {label}.{str(e)}"
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                ok, result = await _run_verification(label, asyncio.to_thread(mdl.encode, ["Test if the api key is available"]), timeout_seconds)
                if not ok:
                    msg += result
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                if len(result[0]) == 0:
                    msg += f"\nFail to access {label}."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                passed = True

            elif mt_value == LLMType.CHAT.value:
                if provider_name not in ChatModel:
                    msg += f"\nChat model from {provider_name} is not supported yet."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                label = f"model({provider_name}/{llm['llm_name']})"
                try:
                    mdl = _probe_model(ChatModel, provider_name, api_key_str, llm["llm_name"], base_url=base_url, default_headers=default_headers, **extra)
                except Exception as e:
                    logging.exception("Fail to init %s", label)
                    msg += f"\nFail to access {label}.{str(e)}"
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue

                temperature = 1 if llm["llm_name"] in ("kimi-k3", "kimi-k2.7-code") else 0.9

                async def check_streamly():
                    async for chunk in mdl.async_chat_streamly(
                        None,
                        [{"role": "user", "content": "Hi"}],
                        {"temperature": temperature},
                    ):
                        if chunk and isinstance(chunk, str) and chunk.find("**ERROR**") < 0:
                            return True
                    return False

                ok, result = await _run_verification(label, check_streamly(), timeout_seconds)
                if not ok:
                    msg += result
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                if not result:
                    msg += f"\nFail to access {label}.No valid response received"
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                passed = True

            elif mt_value == LLMType.RERANK.value:
                if provider_name not in RerankModel:
                    msg += f"\nRerank model from {provider_name} is not supported yet."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                mdl = _probe_model(RerankModel, provider_name, api_key_str, llm["llm_name"], base_url=base_url, default_headers=default_headers)
                label = f"model({provider_name}/{llm['llm_name']})"
                ok, result = await _run_verification(label, asyncio.to_thread(mdl.similarity, "What's the weather?", ["Is it sunny today?"]), timeout_seconds)
                if not ok:
                    msg += result
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                arr, tc = result
                if len(arr) == 0 or tc == 0:
                    msg += f"\nFail to access {label}."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                passed = True

            elif mt_value == LLMType.OCR.value:
                if provider_name not in OcrModel:
                    msg += f"\nOCR model from {provider_name} is not supported yet."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                mdl = OcrModel[provider_name](key=api_key_str, model_name=llm["llm_name"], base_url=base_url)
                label = f"model({provider_name}/{llm['llm_name']})"
                ok, result = await _run_verification(label, asyncio.to_thread(mdl.check_available), timeout_seconds)
                if not ok:
                    msg += result
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                ok2, reason = result
                if not ok2:
                    msg += f"\nFail to access {label}.{reason or 'Model not available'}"
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                passed = True

            elif mt_value == LLMType.TTS.value:
                if provider_name not in TTSModel:
                    msg += f"\nTTS model from {provider_name} is not supported yet."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                mdl = TTSModel[provider_name](key=api_key_str, model_name=llm["llm_name"], base_url=base_url)

                def drain_tts():
                    for _ in mdl.tts("Hello~ RAGFlower!"):
                        pass

                label = f"model({provider_name}/{llm['llm_name']})"
                ok, result = await _run_verification(label, asyncio.to_thread(drain_tts), timeout_seconds)
                if not ok:
                    msg += result
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                passed = True

            elif mt_value == LLMType.VISION.value:
                if provider_name not in CvModel:
                    msg += f"\nImage to text model from {provider_name} is not supported yet."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                from rag.utils.base64_image import test_image

                mdl = CvModel[provider_name](key=api_key_str, model_name=llm["llm_name"], base_url=base_url)
                label = f"model({provider_name}/{llm['llm_name']})"
                ok, result = await _run_verification(label, asyncio.to_thread(mdl.describe, test_image), timeout_seconds)
                if not ok:
                    msg += result
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                m, tc = result
                if not tc and m.find("**ERROR**:") >= 0:
                    msg += f"\nFail to access {label}.{m}"
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                passed = True

            elif mt_value == LLMType.ASR.value:
                if provider_name not in Seq2txtModel:
                    msg += f"\nSpeech model from {provider_name} is not supported yet."
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                mdl = Seq2txtModel[provider_name](key=api_key_str, model_name=llm["llm_name"], base_url=base_url)
                label = f"model({provider_name}/{llm['llm_name']})"
                ok, result = await _run_verification(label, asyncio.to_thread(mdl.check_available), timeout_seconds)
                if not ok:
                    msg += result
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                ok2, reason = result
                if not ok2:
                    msg += f"\nFail to access {label}.{reason or 'Model not available'}"
                    model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
                    continue
                passed = True

            if passed:
                logging.debug("passed model %s type=%s", llm["llm_name"], mt_value)
                passed_types.add(mt_value)
                model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.SUCCESS.value
                any_passed = True
                break
            else:
                model_verify_result[llm["llm_name"]] = ModelVerifyStatusEnum.FAIL.value
        if any_passed:
            msg = ""
            break
        else:
            msg = msg or "No model passed verification"

    success = bool(passed_types)
    return success, "success" if success else msg, model_verify_result


def show_provider_instance(tenant_id: str, provider_id_or_name: str, instance_id_or_name: str):
    """
    Show a specific provider instance.

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :param instance_id_or_name: instance ID or name
    :return: (success, result_or_error_message)
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"
    provider_id = provider_obj.id
    instance_obj = None
    if instance_id_or_name:
        _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
    if instance_obj and instance_obj.provider_id != provider_id:
        instance_obj = None
    if not instance_obj:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_id, instance_id_or_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    extra_fields = json.loads(instance_obj.extra) if instance_obj.extra else {}

    return True, {
        "id": instance_obj.id,
        "instance_name": instance_obj.instance_name,
        "provider_id": provider_id,
        "region": extra_fields.get("region", ""),
        "base_url": extra_fields.get("base_url", ""),
        "api_key": _mask_api_key(instance_obj.api_key),
        "status": instance_obj.status,
    }


def drop_provider_instances(tenant_id: str, provider_id_or_name: str, instance_id_or_names: list):
    """
    Drop provider instances.
    for the specified models/instances.

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :param instance_id_or_names: list of instance IDs or names to drop
    :return: (success, result_or_error_message)
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"
    provider_id = provider_obj.id
    not_exist_instances = []
    instance_objs = []
    instance_ids = []
    for instance_id_or_name in instance_id_or_names:
        instance_obj = None
        if instance_id_or_name:
            _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
        if instance_obj and instance_obj.provider_id != provider_id:
            instance_obj = None
        if not instance_obj:
            instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_id, instance_id_or_name)
        if not instance_obj:
            not_exist_instances.append(instance_id_or_name)
            continue
        instance_objs.append(instance_obj)
        instance_ids.append(instance_obj.id)
    if not_exist_instances:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{not_exist_instances}'"

    models = _instance_models(instance_objs)
    blockers = _model_deletion_blockers(tenant_id, provider_obj.provider_name, models)
    if blockers:
        return False, " ".join(blockers + ["Repoint or delete the dependents before deleting the instance."])
    _clear_tenant_default_models(tenant_id, provider_obj.provider_name, models)

    delete_models_by_instance_ids(instance_ids)
    TenantModelInstanceService.delete_by_ids(instance_ids)
    return True, None


def _public_factory_model(llm: dict) -> dict:
    return {
        "name": _factory_llm_name(llm),
        "max_tokens": llm.get("max_tokens", 8192),
        "model_types": _factory_model_types(llm),
        "features": (llm.get("features") if llm.get("features") is not None else ((["is_tools"] if llm.get("is_tools") else []) + (["thinking"] if llm.get("thinking") else []))),
    }


def _merge_nvidia_models(factory_info: dict, remote_models: list[dict]) -> list[dict]:
    static_models = {_factory_llm_name(llm): _public_factory_model(llm) for llm in factory_info.get("llm", [])}
    merged = []
    seen = set()
    for remote in remote_models:
        model_name = str(remote.get("name", "")).strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        model = dict(remote)
        model["name"] = model_name
        if preset := static_models.get(model_name):
            model = {**model, **preset}
        if not model.get("model_types"):
            model["model_types"] = [LLMType.CHAT.value]
        model["max_tokens"] = _to_int(model.get("max_tokens"), 8192)
        model.setdefault("features", [])
        merged.append(model)
    merged.sort(key=lambda model: model["name"])
    return merged


def _set_discovered_model_metadata(extra: dict, model: dict):
    extra["max_tokens"] = _to_int(model.get("max_tokens"), 8192)
    if model.get("max_dimension") is not None:
        extra["max_dimension"] = model["max_dimension"]
    if model.get("dimensions"):
        extra["dimensions"] = model["dimensions"]
    features = model.get("features") or []
    extra["is_tools"] = "is_tools" in features
    extra["thinking"] = "thinking" in features


def _reconcile_nvidia_instance_models(provider_obj, instance_obj, remote_models: list[dict]):
    normalized = []
    seen = set()
    for model in remote_models:
        model_name = str(model.get("name", "")).strip()
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        normalized.append({**model, "name": model_name})
    if not normalized:
        raise ValueError("NVIDIA model discovery returned no usable models")

    with DB.atomic():
        existing_models = TenantModelService.get_models_by_instance_id(instance_obj.id)
        existing_by_name = {model.model_name: model for model in existing_models}

        for model in normalized:
            model_name = model["name"]
            model_types = model.get("model_types") or [LLMType.CHAT.value]
            model_type = model_utils.calculate_model_type(model_types)
            if existing := existing_by_name.pop(model_name, None):
                extra = json.loads(existing.extra or "{}")
                _set_discovered_model_metadata(extra, model)
                TenantModelService.update_model(existing.id, {"model_type": model_type, "extra": json.dumps(extra)})
                continue

            extra = {"verify": ModelVerifyStatusEnum.UNKNOWN.value}
            _set_discovered_model_metadata(extra, model)
            TenantModelService.insert(
                model_name=model_name,
                provider_id=provider_obj.id,
                instance_id=instance_obj.id,
                model_type=model_type,
                status=ActiveStatusEnum.ACTIVE.value,
                extra=json.dumps(extra),
            )

        TenantModelService.delete_by_ids([model.id for model in existing_by_name.values()])


def _get_provider_instance(provider_obj, instance_id_or_name: str):
    instance_obj = None
    if instance_id_or_name:
        _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
    if instance_obj and instance_obj.provider_id != provider_obj.id:
        instance_obj = None
    if not instance_obj:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_id_or_name)
    return instance_obj


async def list_instance_models(tenant_id: str, provider_id_or_name: str, instance_id_or_name: str, supported_only: bool = False):
    """
    List models for a provider instance.

    Follows the Go version's logic:
    - Reads tenant_model table to determine disabled models (records exist = disabled).
    - Lists all models from the LLM dictionary for the provider.
    - Models present in tenant_model table are marked "inactive", others "active".

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :param instance_id_or_name: instance ID or name
    :param supported_only: if True, only list supported models (from LLM dictionary)
    :return: (success, result_or_error_message)
    """
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"

    if supported_only:
        # List all models supported by this provider from the LLM dictionary.
        factory_infos = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_obj.provider_name]
        if not factory_infos:
            return False, f"Provider '{provider_id_or_name}' not found"
        factory_info = factory_infos[0]

        if provider_obj.provider_name == "NVIDIA":
            instance_obj = _get_provider_instance(provider_obj, instance_id_or_name)
            if not instance_obj:
                return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"
            instance_extra = json.loads(instance_obj.extra or "{}")
            base_url = instance_extra.get("base_url") or factory_info.get("url", "")
            remote_models = await ModelMeta["NVIDIA"](instance_obj.api_key, base_url).get_model_list()
            models = _merge_nvidia_models(factory_info, remote_models)
            if not models:
                return False, "NVIDIA model discovery returned no usable models"
            _reconcile_nvidia_instance_models(provider_obj, instance_obj, models)
            return True, models

        llms = factory_info.get("llm", [])
        models = [{"name": llm["llm_name"], "rank": _to_int(llm.get("rank", 500))} for llm in llms]
        models.sort(key=lambda x: (-x["rank"], x["name"]))
        return True, models

    instance_obj = _get_provider_instance(provider_obj, instance_id_or_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    # Build rank mapping from LLM dictionary for the provider
    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_obj.provider_name]
    model_rank_map = {}
    if factory_info:
        for llm in factory_info[0].get("llm", []):
            model_rank_map[llm["llm_name"]] = _to_int(llm.get("rank", 500))

    # Get models
    model_objs = TenantModelService.get_models_by_instance_id(instance_obj.id)
    model_list = []
    for model in model_objs:
        model_extra = json.loads(model.extra)
        model_list.append(
            {
                "name": model.model_name,
                "model_type": model_utils.get_model_type_human(model.model_type),
                "max_tokens": model_extra.get("max_tokens", 8192) if model.extra else 8192,
                "status": model.status,
                "verify": model_extra.get("verify", ModelVerifyStatusEnum.UNKNOWN.value),
                "features": (["is_tools"] if model_extra.get("is_tools") else []) + (["thinking"] if model_extra.get("thinking") else []),
                "rank": model_rank_map.get(model.model_name, 500),
                "extra": model_extra,
            }
        )
    model_list.sort(key=lambda x: (-x["rank"], x["name"]))

    return True, model_list


def update_instance_models(tenant_id: str, provider_id_or_name: str, instance_id_or_name: str, model_names: list, model_types: list):
    if not model_names or not model_types:
        return False, "model_name and model_type are required"

    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"
    instance_obj = None
    if instance_id_or_name:
        _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
    if instance_obj and instance_obj.provider_id != provider_obj.id:
        instance_obj = None
    if not instance_obj:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_id_or_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    model_objs = TenantModelService.get_models_by_instance_id(instance_obj.id)
    not_exist_models = set(model_names) - {model_obj.model_name for model_obj in model_objs}
    if not_exist_models:
        return False, f"Models {not_exist_models} not found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    target_model_type_bin = model_utils.calculate_model_type(model_types)
    to_update = [model_obj.id for model_obj in model_objs if model_obj.model_type != target_model_type_bin and model_obj.model_name in model_names]
    if to_update:
        TenantModelService.batch_update_model_type(to_update, target_model_type_bin)

    return True, "success"


def add_model_to_instance(tenant_id: str, provider_id_or_name: str, instance_id_or_name: str, model_name: str, model_type: str | list[str], max_tokens: int = 8192, extra: dict = None):
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"
    instance_obj = None
    if instance_id_or_name:
        _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
    if instance_obj and instance_obj.provider_id != provider_obj.id:
        instance_obj = None
    if not instance_obj:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_id_or_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"
    model_obj = TenantModelService.get_by_provider_id_and_instance_id_and_model_name(provider_obj.id, instance_obj.id, model_name)
    if model_obj:
        return False, f"Model '{model_name}' already exists for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"
    factory_info = [f for f in FACTORY_LLM_INFOS if f["name"] == provider_obj.provider_name]
    if not factory_info:
        return False, f"Provider '{provider_id_or_name}' not found"
    llms = factory_info[0].get("llm", [])
    if isinstance(model_type, str):
        model_type = [model_type]

    model_type_bin = model_utils.calculate_model_type(model_type)
    extra_fields = {"max_tokens": max_tokens}
    target_model = [llm for llm in llms if llm["llm_name"] == model_name]
    if target_model:
        extra_fields.update({"is_tools": target_model[0].get("is_tools", False)})
        extra_fields.update({"thinking": "thinking" in target_model[0].get("features", [])})
    if extra:
        extra_fields.update(extra)
    TenantModelService.insert(model_name=model_name, provider_id=provider_obj.id, instance_id=instance_obj.id, model_type=model_type_bin, extra=json.dumps(extra_fields))

    return True, "success"


def update_model(tenant_id: str, provider_id_or_name: str, instance_id_or_name: str, model_name: str, update_dict: dict):
    """
    Enable or disable a model for a provider instance.

    - If the model record exists in tenant_model, update its status.
    - If the model record does not exist:
      - status="active": no need to add a record (default is active/enabled).
      - status="inactive": create a record with status="inactive".

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :param instance_id_or_name: instance ID or name
    :param model_name: model name
    :param update_dict:
        status: "active" or "inactive" (ActiveStatusEnum values)
        max_tokens: > 0
    :return: (success, result_or_error_message)
    """
    if update_dict.get("status") and update_dict["status"] not in (ActiveStatusEnum.ACTIVE.value, ActiveStatusEnum.INACTIVE.value):
        return False, f"status must be '{ActiveStatusEnum.ACTIVE.value}' or '{ActiveStatusEnum.INACTIVE.value}'"

    # Check if provider exists for this tenant
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"

    # Check if instance exists
    instance_obj = None
    if instance_id_or_name:
        _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
    if instance_obj and instance_obj.provider_id != provider_obj.id:
        instance_obj = None
    if not instance_obj:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_id_or_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    model_obj = TenantModelService.get_by_provider_id_and_instance_id_and_model_name(provider_obj.id, instance_obj.id, model_name)
    if not model_obj:
        return False, f"Model '{model_name}' not added for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    to_update = {}
    if "status" in update_dict and update_dict.get("status") != model_obj.status:
        to_update.update({"status": update_dict["status"]})
    new_extra = update_dict.get("extra", {})
    if "max_tokens" in update_dict:
        new_extra.update({"max_tokens": update_dict["max_tokens"]})
    if "verify" in update_dict:
        new_extra.update({"verify": update_dict["verify"]})
    if new_extra:
        db_extra = json.loads(model_obj.extra)
        db_extra.update(**new_extra)
        to_update.update({"extra": json.dumps(db_extra)})
    if "model_type" in update_dict:
        target_model_type = model_utils.calculate_model_type(update_dict["model_type"])
        if target_model_type != model_obj.model_type:
            to_update.update({"model_type": target_model_type})

    if to_update:
        TenantModelService.update_model(model_obj.id, to_update)

    return True, "success"


async def delete_models_from_instance(tenant_id: str, provider_id_or_name: str, instance_id_or_name: str, model_name: list[str]):
    """
    Delete models from instance.

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :param instance_id_or_name: instance ID or name
    :param model_name: list of model name
    """
    # Check if provider exists for this tenant (by ID first, then by name)
    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"

    # Check if instance exists (by ID first, then by name)
    instance_obj = None
    if instance_id_or_name:
        _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
    if instance_obj and instance_obj.provider_id != provider_obj.id:
        instance_obj = None
    if not instance_obj:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_id_or_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    model_objs = TenantModelService.get_models_by_instance_id(instance_obj.id)
    not_exist_models = set(model_name) - {model_obj.model_name for model_obj in model_objs}
    if not_exist_models:
        return False, f"Models {not_exist_models} not found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    models_to_delete = [(instance_obj.instance_name, model_obj) for model_obj in model_objs if model_obj.model_name in model_name]
    blockers = _model_deletion_blockers(tenant_id, provider_obj.provider_name, models_to_delete)
    if blockers:
        return False, " ".join(blockers + ["Repoint or delete the dependents before deleting the model."])
    _clear_tenant_default_models(tenant_id, provider_obj.provider_name, models_to_delete)

    TenantModelService.delete_by_ids([model_obj.id for _instance_name, model_obj in models_to_delete])

    return True, "success"


async def chat_to_model(tenant_id: str, provider_id_or_name: str, instance_id_or_name: str, model_name: str, message: str, stream: bool = False, thinking: bool = False):
    """
    Chat to a model.

    :param tenant_id: tenant ID
    :param provider_id_or_name: provider/factory ID or name
    :param instance_id_or_name: instance ID or name
    :param model_name: model name
    :param message: chat message
    :param stream: whether to stream the response
    :param thinking: whether to enable thinking/reasoning
    :return: (success, result_or_error_message)
    """
    from api.db.services.llm_service import LLMBundle

    provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_id(tenant_id, provider_id_or_name)
    if not provider_obj:
        provider_obj = TenantModelProviderService.get_by_tenant_id_and_provider_name(tenant_id, provider_id_or_name)
    if not provider_obj:
        return False, f"No provider found for provider '{provider_id_or_name}'"

    instance_obj = None
    if instance_id_or_name:
        _, instance_obj = TenantModelInstanceService.get_by_id(instance_id_or_name)
    if instance_obj and instance_obj.provider_id != provider_obj.id:
        instance_obj = None
    if not instance_obj:
        instance_obj = TenantModelInstanceService.get_by_provider_id_and_instance_name(provider_obj.id, instance_id_or_name)
    if not instance_obj:
        return False, f"No instance found for provider '{provider_id_or_name}' and instance '{instance_id_or_name}'"

    provider_name = provider_obj.provider_name
    instance_name = instance_obj.instance_name

    # Get model config
    composite_name = f"{model_name}@{instance_name}@{provider_name}"
    try:
        model_config = resolve_model_config(tenant_id, LLMType.CHAT, composite_name)
    except LookupError:
        return False, f"Model '{composite_name}' not authorized"

    if not model_config:
        return False, f"Model '{composite_name}' not found"

    llm = LLMBundle(tenant_id, model_config)

    if stream:
        return True, {"type": "stream", "llm": llm, "model_config": model_config}

    # Non-streaming chat
    try:
        response = await llm.async_chat(
            None,
            [{"role": "user", "content": message}],
            {"temperature": 0.9},
        )
        result = {
            "answer": response,
            "reasoning_content": "",
        }
        return True, result
    except Exception as e:
        logging.exception(f"Chat to model failed: {e}")
        return False, str(e)
