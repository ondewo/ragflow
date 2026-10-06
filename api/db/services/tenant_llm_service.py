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
import os
import functools
import inspect
import json
import logging
import re
from typing import Any
from peewee import IntegrityError
from langfuse import Langfuse
from common import settings
from common.constants import MINERU_DEFAULT_CONFIG, MINERU_ENV_KEYS, OPENDATALOADER_DEFAULT_CONFIG, OPENDATALOADER_ENV_KEYS, PADDLEOCR_DEFAULT_CONFIG, PADDLEOCR_ENV_KEYS, LLMType
from api.db.db_models import DB, LLMFactories, TenantLLM
from api.db.services.common_service import CommonService
from api.db.services.langfuse_service import TenantLangfuseService
from api.db.services.user_service import TenantService


# RFC 7230 field-name token characters and the printable subset allowed in a
# field value (plus horizontal tab). Custom model headers come from the tenant
# API, so a value carrying CR/LF would let a caller inject further request
# headers or a body into every request made with that model.
# Applied with `fullmatch`, so the pattern needs no anchors: a `$`-anchored
# `match` also succeeds just before a trailing newline, and would accept the
# very character these patterns exist to reject at the end of a name or value.
# A field-name must have at least one character; a field-value may be empty,
# which the write path in the provider service also accepts -- rejecting it
# here would leave a stored config that fails every request made with it.
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_HEADER_VALUE_RE = re.compile(r"[\t\x20-\x7e]*")
# Headers the HTTP transport computes itself; overriding them desynchronizes the
# request from what is actually sent.
_HEADER_FORBIDDEN = frozenset({"host", "content-length"})
_HEADER_MAX_COUNT = 32
_HEADER_NAME_MAX_LEN = 128
_HEADER_VALUE_MAX_LEN = 4096


def validate_default_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """
    Validate custom per-model HTTP headers and return them unchanged.

    Args:
        headers (dict[str, str] | None): Header name/value pairs as stored in the model instance config, or None.

    Returns:
        dict[str, str] | None: The very same object that was passed in, once every entry is known to be legal.

    Raises:
        ValueError: If the mapping, a header name or a header value is not usable as an HTTP header.
    """
    if headers is None:
        return None
    if not isinstance(headers, dict):
        raise ValueError("default_headers must be an object")
    if len(headers) > _HEADER_MAX_COUNT:
        raise ValueError(f"default_headers must contain at most {_HEADER_MAX_COUNT} entries")

    seen: set[str] = set()
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("default_headers names and values must be strings")
        if len(name) > _HEADER_NAME_MAX_LEN:
            raise ValueError(f"default_headers name length must be at most {_HEADER_NAME_MAX_LEN} characters")
        if len(value) > _HEADER_VALUE_MAX_LEN:
            raise ValueError(f"default_headers value length must be at most {_HEADER_VALUE_MAX_LEN} characters")
        if not _HEADER_NAME_RE.fullmatch(name):
            raise ValueError(f"default_headers name {name!r} is not a legal HTTP field name")
        if not _HEADER_VALUE_RE.fullmatch(value):
            raise ValueError(f"default_headers value of {name!r} is not a legal HTTP field value")
        lowered = name.lower()
        if lowered in _HEADER_FORBIDDEN:
            raise ValueError(f"default_headers name {name!r} is managed by the HTTP transport and cannot be overridden")
        if lowered in seen:
            raise ValueError(f"default_headers contains {name!r} more than once (header names are case-insensitive)")
        seen.add(lowered)

    return headers


# The header-capability probe identifies classes by their module and qualified name
# rather than by importing them. A lazy import here is resolved against whatever
# `sys.modules` currently holds for `rag.llm.chat_model`, so a caller that has
# replaced that module turns the probe into an ImportError -- and because the
# probe sits on the model-construction path, that surfaces as a failed model
# verification rather than as the missing import it is. Matching on names also
# degrades safely when a provider class is renamed or dropped upstream: the entry
# simply stops matching, instead of breaking every model construction.
_CHAT_MODEL_MODULE = "rag.llm.chat_model"

# Classes that call `chat_model.Base.__init__` -- which does consume
# `default_headers` -- and then overwrite `self.client` with a transport the
# mapping was never handed to: another SDK entirely (`mistralai`, `replicate`,
# `qianfan`, `google.genai`/`AnthropicVertex`, `jina`), a second bare `OpenAI(...)`
# built without the headers (`LocalAIChat`, `LmStudioChat`), or, for `MWSChat`, a
# `requests`/`aiohttp` path driven by its own `self.headers` dict. Inheriting from a
# header-aware base is therefore not sufficient to conclude the headers are sent.
_TRANSPORT_REPLACING_CLIENT_NAMES: frozenset[tuple[str, str]] = frozenset(
    (_CHAT_MODEL_MODULE, qualname)
    for qualname in (
        "BaiduYiyanChat",
        "GoogleChat",
        "LmStudioChat",
        "LocalAIChat",
        "LocalLLM",
        "MistralChat",
        "MWSChat",
        "ReplicateChat",
    )
)

# Constructors that consume `default_headers` out of `**kwargs` instead of naming it.
_HEADER_AWARE_KWARGS_CONSTRUCTOR_NAMES: frozenset[tuple[str, str]] = frozenset({(_CHAT_MODEL_MODULE, "Base")})


def _class_identity(klass: type[Any]) -> tuple[str, str]:
    """
    Return the (module, qualified name) pair that identifies `klass` without importing it.

    Args:
        klass (type[Any]): The class to identify.

    Returns:
        tuple[str, str]: Its defining module and qualified name.
    """
    return (getattr(klass, "__module__", ""), getattr(klass, "__qualname__", getattr(klass, "__name__", "")))


def _resolve_class_names(class_names: frozenset[tuple[str, str]]) -> frozenset[type[Any]]:
    """
    Resolve (module, qualified name) pairs to classes, skipping any the module no longer defines.

    Args:
        class_names (frozenset[tuple[str, str]]): The pairs to resolve.

    Returns:
        frozenset[type[Any]]: The classes that resolved; a name the module does not define is skipped.
    """
    import importlib

    resolved: set[type[Any]] = set()
    for module_name, qualname in class_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        klass = getattr(module, qualname, None)
        if isinstance(klass, type):
            resolved.add(klass)

    return frozenset(resolved)


@functools.lru_cache(maxsize=1)
def _header_aware_kwargs_constructors() -> frozenset[type[Any]]:
    """
    Return the client constructors that consume ``default_headers`` out of ``**kwargs`` instead of naming it.

    Returns:
        frozenset[type[Any]]: The classes whose own ``__init__`` pops the key and hands it to its HTTP client.
    """
    # `rag.llm.chat_model.Base.__init__` pops `default_headers` and passes it to
    # the OpenAI and AsyncOpenAI clients it builds, so all of its subclasses send
    # the headers even though none of them names the keyword. Imported here rather
    # than at module scope because `rag.llm` is itself only imported lazily, inside
    # `TenantLLMService.model_instance`.
    return _resolve_class_names(_HEADER_AWARE_KWARGS_CONSTRUCTOR_NAMES)


@functools.lru_cache(maxsize=1)
def _transport_replacing_clients() -> frozenset[type[Any]]:
    """
    Return the client classes that throw away the header-carrying HTTP client their base built for them.

    Returns:
        frozenset[type[Any]]: The classes whose own ``__init__`` installs a transport that the configured headers never reach.
    """
    # Each of these calls `chat_model.Base.__init__` -- which does consume
    # `default_headers` -- and then overwrites `self.client` with a transport of
    # its own that the mapping was never handed to: another SDK entirely
    # (`mistralai`, `replicate`, `qianfan`, `google.genai`, `AnthropicVertex`,
    # `jina`), a second bare `OpenAI(...)` built without the headers
    # (`LocalAIChat`, `LmStudioChat`), or, for `MWSChat`, a `requests`/`aiohttp`
    # path driven by its own `self.headers` dict that it fills before calling up.
    # Inheriting from a header-aware base is therefore not enough to conclude that
    # a client sends the headers, and these have to be named: an accepted header
    # that never reaches the gateway is the failure mode the operator cannot see.
    return _resolve_class_names(_TRANSPORT_REPLACING_CLIENT_NAMES)


def client_sends_default_headers(client_cls: type[Any]) -> bool:
    """
    Report whether constructing `client_cls` with a ``default_headers`` keyword puts those headers on the wire.

    The verdict comes from the chain of ``__init__`` implementations that a call to `client_cls` actually runs,
    walked from the concrete class upwards. A class listed by `_transport_replacing_clients` discards the client
    that would have carried the headers, so it ends the walk immediately. A constructor that names
    ``default_headers`` consumes it; one listed by `_header_aware_kwargs_constructors` consumes it out of
    ``**kwargs``; and a constructor that collects no ``**kwargs`` at all ends the walk, because from there the
    keyword can no longer travel to a base that would. A constructor whose signature cannot be read ends the walk
    the same way.

    Collecting ``**kwargs`` is deliberately not read as support. `rag.llm.chat_model.LiteLLMBase` (33 chat
    factories) and a dozen embedding and rerank classes absorb unknown keywords and never send them, and a
    header the operator believes is in effect but that never reaches the gateway is worse than one that was
    skipped out loud.

    Args:
        client_cls (type[Any]): The client class a factory name resolves to in one of the `rag.llm` registries.

    Returns:
        bool: True when a configured header mapping would reach this client's HTTP layer.
    """
    for klass in client_cls.__mro__:
        identity = _class_identity(klass)
        if identity in _TRANSPORT_REPLACING_CLIENT_NAMES:
            return False

        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        if identity in _HEADER_AWARE_KWARGS_CONSTRUCTOR_NAMES:
            return True

        try:
            parameters = inspect.signature(init).parameters
        except (TypeError, ValueError):
            # A constructor implemented in C has no inspectable signature, and
            # guessing is the trap this function exists to avoid, so end the walk.
            return False

        if "default_headers" in parameters:
            return True
        if not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return False

    return False


def default_headers_kwarg(client_cls: type[Any], default_headers: dict[str, str] | None, factory_name: str) -> dict[str, dict[str, str]]:
    """
    Build the keyword mapping that carries `default_headers` into `client_cls`, warning and dropping it if it cannot.

    Most of the registered provider clients take no ``default_headers`` argument, and several absorb it into
    ``**kwargs`` without ever sending it. Configuring headers on such a model is an operator mistake, not a
    reason to fail every request the model serves, so the headers are skipped and the factory is named in the log.

    Args:
        client_cls (type[Any]): The client class the factory name resolved to.
        default_headers (dict[str, str] | None): The validated headers configured for this model instance, if any.
        factory_name (str): The provider factory the model instance is registered under, for the log line.

    Returns:
        dict[str, dict[str, str]]: ``{"default_headers": ...}`` when the client sends them, otherwise an empty mapping.
    """
    if not default_headers:
        return {}
    if not client_sends_default_headers(client_cls):
        logging.warning(
            "Model instance of factory %r configures %d custom HTTP header(s), but its client %s does not send them. The headers are ignored for this model.",
            factory_name,
            len(default_headers),
            client_cls.__name__,
        )
        return {}

    return {"default_headers": default_headers}


class LLMFactoriesService(CommonService):
    model = LLMFactories


class TenantLLMService(CommonService):
    model = TenantLLM

    @staticmethod
    def _decode_api_key_config(raw_api_key: str) -> tuple[str, bool | None, str | None]:
        if not raw_api_key:
            return raw_api_key, None, None

        try:
            parsed = json.loads(raw_api_key)
        except Exception:
            return raw_api_key, None, None

        if not isinstance(parsed, dict):
            return raw_api_key, None, None

        is_tools = bool(parsed["is_tools"]) if "is_tools" in parsed else None
        if set(parsed.keys()) <= {"api_key", "is_tools"}:
            return parsed.get("api_key", ""), is_tools, None

        return parsed.get("api_key", raw_api_key), is_tools, raw_api_key

    @staticmethod
    def _encode_api_key_config(raw_api_key: str, is_tools: bool | None) -> str:
        if is_tools is None:
            return raw_api_key

        try:
            parsed = json.loads(raw_api_key or "{}")
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            payload = dict(parsed)
            payload["is_tools"] = bool(is_tools)
            return json.dumps(payload)

        return json.dumps({"api_key": raw_api_key or "", "is_tools": bool(is_tools)})

    @classmethod
    @DB.connection_context()
    def get_api_key(cls, tenant_id, model_name, model_type=None):
        mdlnm, fid = TenantLLMService.split_model_name_and_factory(model_name)
        model_type_val = model_type.value if hasattr(model_type, "value") else model_type
        query_kwargs = {"tenant_id": tenant_id, "llm_name": mdlnm}
        if model_type_val is not None:
            query_kwargs["model_type"] = model_type_val
        if not fid:
            objs = cls.query(**query_kwargs)
        else:
            objs = cls.query(**query_kwargs, llm_factory=fid)

        if (not objs) and fid:
            if fid == "LocalAI":
                mdlnm += "___LocalAI"
            elif fid == "HuggingFace":
                mdlnm += "___HuggingFace"
            elif fid == "OpenAI-API-Compatible":
                mdlnm += "___OpenAI-API"
            elif fid == "VLLM":
                mdlnm += "___VLLM"
            query_kwargs["llm_name"] = mdlnm
            objs = cls.query(**query_kwargs, llm_factory=fid)
        if not objs:
            return None
        return objs[0]

    @classmethod
    @DB.connection_context()
    def get_my_llms(cls, tenant_id):
        fields = [cls.model.id, cls.model.llm_factory, LLMFactories.logo, LLMFactories.tags, cls.model.model_type, cls.model.llm_name, cls.model.used_tokens, cls.model.status]
        objs = cls.model.select(*fields).join(LLMFactories, on=(cls.model.llm_factory == LLMFactories.name)).where(cls.model.tenant_id == tenant_id, ~cls.model.api_key.is_null()).dicts()

        return list(objs)

    @staticmethod
    def split_model_name_and_factory(model_name):
        arr = model_name.split("@")
        if len(arr) < 2:
            return model_name, None
        if len(arr) > 2:
            return "@".join(arr[0:-1]), arr[-1]

        # model name must be xxx@yyy
        try:
            model_factories = settings.FACTORY_LLM_INFOS
            model_providers = set([f["name"] for f in model_factories])
            if arr[-1] not in model_providers:
                return model_name, None
            return arr[0], arr[-1]
        except Exception as e:
            logging.exception(f"TenantLLMService.split_model_name_and_factory got exception: {e}")
        return model_name, None

    @classmethod
    @DB.connection_context()
    def get_model_config(cls, tenant_id, llm_type, llm_name=None):
        from api.db.services.llm_service import LLMService

        e, tenant = TenantService.get_by_id(tenant_id)
        if not e:
            raise LookupError("Tenant not found")

        if llm_type == LLMType.EMBEDDING.value:
            mdlnm = tenant.embd_id if not llm_name else llm_name
        elif llm_type == LLMType.ASR.value:
            mdlnm = tenant.asr_id if not llm_name else llm_name
        elif llm_type == LLMType.VISION.value:
            mdlnm = tenant.img2txt_id if not llm_name else llm_name
        elif llm_type == LLMType.CHAT.value:
            mdlnm = tenant.llm_id if not llm_name else llm_name
        elif llm_type == LLMType.RERANK:
            mdlnm = tenant.rerank_id if not llm_name else llm_name
        elif llm_type == LLMType.TTS:
            mdlnm = tenant.tts_id if not llm_name else llm_name
        elif llm_type == LLMType.OCR:
            if not llm_name:
                raise LookupError("OCR model name is required")
            mdlnm = llm_name
        else:
            assert False, "LLM type error"

        model_config = cls.get_api_key(tenant_id, mdlnm, llm_type)
        mdlnm, fid = TenantLLMService.split_model_name_and_factory(mdlnm)
        if not model_config:  # for some cases seems fid mismatch
            model_config = cls.get_api_key(tenant_id, mdlnm, llm_type)
        if model_config:
            model_config = model_config.to_dict()
            api_key, is_tools, api_key_payload = cls._decode_api_key_config(model_config.get("api_key", ""))
            model_config["api_key"] = api_key
            if api_key_payload is not None:
                model_config["api_key_payload"] = api_key_payload
            if is_tools is not None:
                model_config["is_tools"] = is_tools
        elif llm_type == LLMType.EMBEDDING and fid == "Builtin" and "tei-" in os.getenv("COMPOSE_PROFILES", "") and mdlnm == os.getenv("TEI_MODEL", ""):
            embedding_cfg = settings.EMBEDDING_CFG
            model_config = {"llm_factory": "Builtin", "api_key": embedding_cfg["api_key"], "llm_name": mdlnm, "api_base": embedding_cfg["base_url"]}
        else:
            raise LookupError(f"Model({mdlnm}@{fid}) not authorized")

        llm = LLMService.query(llm_name=mdlnm) if not fid else LLMService.query(llm_name=mdlnm, fid=fid)
        if not llm and fid:  # for some cases seems fid mismatch
            llm = LLMService.query(llm_name=mdlnm)
        if "is_tools" not in model_config and llm:
            model_config["is_tools"] = llm[0].is_tools
        return model_config

    @classmethod
    @DB.connection_context()
    def model_instance(cls, model_config: dict, lang="Chinese", **kwargs):
        if not model_config:
            raise LookupError("Model config is required")
        from rag.llm import ChatModel, CvModel, EmbeddingModel, OcrModel, RerankModel, Seq2txtModel, TTSModel

        kwargs.update({"provider": model_config["llm_factory"]})
        api_key = model_config.get("api_key_payload", model_config["api_key"])
        # Custom HTTP headers are opt-in: when the model instance configures none,
        # `default_headers_kwarg` returns an empty mapping, so the many provider
        # classes that take no `default_headers` argument are constructed exactly as
        # before. An explicit caller kwarg wins, and popping it keeps the chat branch
        # below -- the one that forwards **kwargs -- from passing the keyword twice.
        default_headers = validate_default_headers(kwargs.pop("default_headers", None) or model_config.get("default_headers"))
        if model_config["model_type"] == LLMType.EMBEDDING.value:
            if model_config["llm_factory"] not in EmbeddingModel:
                logging.error("Factory not in embedding model. Supported factories: %s", list(EmbeddingModel.keys()))
                return None
            embedding_cls = EmbeddingModel[model_config["llm_factory"]]
            headers_kwarg = default_headers_kwarg(client_cls=embedding_cls, default_headers=default_headers, factory_name=model_config["llm_factory"])
            return embedding_cls(api_key, model_config["llm_name"], base_url=model_config["api_base"], **headers_kwarg)

        elif model_config["model_type"] == LLMType.RERANK.value:
            if model_config["llm_factory"] not in RerankModel:
                logging.error("Factory not in rerank model. Supported factories: %s", list(RerankModel.keys()))
                return None
            rerank_cls = RerankModel[model_config["llm_factory"]]
            headers_kwarg = default_headers_kwarg(client_cls=rerank_cls, default_headers=default_headers, factory_name=model_config["llm_factory"])
            return rerank_cls(api_key, model_config["llm_name"], base_url=model_config["api_base"], max_token=model_config.get("max_tokens"), **headers_kwarg)

        elif model_config["model_type"] == LLMType.VISION.value:
            if model_config["llm_factory"] not in CvModel:
                logging.error("Factory not in cv model. Supported factories: %s", list(CvModel.keys()))
                return None
            return CvModel[model_config["llm_factory"]](api_key, model_config["llm_name"], lang, base_url=model_config["api_base"], **kwargs)

        elif model_config["model_type"] == LLMType.CHAT.value:
            if model_config["llm_factory"] not in ChatModel:
                logging.error("Factory not in chat model. Supported factories: %s", list(ChatModel.keys()))
                return None
            chat_cls = ChatModel[model_config["llm_factory"]]
            headers_kwarg = default_headers_kwarg(client_cls=chat_cls, default_headers=default_headers, factory_name=model_config["llm_factory"])
            return chat_cls(api_key, model_config["llm_name"], base_url=model_config["api_base"], **kwargs, **headers_kwarg)

        elif model_config["model_type"] == LLMType.ASR.value:
            if model_config["llm_factory"] not in Seq2txtModel:
                logging.error("Factory not in asr model. Supported factories: %s", list(Seq2txtModel.keys()))
                return None
            return Seq2txtModel[model_config["llm_factory"]](key=api_key, model_name=model_config["llm_name"], lang=lang, base_url=model_config["api_base"])
        elif model_config["model_type"] == LLMType.TTS.value:
            if model_config["llm_factory"] not in TTSModel:
                logging.error("Factory not in tts model. Supported factories: %s", list(TTSModel.keys()))
                return None
            return TTSModel[model_config["llm_factory"]](
                api_key,
                model_config["llm_name"],
                base_url=model_config["api_base"],
            )

        elif model_config["model_type"] == LLMType.OCR.value:
            if model_config["llm_factory"] not in OcrModel:
                logging.error("Factory not in ocr model. Supported factories: %s", list(OcrModel.keys()))
                return None
            return OcrModel[model_config["llm_factory"]](
                key=api_key,
                model_name=model_config["llm_name"],
                base_url=model_config.get("api_base", ""),
                **kwargs,
            )

        return None

    @classmethod
    @DB.connection_context()
    def increase_usage(cls, tenant_id, llm_type, used_tokens, llm_name=None):
        e, tenant = TenantService.get_by_id(tenant_id)
        if not e:
            logging.error(f"Tenant not found: {tenant_id}")
            return 0

        llm_map = {
            LLMType.EMBEDDING.value: tenant.embd_id if not llm_name else llm_name,
            LLMType.ASR.value: tenant.asr_id,
            LLMType.VISION.value: tenant.img2txt_id,
            LLMType.CHAT.value: tenant.llm_id if not llm_name else llm_name,
            LLMType.RERANK.value: tenant.rerank_id if not llm_name else llm_name,
            LLMType.TTS.value: tenant.tts_id if not llm_name else llm_name,
            LLMType.OCR.value: llm_name,
        }

        mdlnm = llm_map.get(llm_type)
        if mdlnm is None:
            logging.error(f"LLM type error: {llm_type}")
            return 0

        llm_name, llm_factory = TenantLLMService.split_model_name_and_factory(mdlnm)

        try:
            num = (
                cls.model.update(used_tokens=cls.model.used_tokens + used_tokens)
                .where(cls.model.tenant_id == tenant_id, cls.model.llm_name == llm_name, cls.model.llm_factory == llm_factory if llm_factory else True)
                .execute()
            )
        except Exception:
            logging.exception("TenantLLMService.increase_usage got exception,Failed to update used_tokens for tenant_id=%s, llm_name=%s", tenant_id, llm_name)
            return 0

        return num

    @classmethod
    @DB.connection_context()
    def increase_usage_by_id(cls, tenant_model_id: int, used_tokens: int):
        try:
            update_cnt = cls.model.update(used_tokens=cls.model.used_tokens + used_tokens).where(cls.model.id == tenant_model_id).execute()
        except Exception as e:
            logging.exception(f"TenantLLMService.increase_usage got exception {e}, Failed to update used_tokens for tenant_model_id {tenant_model_id}")
            return 0
        return update_cnt

    @classmethod
    @DB.connection_context()
    def get_openai_models(cls):
        objs = cls.model.select().where((cls.model.llm_factory == "OpenAI"), ~(cls.model.llm_name == "text-embedding-3-small"), ~(cls.model.llm_name == "text-embedding-3-large")).dicts()
        return list(objs)

    @classmethod
    def _collect_mineru_env_config(cls) -> dict | None:
        cfg = MINERU_DEFAULT_CONFIG
        found = False
        for key in MINERU_ENV_KEYS:
            val = os.environ.get(key)
            if val:
                found = True
                cfg[key] = val
        return cfg if found else None

    @classmethod
    @DB.connection_context()
    def ensure_mineru_from_env(cls, tenant_id: str) -> str | None:
        """
        Ensure a MinerU OCR model exists for the tenant if env variables are present.
        Return the existing or newly created llm_name, or None if env not set.
        """
        cfg = cls._collect_mineru_env_config()
        if not cfg:
            return None

        saved_mineru_models = cls.query(tenant_id=tenant_id, llm_factory="MinerU", model_type=LLMType.OCR.value)

        def _parse_api_key(raw: str) -> dict:
            try:
                return json.loads(raw or "{}")
            except Exception:
                return {}

        for item in saved_mineru_models:
            api_cfg = _parse_api_key(item.api_key)
            normalized = {k: api_cfg.get(k, MINERU_DEFAULT_CONFIG.get(k)) for k in MINERU_ENV_KEYS}
            if normalized == cfg:
                return item.llm_name

        used_names = {item.llm_name for item in saved_mineru_models}
        idx = 1
        base_name = "mineru-from-env"
        while True:
            candidate = f"{base_name}-{idx}"
            if candidate in used_names:
                idx += 1
                continue

            try:
                cls.save(
                    tenant_id=tenant_id,
                    llm_factory="MinerU",
                    llm_name=candidate,
                    model_type=LLMType.OCR.value,
                    api_key=json.dumps(cfg),
                    api_base="",
                    max_tokens=0,
                )
                return candidate
            except IntegrityError:
                logging.warning("MinerU env model %s already exists for tenant %s, retry with next name", candidate, tenant_id)
                used_names.add(candidate)
                idx += 1
                continue

    @classmethod
    def _collect_paddleocr_env_config(cls) -> dict | None:
        cfg = PADDLEOCR_DEFAULT_CONFIG
        found = False
        for key in PADDLEOCR_ENV_KEYS:
            val = os.environ.get(key)
            if val:
                found = True
                cfg[key] = val
        return cfg if found else None

    @classmethod
    @DB.connection_context()
    def ensure_paddleocr_from_env(cls, tenant_id: str) -> str | None:
        """
        Ensure a PaddleOCR model exists for the tenant if env variables are present.
        Return the existing or newly created llm_name, or None if env not set.
        """
        cfg = cls._collect_paddleocr_env_config()
        if not cfg:
            return None

        saved_paddleocr_models = cls.query(tenant_id=tenant_id, llm_factory="PaddleOCR", model_type=LLMType.OCR.value)

        def _parse_api_key(raw: str) -> dict:
            try:
                return json.loads(raw or "{}")
            except Exception:
                return {}

        for item in saved_paddleocr_models:
            api_cfg = _parse_api_key(item.api_key)
            normalized = {k: api_cfg.get(k, PADDLEOCR_DEFAULT_CONFIG.get(k)) for k in PADDLEOCR_ENV_KEYS}
            if normalized == cfg:
                return item.llm_name

        used_names = {item.llm_name for item in saved_paddleocr_models}
        idx = 1
        base_name = "paddleocr-from-env"
        while True:
            candidate = f"{base_name}-{idx}"
            if candidate in used_names:
                idx += 1
                continue

            try:
                cls.save(
                    tenant_id=tenant_id,
                    llm_factory="PaddleOCR",
                    llm_name=candidate,
                    model_type=LLMType.OCR.value,
                    api_key=json.dumps(cfg),
                    api_base="",
                    max_tokens=0,
                )
                return candidate
            except IntegrityError:
                logging.warning("PaddleOCR env model %s already exists for tenant %s, retry with next name", candidate, tenant_id)
                used_names.add(candidate)
                idx += 1
                continue

    @classmethod
    def _collect_opendataloader_env_config(cls) -> dict | None:
        cfg = dict(OPENDATALOADER_DEFAULT_CONFIG)
        found = False
        for key in OPENDATALOADER_ENV_KEYS:
            val = os.environ.get(key)
            if val:
                found = True
                cfg[key] = val
        return cfg if found else None

    @classmethod
    @DB.connection_context()
    def ensure_opendataloader_from_env(cls, tenant_id: str) -> str | None:
        """
        Ensure an OpenDataLoader OCR model exists for the tenant if env variables are present.
        Return the existing or newly created llm_name, or None if env not set.
        """
        cfg = cls._collect_opendataloader_env_config()
        if not cfg:
            return None

        saved_models = cls.query(tenant_id=tenant_id, llm_factory="OpenDataLoader", model_type=LLMType.OCR.value)

        def _parse_api_key(raw: str) -> dict:
            try:
                return json.loads(raw or "{}")
            except Exception:
                return {}

        for item in saved_models:
            api_cfg = _parse_api_key(item.api_key)
            normalized = {k: api_cfg.get(k, OPENDATALOADER_DEFAULT_CONFIG.get(k)) for k in OPENDATALOADER_ENV_KEYS}
            if normalized == cfg:
                return item.llm_name

        used_names = {item.llm_name for item in saved_models}
        idx = 1
        base_name = "opendataloader-from-env"
        while True:
            candidate = f"{base_name}-{idx}"
            if candidate in used_names:
                idx += 1
                continue
            try:
                cls.save(
                    tenant_id=tenant_id,
                    llm_factory="OpenDataLoader",
                    llm_name=candidate,
                    model_type=LLMType.OCR.value,
                    api_key=json.dumps(cfg),
                    api_base="",
                    max_tokens=0,
                )
                return candidate
            except IntegrityError:
                logging.warning("OpenDataLoader env model %s already exists for tenant %s, retry with next name", candidate, tenant_id)
                used_names.add(candidate)
                idx += 1
                continue

    @classmethod
    @DB.connection_context()
    def delete_by_tenant_id(cls, tenant_id):
        return cls.model.delete().where(cls.model.tenant_id == tenant_id).execute()

    @staticmethod
    def llm_id2llm_type(llm_id: str) -> str | None:
        from api.db.services.llm_service import LLMService

        llm_id, *_ = TenantLLMService.split_model_name_and_factory(llm_id)
        llm_factories = settings.FACTORY_LLM_INFOS
        for llm_factory in llm_factories:
            for llm in llm_factory["llm"]:
                if llm_id == llm["llm_name"]:
                    return llm["model_type"].split(",")[-1]

        for llm in LLMService.query(llm_name=llm_id):
            return llm.model_type

        llm = TenantLLMService.get_or_none(llm_name=llm_id)
        if llm:
            return llm.model_type
        for llm in TenantLLMService.query(llm_name=llm_id):
            return llm.model_type
        return None


class LLM4Tenant:
    def __init__(self, tenant_id: str, model_config: dict, lang="Chinese", **kwargs):
        self.trace_context = kwargs.pop("trace_context", None) or {}
        self.langfuse_session_id = kwargs.pop("langfuse_session_id", None)
        self.tenant_id = tenant_id
        self.lang = lang
        self.llm_name = model_config["llm_name"]
        self.model_config = model_config
        self.mdl = TenantLLMService.model_instance(model_config, lang=lang, **kwargs)
        assert self.mdl, "Can't find model for {}/{}/{}".format(tenant_id, model_config["model_type"], model_config["llm_name"])
        self.max_length = model_config.get("max_tokens") or 8192

        self.is_tools = model_config.get("is_tools", False)
        self.verbose_tool_use = kwargs.get("verbose_tool_use")

        langfuse_keys = TenantLangfuseService.filter_by_tenant(tenant_id=tenant_id)
        self.langfuse = None
        if langfuse_keys:
            langfuse = Langfuse(public_key=langfuse_keys.public_key, secret_key=langfuse_keys.secret_key, host=langfuse_keys.host)
            try:
                if langfuse.auth_check():
                    self.langfuse = langfuse
                    if not self.trace_context:
                        trace_id = self.langfuse.create_trace_id()
                        self.trace_context = {"trace_id": trace_id}
            except Exception:
                # Skip langfuse tracing if connection fails
                pass

    def close(self):
        """Release resources held by this LLM4Tenant instance.

        IMPORTANT: do NOT call ``langfuse.flush()`` or ``langfuse.shutdown()``
        here. ``close()`` runs once per task, synchronously, on the asyncio
        event-loop thread of the task executor. Two problems follow:

        - ``flush()`` blocks on an unbounded ``queue.join()`` in the underlying
          OpenTelemetry span processor. If the exporter cannot drain (slow or
          unreachable Langfuse, or an already-shutdown processor) it never
          returns.
        - ``shutdown()`` permanently tears down the process-wide Langfuse /
          OpenTelemetry tracer provider that every ``LLMBundle`` shares. After
          the first task shuts it down, every subsequent ``flush()`` blocks
          forever.

        Because this runs on the event loop, a single stuck ``flush()`` freezes
        the entire task executor: all in-flight parse tasks stop making
        progress and no new tasks are ever picked up (observed as document
        parsing being stuck with every executor thread parked on a lock).

        Langfuse already exports spans from its own background processor and
        flushes at process exit, so releasing the reference is sufficient here.
        """
        # Release the Langfuse client reference. ``Langfuse.flush()`` waits on
        # ``Queue.join()`` with no timeout, so we never call it here: the shared
        # ``LangfuseResourceManager`` flushes its queues at process exit, and
        # a per-task ``flush()`` would block the task executor indefinitely
        # if a consumer is wedged. ``self.langfuse = None`` drops our handle so
        # the next ``LLM4Tenant`` reuses the same shared client.
        self.langfuse = None

        # Release underlying model instance if it has a close method
        if self.mdl and hasattr(self.mdl, "close") and callable(getattr(self.mdl, "close")):
            try:
                self.mdl.close()
            except Exception:
                logging.warning("LLM4Tenant.close: error while closing model instance", exc_info=True)
