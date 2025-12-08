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
import json
import logging
from typing import Any, Dict, List, Optional

from api.db.services.user_service import TenantService
from api.db.services.tenant_llm_service import TenantLLMService, LLMFactoriesService
from api.db.services.llm_service import LLMService
from api.db.db_models import TenantLLM
from api.utils.api_utils import (
    get_allowed_llm_factories,
    get_data_error_result,
    get_error_data_result,
    get_json_result,
    get_request_json,
    server_error_response,
    token_required,
)
from common.constants import StatusEnum, LLMType
from rag.llm import ChatModel, EmbeddingModel, RerankModel
from quart import Response, request


@manager.route("/models/default", methods=["POST"])  # noqa: F821
@token_required
async def set_default_models(tenant_id: str) -> Response:
    """
    Set default models for the tenant.

    Args:
        tenant_id (str): The tenant ID extracted from the API token.

    Request Parameters:
        Request body (JSON, all optional):
            - llm_id (str): Default LLM (chat) model ID
            - embd_id (str): Default embedding model ID
            - asr_id (str): Default ASR (speech-to-text) model ID
            - img2txt_id (str): Default image-to-text model ID
            - rerank_id (str): Default rerank model ID
            - tts_id (str): Default TTS (text-to-speech) model ID

    Returns:
        Response: A JSON response containing the operation result.
            - On success: Returns code 200 with data=True
            - On error: Returns error response with appropriate error code and message

    Notes:
        - The method validates that all provided model IDs exist in the tenant's configured models.
        - Builtin models are always allowed and do not require prior configuration.
        - At least one model ID must be provided in the request.
        - Only the model IDs provided in the request will be updated; others remain unchanged.
    """
    try:
        req: Dict[str, Any] = await get_request_json()
        
        # Validate that tenant exists
        tenants: List[Dict[str, Any]] = TenantService.get_info_by(tenant_id)
        if not tenants:
            return get_error_data_result("Tenant not found!")
        
        tenant: Dict[str, Any] = tenants[0]
        tenant_id_db: str = tenant["tenant_id"]
        
        # Prepare update data - only include fields that are provided
        update_data: Dict[str, str] = {}
        model_type_mapping: Dict[str, Any] = {
            "llm_id": LLMType.CHAT.value,
            "embd_id": LLMType.EMBEDDING.value,
            "asr_id": LLMType.SPEECH2TEXT.value,
            "img2txt_id": LLMType.IMAGE2TEXT.value,
            "rerank_id": LLMType.RERANK,
            "tts_id": LLMType.TTS,
        }
        
        # Validate each model ID before adding to update_data
        for field_name, model_type in model_type_mapping.items():
            if field_name in req and req[field_name]:
                model_id: str = req[field_name]
                # Skip validation for empty strings
                if not model_id or model_id.strip() == "":
                    update_data[field_name] = ""
                    continue
                
                # Split model name and factory
                llm_name: str
                llm_factory: Optional[str]
                llm_name, llm_factory = TenantLLMService.split_model_name_and_factory(model_id)
                
                # Check if model exists in user's configured models
                model_exists: List[Any] = TenantLLMService.query(
                    tenant_id=tenant_id,
                    llm_name=llm_name,
                    llm_factory=llm_factory,
                    model_type=model_type
                )
                
                # Builtin models are always allowed (no API key required)
                is_builtin: bool = llm_factory == "Builtin"
                
                if not model_exists and not is_builtin:
                    return get_error_data_result(
                        f"Model '{model_id}' (type: {model_type}) is not configured. "
                        f"Please add the model first using POST /api/v1/models"
                    )
                
                update_data[field_name] = req[field_name]
        
        if not update_data:
            return get_error_data_result("At least one model ID must be provided")
        
        # Update tenant with new default models
        TenantService.update_by_id(tenant_id_db, update_data)
        
        return get_json_result(data=True)
    except Exception as e:
        logging.exception(f"Error setting default models: {e}")
        return server_error_response(e)


@manager.route("/models/default", methods=["GET"])  # noqa: F821
@token_required
async def get_default_models(tenant_id: str) -> Response:
    """
    Get default models for the tenant.

    Args:
        tenant_id (str): The tenant ID extracted from the API token.

    Request Parameters:
        None (GET request with no query parameters or body)

    Returns:
        Response: A JSON response containing the default model IDs.
            - On success: Returns code 200 with data containing:
                - llm_id: Default LLM (chat) model ID
                - embd_id: Default embedding model ID
                - asr_id: Default ASR (speech-to-text) model ID
                - img2txt_id: Default image-to-text model ID
                - rerank_id: Default rerank model ID
                - tts_id: Default TTS (text-to-speech) model ID
            - On error: Returns error response with appropriate error code and message

    Notes:
        - Returns empty strings for model IDs that have not been set.
        - The tenant must exist and be accessible via the provided API token.
    """
    try:
        tenants: List[Dict[str, Any]] = TenantService.get_info_by(tenant_id)
        if not tenants:
            return get_error_data_result("Tenant not found!")
        
        tenant: Dict[str, Any] = tenants[0]
        
        return get_json_result(data={
            "llm_id": tenant.get("llm_id", ""),
            "embd_id": tenant.get("embd_id", ""),
            "asr_id": tenant.get("asr_id", ""),
            "img2txt_id": tenant.get("img2txt_id", ""),
            "rerank_id": tenant.get("rerank_id", ""),
            "tts_id": tenant.get("tts_id", ""),
        })
    except Exception as e:
        logging.exception(f"Error getting default models: {e}")
        return server_error_response(e)


@manager.route("/models", methods=["GET"])  # noqa: F821
@token_required
async def list_user_models(tenant_id: str) -> Response:
    """
    List all configured models for the user.

    Args:
        tenant_id (str): The tenant ID extracted from the API token.

    Request Parameters:
        Query parameters:
            - include_details (bool, optional): Whether to include detailed information (api_base, max_tokens, used_token, status). Default: false

    Returns:
        Response: A JSON response containing the list of configured models.
            - On success: Returns code 200 with data containing a dictionary keyed by factory name.
                Each factory entry contains:
                - tags: Factory tags/metadata
                - llm: List of models with type, name, used_token, and optionally:
                    - api_base: API base URL (if include_details=true)
                    - max_tokens: Maximum tokens (if include_details=true)
                    - status: Model status (if include_details=true)
            - On error: Returns error response with appropriate error code and message

    Notes:
        - The include_details query parameter controls whether detailed information is included.
        - When include_details=false, only basic model information is returned.
        - When include_details=true, additional fields (api_base, max_tokens, status) are included.
    """
    try:
        include_details: bool = request.args.get("include_details", "false").lower() == "true"
        
        if include_details:
            res: Dict[str, Dict[str, Any]] = {}
            objs: List[Any] = TenantLLMService.query(tenant_id=tenant_id)
            factories: List[Any] = LLMFactoriesService.query(status=StatusEnum.VALID.value)
            
            for o in objs:
                o_dict: Dict[str, Any] = o.to_dict()
                factory_tags: Optional[Any] = None
                for f in factories:
                    if f.name == o_dict["llm_factory"]:
                        factory_tags = f.tags
                        break
                
                if o_dict["llm_factory"] not in res:
                    res[o_dict["llm_factory"]] = {"tags": factory_tags, "llm": []}
                
                res[o_dict["llm_factory"]]["llm"].append({
                    "type": o_dict["model_type"],
                    "name": o_dict["llm_name"],
                    "used_token": o_dict.get("used_tokens", 0),
                    "api_base": o_dict.get("api_base") or "",
                    "max_tokens": o_dict.get("max_tokens") or 8192,
                    "status": o_dict.get("status") or "1",
                })
        else:
            res: Dict[str, Dict[str, Any]] = {}
            for o in TenantLLMService.get_my_llms(tenant_id):
                if o["llm_factory"] not in res:
                    res[o["llm_factory"]] = {"tags": o["tags"], "llm": []}
                res[o["llm_factory"]]["llm"].append({
                    "type": o["model_type"],
                    "name": o["llm_name"],
                    "used_token": o.get("used_tokens", 0),
                    "status": o.get("status", "1")
                })
        
        return get_json_result(data=res)
    except Exception as e:
        logging.exception(f"Error listing user models: {e}")
        return server_error_response(e)


@manager.route("/models", methods=["POST"])  # noqa: F821
@token_required
async def add_model(tenant_id: str) -> Response:
    """
    Add models for the user by factory and API key.

    Args:
        tenant_id (str): The tenant ID extracted from the API token.

    Request Parameters:
        Request body (JSON):
            - llm_factory (str, required): LLM factory/provider name (e.g., OpenAI, Anthropic, ZHIPU-AI)
            - api_key (str, required): API key for the factory
            - base_url (str, optional): API base URL (for self-deployed models)
            - Special factory parameters (depending on factory):
                * VolcEngine: ark_api_key, endpoint_id
                * Tencent Hunyuan: hunyuan_sid, hunyuan_sk
                * Tencent Cloud: tencent_cloud_sid, tencent_cloud_sk
                * Bedrock: bedrock_ak, bedrock_sk, bedrock_region
                * BaiduYiyan: yiyan_ak, yiyan_sk
                * Fish Audio: fish_audio_ak, fish_audio_refid
                * Google Cloud: google_project_id, google_region, google_service_account_key
                * Azure-OpenAI: api_key, api_version
                * OpenRouter: api_key, provider_order

    Returns:
        Response: A JSON response containing the operation result.
            - On success: Returns code 200 with data=True
            - On error: Returns error response with appropriate error code and message

    Notes:
        - The method adds all models from the specified factory to the tenant's configuration.
        - The API key is validated by testing access to at least one model of each type (chat, embedding, rerank).
        - Self-deployed models (LocalAI, Ollama, Xinference, etc.) skip API key validation.
        - Special factory authentication methods are supported (VolcEngine, Tencent, Bedrock, etc.).
        - If validation fails, no models are saved and an error is returned.
        - The base_url parameter is optional and used for self-deployed models.
    """
    try:
        req: Dict[str, Any] = await get_request_json()
        
        factory: Optional[str] = req.get("llm_factory")
        api_key: str = req.get("api_key", "x")
        
        if not factory:
            return get_error_data_result("llm_factory is required")
        
        if factory not in [f.name for f in get_allowed_llm_factories()]:
            return get_data_error_result(message=f"LLM factory {factory} is not allowed")

        # Handle special factory authentication methods
        def apikey_json(keys: List[str]) -> str:
            nonlocal req
            return json.dumps({k: req.get(k, "") for k in keys})

        if factory == "VolcEngine":
            api_key = apikey_json(["ark_api_key", "endpoint_id"])
        elif factory == "Tencent Hunyuan":
            api_key = apikey_json(["hunyuan_sid", "hunyuan_sk"])
        elif factory == "Tencent Cloud":
            api_key = apikey_json(["tencent_cloud_sid", "tencent_cloud_sk"])
        elif factory == "Bedrock":
            api_key = apikey_json(["bedrock_ak", "bedrock_sk", "bedrock_region"])
        elif factory == "BaiduYiyan":
            api_key = apikey_json(["yiyan_ak", "yiyan_sk"])
        elif factory == "Fish Audio":
            api_key = apikey_json(["fish_audio_ak", "fish_audio_refid"])
        elif factory == "Google Cloud":
            api_key = apikey_json(["google_project_id", "google_region", "google_service_account_key"])
        elif factory == "Azure-OpenAI":
            api_key = apikey_json(["api_key", "api_version"])
        elif factory == "OpenRouter":
            api_key = apikey_json(["api_key", "provider_order"])

        # Skip API key validation for self-deployed models
        SELF_DEPLOYED_FACTORIES: List[str] = ["LocalAI", "Ollama", "Xinference", "LM-Studio", "GPUStack", "FastEmbed", "Builtin"]
        
        # Test if API key works (skip for self-deployed)
        chat_passed: bool = False
        embd_passed: bool = False
        rerank_passed: bool = False
        extra: Dict[str, str] = {"provider": factory}
        msg: str = ""
        base_url: str = req.get("base_url", "")
        
        if factory not in SELF_DEPLOYED_FACTORIES:
            for llm in LLMService.query(fid=factory):
                if not embd_passed and llm.model_type == LLMType.EMBEDDING.value:
                    assert factory in EmbeddingModel, f"Embedding model from {factory} is not supported yet."
                    mdl = EmbeddingModel[factory](api_key, llm.llm_name, base_url=base_url)
                    try:
                        arr, tc = mdl.encode(["Test if the api key is available"])
                        if len(arr[0]) == 0:
                            raise Exception("Fail")
                        embd_passed = True
                    except Exception as e:
                        msg += f"\nFail to access embedding model({llm.llm_name}) using this api key." + str(e)
                elif not chat_passed and llm.model_type == LLMType.CHAT.value:
                    assert factory in ChatModel, f"Chat model from {factory} is not supported yet."
                    mdl = ChatModel[factory](api_key, llm.llm_name, base_url=base_url, **extra)
                    try:
                        m, tc = mdl.chat(None, [{"role": "user", "content": "Hello! How are you doing!"}], {"temperature": 0.9, "max_tokens": 50})
                        if m.find("**ERROR**") >= 0:
                            raise Exception(m)
                        chat_passed = True
                    except Exception as e:
                        msg += f"\nFail to access model({factory}/{llm.llm_name}) using this api key." + str(e)
                elif not rerank_passed and llm.model_type == LLMType.RERANK:
                    assert factory in RerankModel, f"Re-rank model from {factory} is not supported yet."
                    mdl = RerankModel[factory](api_key, llm.llm_name, base_url=base_url)
                    try:
                        arr, tc = mdl.similarity("What's the weather?", ["Is it sunny today?"])
                        if len(arr) == 0 or tc == 0:
                            raise Exception("Fail")
                        rerank_passed = True
                        logging.debug(f"passed model rerank {llm.llm_name}")
                    except Exception as e:
                        msg += f"\nFail to access model({factory}/{llm.llm_name}) using this api key." + str(e)
                if any([embd_passed, chat_passed, rerank_passed]):
                    msg = ""
                    break

        if msg:
            return get_data_error_result(message=msg)

        # Save all models from this factory
        llm_config: Dict[str, Any] = {"api_key": api_key, "api_base": base_url}
        for llm in LLMService.query(fid=factory):
            llm_config["max_tokens"] = llm.max_tokens
            if not TenantLLMService.filter_update(
                [
                    TenantLLM.tenant_id == tenant_id,
                    TenantLLM.llm_factory == factory,
                    TenantLLM.llm_name == llm.llm_name,
                ],
                llm_config,
            ):
                TenantLLMService.save(
                    tenant_id=tenant_id,
                    llm_factory=factory,
                    llm_name=llm.llm_name,
                    model_type=llm.model_type,
                    api_key=llm_config["api_key"],
                    api_base=llm_config["api_base"],
                    max_tokens=llm_config["max_tokens"],
                )

        return get_json_result(data=True)
    except Exception as e:
        logging.exception(f"Error adding model: {e}")
        return server_error_response(e)


@manager.route("/models", methods=["DELETE"])  # noqa: F821
@token_required
async def remove_model(tenant_id: str) -> Response:
    """
    Remove all models for a factory for the user.

    Args:
        tenant_id (str): The tenant ID extracted from the API token.

    Request Parameters:
        Request body (JSON):
            - llm_factory (str, required): LLM factory/provider name (e.g., OpenAI, Anthropic, ZHIPU-AI)

    Returns:
        Response: A JSON response containing the operation result.
            - On success: Returns code 200 with data=True
            - On error: Returns error response with appropriate error code and message

    Notes:
        - The method removes all models from the specified factory for the tenant.
        - The llm_factory parameter is required in the request body.
        - All models of all types (chat, embedding, rerank, etc.) from the factory are removed.
        - This operation cannot be undone; models must be re-added if needed.
    """
    try:
        req: Dict[str, Any] = await get_request_json()
        
        llm_factory: Optional[str] = req.get("llm_factory")
        
        if not llm_factory:
            return get_error_data_result("llm_factory is required")
        
        TenantLLMService.filter_delete([
            TenantLLM.tenant_id == tenant_id,
            TenantLLM.llm_factory == llm_factory,
        ])
        
        return get_json_result(data=True)
    except Exception as e:
        logging.exception(f"Error removing model: {e}")
        return server_error_response(e)

