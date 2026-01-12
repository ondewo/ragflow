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

from quart import Response, request

from api.db.services.user_service import TenantService
from api.db.services.tenant_llm_service import TenantLLMService, LLMFactoriesService
from api.db.services.llm_service import LLMService
from api.db.db_models import TenantLLM
from api.utils.api_utils import (
    get_allowed_llm_factories,
    get_error_argument_result,
    get_error_data_result,
    get_request_json,
    get_result,
    token_required,
)
from api.utils.validation_utils import (
    AddModelReq,
    validate_and_parse_json_request,
)
from common.constants import RetCode, StatusEnum, LLMType
from rag.llm import ChatModel, EmbeddingModel, RerankModel, CvModel, TTSModel, OcrModel, Seq2txtModel
from rag.utils.base64_image import test_image


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
            - On success: Returns code 200
            - On error: Returns error response with appropriate error code and message

    Notes:
        - The method validates that all provided model IDs exist in the tenant's configured models.
        - Builtin models are always allowed and do not require prior configuration.
        - At least one model ID must be provided in the request.
        - Only the model IDs provided in the request will be updated; others remain unchanged.
    """
    req: Dict[str, Any] = await get_request_json()

    # Validate that tenant exists
    tenants: List[Dict[str, Any]] = TenantService.get_info_by(tenant_id)
    if not tenants:
        return get_error_data_result("Tenant not found!", code=RetCode.DATA_ERROR)

    tenant: Dict[str, Any] = tenants[0]
    tenant_id_db: str = tenant["tenant_id"]

    # Prepare update data - only include fields that are provided
    update_data: Dict[str, str] = {}
    model_type_mapping: Dict[str, LLMType] = {
        "llm_id": LLMType.CHAT,
        "embd_id": LLMType.EMBEDDING,
        "asr_id": LLMType.SPEECH2TEXT,
        "img2txt_id": LLMType.IMAGE2TEXT,
        "rerank_id": LLMType.RERANK,
        "tts_id": LLMType.TTS,
    }

    # Validate each model ID before adding to update_data
    for field_name, model_type in model_type_mapping.items():
        if field_name in req and req[field_name]:
            model_id: str = req[field_name]
            # Skip validation for empty strings
            if model_id.strip() == "":
                update_data[field_name] = ""
                continue

            # Split model name and factory
            llm_name: str
            llm_factory: Optional[str]
            llm_name, llm_factory = TenantLLMService.split_model_name_and_factory(model_id)

            if llm_factory == "Builtin":
                builtin_exists = LLMService.query(fid="Builtin", llm_name=llm_name, model_type=model_type)
                if not builtin_exists:
                    return get_error_argument_result(f"Model '{model_id}' (type: {model_type}) is not configured. Please add the model first using POST /api/v1/models")
            else:
                model_exists: List[Any] = TenantLLMService.query(tenant_id=tenant_id, llm_name=llm_name, llm_factory=llm_factory, model_type=model_type)
                if not model_exists:
                    return get_error_argument_result(f"Model '{model_id}' (type: {model_type}) is not configured. Please add the model first using POST /api/v1/models")

            update_data[field_name] = req[field_name]

    if not update_data:
        return get_error_argument_result("At least one model ID must be provided")

    # Update tenant with new default models
    TenantService.update_by_id(tenant_id_db, update_data)

    return get_result()


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
    tenants: List[Dict[str, Any]] = TenantService.get_info_by(tenant_id)
    if not tenants:
        return get_error_data_result("Tenant not found!", code=RetCode.DATA_ERROR)

    tenant: Dict[str, Any] = tenants[0]

    return get_result(
        data={
            "llm_id": tenant.get("llm_id", ""),
            "embd_id": tenant.get("embd_id", ""),
            "asr_id": tenant.get("asr_id", ""),
            "img2txt_id": tenant.get("img2txt_id", ""),
            "rerank_id": tenant.get("rerank_id", ""),
            "tts_id": tenant.get("tts_id", ""),
        }
    )


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
    include_details: bool = request.args.get("include_details", "false").lower() == "true"

    if include_details:
        res: Dict[str, Dict[str, Any]] = {}
        tenant_llms: List[Any] = TenantLLMService.query(tenant_id=tenant_id)
        factories: List[Any] = LLMFactoriesService.query(status=StatusEnum.VALID.value)

        for tenant_llm in tenant_llms:
            tenant_llm_dict: Dict[str, Any] = tenant_llm.to_dict()
            factory_tags: Optional[Any] = None
            for factory in factories:
                if factory.name == tenant_llm_dict["llm_factory"]:
                    factory_tags = factory.tags
                    break

            if tenant_llm_dict["llm_factory"] not in res:
                res[tenant_llm_dict["llm_factory"]] = {"tags": factory_tags, "llm": []}

            res[tenant_llm_dict["llm_factory"]]["llm"].append(
                {
                    "type": tenant_llm_dict["model_type"],
                    "name": tenant_llm_dict["llm_name"],
                    "used_token": tenant_llm_dict.get("used_tokens", 0),
                    "api_base": tenant_llm_dict.get("api_base") or "",
                    "max_tokens": tenant_llm_dict.get("max_tokens") or 8192,
                    "status": tenant_llm_dict.get("status") or "1",
                }
            )
    else:
        res: Dict[str, Dict[str, Any]] = {}
        tenant_llms: List[Dict[str, Any]] = TenantLLMService.get_my_llms(tenant_id)
        for tenant_llm in tenant_llms:
            if tenant_llm["llm_factory"] not in res:
                res[tenant_llm["llm_factory"]] = {"tags": tenant_llm["tags"], "llm": []}
            res[tenant_llm["llm_factory"]]["llm"].append(
                {
                    "type": tenant_llm["model_type"],
                    "name": tenant_llm["llm_name"],
                    "used_token": tenant_llm.get("used_tokens", 0),
                    "status": tenant_llm.get("status", "1"),
                }
            )

    return get_result(data=res)


@manager.route("/models", methods=["POST"])  # noqa: F821
@token_required
async def add_model(tenant_id: str) -> Response:
    """
    Add models for the user. Supports two modes:
    1. Factory-level: Add all models from a factory (for AI service providers)
    2. Individual model: Add a single model (for local/self-hosted models)

    Args:
        tenant_id (str): The tenant ID extracted from the API token.

    Request Parameters:
        Request body (JSON):
            - llm_factory (str, required): LLM factory/provider name

        For factory-level addition:
            - api_key (str, required for most factories): API key for the factory
            - base_url (str, optional): API base URL
            - model_type (str, optional): Filter to only add models of this type
            - llm_name (str, optional): Filter to only add this specific model
            - Special factory parameters (depending on factory):
                * VolcEngine: ark_api_key, endpoint_id
                * Tencent Hunyuan: hunyuan_sid, hunyuan_sk
                * Tencent Cloud: tencent_cloud_sid, tencent_cloud_sk
                * Bedrock: bedrock_ak, bedrock_sk, bedrock_region, auth_mode, aws_role_arn
                * BaiduYiyan: yiyan_ak, yiyan_sk
                * Fish Audio: fish_audio_ak, fish_audio_refid
                * Google Cloud: google_project_id, google_region, google_service_account_key
                * Azure-OpenAI: api_key, api_version
                * OpenRouter: api_key, provider_order
                * XunFei Spark: spark_app_id, spark_api_secret, spark_api_key (for TTS) or spark_api_password (for chat)

        For individual model addition:
            - llm_name (str, required): Model name
            - model_type (str, required): Model type (chat, embedding, rerank, image2text, speech2text, tts, ocr)
            - api_base (str, optional): API base URL (required for local models if api_key not provided)
            - api_key (str, optional): API key (can be empty string for local models)
            - max_tokens (int, optional): Maximum tokens for the model

    Returns:
        Response: A JSON response containing the operation result.
            - On success: Returns code 200
            - On error: Returns error response with appropriate error code and message

    Notes:
        - Factory-level mode: Adds all models from the specified factory. API key is validated by testing access.
        - Individual model mode: Adds a single model. The model is tested before being saved.
        - Self-deployed models (LocalAI, Ollama, Xinference, etc.) skip API key validation.
        - Special factory authentication methods are supported.
        - If validation fails, no models are saved and an error is returned.
        - Tencent Hunyuan and Tencent Cloud always use factory-level addition (like set_api_key).
    """
    # Validate request using Pydantic model
    req, err = await validate_and_parse_json_request(request, AddModelReq)
    if err is not None:
        return get_error_argument_result(err)

    factory: str = req["llm_factory"]
    llm_name: Optional[str] = req.get("llm_name")
    model_type: Optional[str] = req.get("model_type")
    api_base: Optional[str] = req.get("base_url")  # base_url is the alias for api_base
    max_tokens: Optional[int] = req.get("max_tokens")

    # Validate tenant exists
    tenants: List[Dict[str, Any]] = TenantService.get_info_by(tenant_id)
    if not tenants:
        return get_error_data_result("Tenant not found!", code=RetCode.DATA_ERROR)

    # Builtin should always be available and must not be added explicitly
    if factory == "Builtin":
        return get_error_argument_result("LLM factory Builtin is not allowed")

    if factory not in [f.name for f in get_allowed_llm_factories()]:
        return get_error_argument_result(f"LLM factory {factory} is not allowed")

    # Helper function to validate required fields are present and non-empty
    def validate_required_fields(field_names: List[str], factory_name: str) -> Optional[str]:
        """
        Validate that required fields are present and non-empty (after stripping whitespace).

        Args:
            field_names: List of field names to validate
            factory_name: Name of the factory (for error messages)

        Returns:
            Error message string if validation fails, None if validation passes
        """
        missing_fields = []
        for field_name in field_names:
            value = req.get(field_name)
            if not value or (isinstance(value, str) and not value.strip()):
                missing_fields.append(field_name)

        if missing_fields:
            fields_str = ", ".join(missing_fields)
            return f"{fields_str} are required for {factory_name}"
        return None

    # Handle special factory authentication methods
    def apikey_json(keys: List[str]) -> str:
        nonlocal req
        return json.dumps({k: req.get(k, "") for k in keys})

    # Initialize api_key - will be set based on factory type
    api_key: str = "x"
    provided_api_key = req.get("api_key")

    # Determine if this is individual model addition or factory-level addition
    # We need to determine this early to validate parameters correctly
    is_individual_model: bool = llm_name is not None and model_type is not None

    # Validate required parameters BEFORE assembling API keys
    # This ensures we catch missing parameters early rather than creating JSON with empty values

    # Tencent Hunyuan and Tencent Cloud always use factory-level addition (like set_api_key)
    # They delegate to set_api_key behavior, so we handle them specially
    if factory == "Tencent Hunyuan":
        # Validate required fields before assembling
        err_msg = validate_required_fields(["hunyuan_sid", "hunyuan_sk"], "Tencent Hunyuan")
        if err_msg:
            return get_error_argument_result(err_msg)
        api_key = apikey_json(["hunyuan_sid", "hunyuan_sk"])
        # Force factory-level mode for these factories
        llm_name = None
        model_type = None
        is_individual_model = False
    elif factory == "Tencent Cloud":
        # Validate required fields before assembling
        err_msg = validate_required_fields(["tencent_cloud_sid", "tencent_cloud_sk"], "Tencent Cloud")
        if err_msg:
            return get_error_argument_result(err_msg)
        api_key = apikey_json(["tencent_cloud_sid", "tencent_cloud_sk"])
        # Force factory-level mode for these factories
        llm_name = None
        model_type = None
        is_individual_model = False
    elif factory == "VolcEngine":
        if is_individual_model:
            # Individual model mode: validate required fields
            err_msg = validate_required_fields(["ark_api_key", "endpoint_id"], "VolcEngine individual model addition")
            if err_msg:
                return get_error_argument_result(err_msg)
        else:
            # Factory-level mode: validate required fields
            err_msg = validate_required_fields(["ark_api_key", "endpoint_id"], "VolcEngine")
            if err_msg:
                return get_error_argument_result(err_msg)
        api_key = apikey_json(["ark_api_key", "endpoint_id"])
    elif factory == "Bedrock":
        if is_individual_model:
            # Individual model mode: validate required fields (bedrock_ak, bedrock_sk, bedrock_region are required)
            err_msg = validate_required_fields(["bedrock_ak", "bedrock_sk", "bedrock_region"], "Bedrock individual model addition")
            if err_msg:
                return get_error_argument_result(err_msg)
        else:
            # Factory-level mode: validate required fields (auth_mode and aws_role_arn are optional)
            err_msg = validate_required_fields(["bedrock_ak", "bedrock_sk", "bedrock_region"], "Bedrock")
            if err_msg:
                return get_error_argument_result(err_msg)
        api_key = apikey_json(["auth_mode", "bedrock_ak", "bedrock_sk", "bedrock_region", "aws_role_arn"])
    elif factory == "BaiduYiyan":
        if is_individual_model:
            # Individual model mode: validate required fields
            err_msg = validate_required_fields(["yiyan_ak", "yiyan_sk"], "BaiduYiyan individual model addition")
            if err_msg:
                return get_error_argument_result(err_msg)
        else:
            # Factory-level mode: validate required fields
            err_msg = validate_required_fields(["yiyan_ak", "yiyan_sk"], "BaiduYiyan")
            if err_msg:
                return get_error_argument_result(err_msg)
        api_key = apikey_json(["yiyan_ak", "yiyan_sk"])
    elif factory == "Fish Audio":
        if is_individual_model:
            # Individual model mode: validate required fields
            err_msg = validate_required_fields(["fish_audio_ak", "fish_audio_refid"], "Fish Audio individual model addition")
            if err_msg:
                return get_error_argument_result(err_msg)
        else:
            # Factory-level mode: validate required fields
            err_msg = validate_required_fields(["fish_audio_ak", "fish_audio_refid"], "Fish Audio")
            if err_msg:
                return get_error_argument_result(err_msg)
        api_key = apikey_json(["fish_audio_ak", "fish_audio_refid"])
    elif factory == "Google Cloud":
        if is_individual_model:
            # Individual model mode: validate required fields
            err_msg = validate_required_fields(["google_project_id", "google_region", "google_service_account_key"], "Google Cloud individual model addition")
            if err_msg:
                return get_error_argument_result(err_msg)
        else:
            # Factory-level mode: validate required fields
            err_msg = validate_required_fields(["google_project_id", "google_region", "google_service_account_key"], "Google Cloud")
            if err_msg:
                return get_error_argument_result(err_msg)
        api_key = apikey_json(["google_project_id", "google_region", "google_service_account_key"])
    elif factory == "Azure-OpenAI":
        if is_individual_model:
            # Individual model mode: validate required fields
            err_msg = validate_required_fields(["api_key", "api_version"], "Azure-OpenAI individual model addition")
            if err_msg:
                return get_error_argument_result(err_msg)
        else:
            # Factory-level mode: validate required fields
            err_msg = validate_required_fields(["api_key", "api_version"], "Azure-OpenAI")
            if err_msg:
                return get_error_argument_result(err_msg)
        api_key = apikey_json(["api_key", "api_version"])
    elif factory == "OpenRouter":
        if is_individual_model:
            # Individual model mode: validate required fields
            err_msg = validate_required_fields(["api_key", "provider_order"], "OpenRouter individual model addition")
            if err_msg:
                return get_error_argument_result(err_msg)
        else:
            # Factory-level mode: validate required fields
            err_msg = validate_required_fields(["api_key", "provider_order"], "OpenRouter")
            if err_msg:
                return get_error_argument_result(err_msg)
        api_key = apikey_json(["api_key", "provider_order"])
    elif factory == "XunFei Spark":
        if is_individual_model:
            # Individual model mode
            if model_type == "tts":
                # Validate required fields for XunFei Spark TTS
                err_msg = validate_required_fields(["spark_app_id", "spark_api_secret", "spark_api_key"], "XunFei Spark TTS models")
                if err_msg:
                    return get_error_argument_result(err_msg)
                api_key = apikey_json(["spark_app_id", "spark_api_secret", "spark_api_key"])
            elif model_type == "chat":
                # Validate required field for XunFei Spark chat
                err_msg = validate_required_fields(["spark_api_password"], "XunFei Spark chat models")
                if err_msg:
                    return get_error_argument_result(err_msg)
                api_key = req.get("spark_api_password", "")
            else:
                # For other model types, use api_key if provided
                api_key = req.get("api_key", "x")
        else:
            # Factory-level mode: api_key is required
            if not provided_api_key or (isinstance(provided_api_key, str) and not provided_api_key.strip()):
                return get_error_argument_result("api_key is required for XunFei Spark factory-level addition")
            api_key = provided_api_key
    elif factory == "MinerU":
        # MinerU uses a special config structure (from web UI)
        # Note: llm_app.py uses api_key + provider_order, but web UI and SDK use this structure
        # This matches the web UI implementation
        mineru_config: Dict[str, Any] = {}
        if req.get("mineru_backend"):
            mineru_config["mineru_backend"] = req["mineru_backend"]
        if req.get("mineru_server_url"):
            mineru_config["mineru_server_url"] = req["mineru_server_url"]
        if req.get("mineru_delete_output") is not None:
            mineru_config["mineru_delete_output"] = req["mineru_delete_output"]
        api_key = json.dumps(mineru_config) if mineru_config else "x"
    else:
        # Use provided api_key or default
        if isinstance(provided_api_key, dict):
            api_key = json.dumps(provided_api_key)
        elif provided_api_key is not None:
            api_key = provided_api_key

    # Local/self-hosted factories
    LOCAL_FACTORIES: List[str] = [
        "LocalAI",
        "Ollama",
        "Xinference",
        "LM-Studio",
        "GPUStack",
        "FastEmbed",
        "HuggingFace",
        "OpenAI-API-Compatible",
        "VLLM",
        "ModelScope",
        "TogetherAI",
        "Replicate",
        "OpenRouter",
        "Builtin",
    ]

    is_local: bool = factory in LOCAL_FACTORIES

    # For local models, allow empty api_key (treat "x" as empty/default)
    if is_local and (not api_key or api_key == "x"):
        api_key = ""

    # Additional validation: if one is provided, both must be provided for individual model mode
    # (This is also validated by Pydantic, but adding explicit check for clarity)
    if (llm_name and not model_type) or (model_type and not llm_name):
        return get_error_argument_result("Both llm_name and model_type must be provided together for individual model addition")

    # Individual model addition mode
    if is_individual_model:
        # Validate llm_name is present and non-empty (required for all individual models)
        if not llm_name or not llm_name.strip():
            return get_error_argument_result("llm_name is required and cannot be empty for individual model addition")

        # Validate model_type is present and non-empty (required for all individual models)
        if not model_type or not model_type.strip():
            return get_error_argument_result("model_type is required and cannot be empty for individual model addition")

        # Note: Factory-specific parameter validation (for special auth fields) is done above
        # before apikey_json calls, ensuring we catch missing parameters early

        # Process model name for local factories
        processed_llm_name = llm_name
        if factory == "LocalAI":
            processed_llm_name = llm_name + "___LocalAI"
        elif factory == "HuggingFace":
            processed_llm_name = llm_name + "___HuggingFace"
        elif factory == "OpenAI-API-Compatible":
            processed_llm_name = llm_name + "___OpenAI-API"
        elif factory == "VLLM":
            processed_llm_name = llm_name + "___VLLM"

        # Test the model before adding
        msg = ""
        mdl_nm = processed_llm_name.split("___")[0]
        extra = {"provider": factory}
        model_api_key = api_key
        model_base_url = api_base or ""

        # Test model based on type
        match model_type:
            case LLMType.EMBEDDING.value:
                if factory not in EmbeddingModel:
                    return get_error_argument_result(f"Embedding model from {factory} is not supported yet.")
                mdl = EmbeddingModel[factory](key=model_api_key, model_name=mdl_nm, base_url=model_base_url)
                try:
                    arr, tc = mdl.encode(["Test if the api key is available"])
                    if len(arr[0]) == 0:
                        raise Exception("Fail")
                except Exception as e:
                    msg = f"\nFail to access embedding model({factory}/{mdl_nm})." + str(e)

            case LLMType.CHAT.value:
                if factory not in ChatModel:
                    return get_error_argument_result(f"Chat model from {factory} is not supported yet.")
                mdl = ChatModel[factory](
                    key=model_api_key,
                    model_name=mdl_nm,
                    base_url=model_base_url,
                    **extra,
                )
                try:
                    # Use async_chat to match llm_app.py behavior
                    m, tc = await mdl.async_chat(None, [{"role": "user", "content": "Hello! How are you doing!"}], {"temperature": 0.9})
                    if not tc and m.find("**ERROR**:") >= 0:
                        raise Exception(m)
                except Exception as e:
                    msg = f"\nFail to access model({factory}/{mdl_nm})." + str(e)

            case LLMType.RERANK.value:
                if factory not in RerankModel:
                    return get_error_argument_result(f"Re-rank model from {factory} is not supported yet.")
                try:
                    mdl = RerankModel[factory](key=model_api_key, model_name=mdl_nm, base_url=model_base_url)
                    arr, tc = mdl.similarity("Hello~ RAGFlower!", ["Hi, there!", "Ohh, my friend!"])
                    if len(arr) == 0:
                        raise Exception("Not known.")
                except KeyError:
                    msg = f"{factory} does not support this model({factory}/{mdl_nm})"
                except Exception as e:
                    msg = f"\nFail to access model({factory}/{mdl_nm})." + str(e)

            case LLMType.IMAGE2TEXT.value:
                if factory not in CvModel:
                    return get_error_argument_result(f"Image to text model from {factory} is not supported yet.")
                mdl = CvModel[factory](key=model_api_key, model_name=mdl_nm, base_url=model_base_url)
                try:
                    image_data = test_image
                    m, tc = mdl.describe(image_data)
                    if not tc and m.find("**ERROR**:") >= 0:
                        raise Exception(m)
                except Exception as e:
                    msg = f"\nFail to access model({factory}/{mdl_nm})." + str(e)

            case LLMType.TTS.value:
                if factory not in TTSModel:
                    return get_error_argument_result(f"TTS model from {factory} is not supported yet.")
                mdl = TTSModel[factory](key=model_api_key, model_name=mdl_nm, base_url=model_base_url)
                try:
                    for resp in mdl.tts("Hello~ RAGFlower!"):
                        pass
                except RuntimeError as e:
                    msg = f"\nFail to access model({factory}/{mdl_nm})." + str(e)
                except (AttributeError, TypeError) as e:
                    # Handle case where required fields are missing (e.g., None.encode())
                    msg = f"\nFail to access model({factory}/{mdl_nm}). Missing or invalid authentication fields." + str(e)

            case LLMType.OCR.value:
                if factory not in OcrModel:
                    return get_error_argument_result(f"OCR model from {factory} is not supported yet.")
                try:
                    mdl = OcrModel[factory](key=model_api_key, model_name=mdl_nm, base_url=model_base_url)
                    ok, reason = mdl.check_available()
                    if not ok:
                        raise RuntimeError(reason or "Model not available")
                except Exception as e:
                    msg = f"\nFail to access model({factory}/{mdl_nm})." + str(e)

            case LLMType.SPEECH2TEXT.value:
                if factory not in Seq2txtModel:
                    return get_error_argument_result(f"Speech model from {factory} is not supported yet.")
                try:
                    mdl = Seq2txtModel[factory](key=model_api_key, model_name=mdl_nm, base_url=model_base_url)
                    # TODO: check the availability
                except Exception as e:
                    msg = f"\nFail to access model({factory}/{mdl_nm})." + str(e)

            case _:
                return get_error_argument_result(f"Unknown model type: {model_type}")

        # Skip validation for local models
        if msg and not is_local:
            return get_error_data_result(message=msg, code=RetCode.AUTHENTICATION_ERROR)

        # Save the individual model
        llm_config: Dict[str, Any] = {
            "api_key": api_key,
            "api_base": api_base or "",
            "max_tokens": max_tokens or 8192,
        }
        if not TenantLLMService.filter_update(
            [
                TenantLLM.tenant_id == tenant_id,
                TenantLLM.llm_factory == factory,
                TenantLLM.llm_name == processed_llm_name,
            ],
            llm_config,
        ):
            TenantLLMService.save(
                tenant_id=tenant_id,
                llm_factory=factory,
                llm_name=processed_llm_name,
                model_type=model_type,
                api_key=llm_config["api_key"],
                api_base=llm_config["api_base"],
                max_tokens=llm_config["max_tokens"],
            )

        return get_result()

    # Factory-level addition mode (like set_api_key)
    else:
        # Validate required parameters for factory-level mode (before API key testing)
        # Note: Special factories (VolcEngine, Bedrock, BaiduYiyan, Fish Audio, Google Cloud,
        # Azure-OpenAI, OpenRouter, XunFei Spark, Tencent Hunyuan, Tencent Cloud) are already
        # validated above before apikey_json calls. For other non-local factories, api_key is required.
        if factory not in ["Tencent Hunyuan", "Tencent Cloud", "VolcEngine", "Bedrock", "BaiduYiyan", "Fish Audio", "Google Cloud", "Azure-OpenAI", "OpenRouter", "XunFei Spark"]:
            if not is_local:
                # For non-local factories that don't use special auth, api_key is required
                if not provided_api_key or (isinstance(provided_api_key, str) and not provided_api_key.strip()):
                    return get_error_argument_result("api_key is required for factory-level addition")

        # Test if API key works (skip for self-deployed)
        chat_passed: bool = False
        embd_passed: bool = False
        rerank_passed: bool = False
        extra: Dict[str, str] = {"provider": factory}
        msg: str = ""
        base_url: str = api_base or ""

        if not is_local:
            # Track if any models were tested (for filtering validation)
            tested_any = False
            for llm in LLMService.query(fid=factory):
                # Optional filtering: if model_type or llm_name provided, only test/add those
                if model_type and llm.model_type != model_type:
                    continue
                if llm_name and llm.llm_name != llm_name:
                    continue

                tested_any = True

                if not embd_passed and llm.model_type == LLMType.EMBEDDING.value:
                    if factory not in EmbeddingModel:
                        continue
                    mdl = EmbeddingModel[factory](api_key, llm.llm_name, base_url=base_url)
                    try:
                        arr, tc = mdl.encode(["Test if the api key is available"])
                        if len(arr[0]) == 0:
                            raise Exception("Fail")
                        embd_passed = True
                    except Exception as e:
                        msg += f"\nFail to access embedding model({llm.llm_name})." + str(e)
                elif not chat_passed and llm.model_type == LLMType.CHAT.value:
                    if factory not in ChatModel:
                        continue
                    mdl = ChatModel[factory](api_key, llm.llm_name, base_url=base_url, **extra)
                    try:
                        # Use async_chat to match llm_app.py behavior
                        m, tc = await mdl.async_chat(None, [{"role": "user", "content": "Hello! How are you doing!"}], {"temperature": 0.9, "max_tokens": 50})
                        if m.find("**ERROR**") >= 0:
                            raise Exception(m)
                        chat_passed = True
                    except Exception as e:
                        msg += f"\nFail to access model({factory}/{llm.llm_name})." + str(e)
                elif not rerank_passed and llm.model_type == LLMType.RERANK.value:
                    if factory not in RerankModel:
                        continue
                    mdl = RerankModel[factory](api_key, llm.llm_name, base_url=base_url)
                    try:
                        arr, tc = mdl.similarity("What's the weather?", ["Is it sunny today?"])
                        if len(arr) == 0 or tc == 0:
                            raise Exception("Fail")
                        rerank_passed = True
                        logging.debug(f"passed model rerank {llm.llm_name}")
                    except Exception as e:
                        msg += f"\nFail to access model({factory}/{llm.llm_name})." + str(e)
                if any([embd_passed, chat_passed, rerank_passed]):
                    msg = ""
                    break

            # If filters were applied but no models matched, return error
            if (model_type or llm_name) and not tested_any:
                filter_msg = []
                if model_type:
                    filter_msg.append(f"model_type={model_type}")
                if llm_name:
                    filter_msg.append(f"llm_name={llm_name}")
                return get_error_argument_result(f"No models found matching filters: {', '.join(filter_msg)}")

        if msg:
            return get_error_data_result(message=msg, code=RetCode.AUTHENTICATION_ERROR)

        # Save all models from this factory (with optional filtering)
        llm_config: Dict[str, Any] = {"api_key": api_key, "api_base": base_url}
        # Add optional filter fields if provided (like set_api_key does)
        if model_type:
            llm_config["model_type"] = model_type
        if llm_name:
            llm_config["llm_name"] = llm_name

        for llm in LLMService.query(fid=factory):
            # Apply optional filtering
            if model_type and llm.model_type != model_type:
                continue
            if llm_name and llm.llm_name != llm_name:
                continue

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

        return get_result()


@manager.route("/models", methods=["DELETE"])  # noqa: F821
@token_required
async def remove_factory(tenant_id: str) -> Response:
    """
    Remove all models for a factory for the user.

    Args:
        tenant_id (str): The tenant ID extracted from the API token.

    Request Parameters:
        Request body (JSON):
            - llm_factory (str, required): LLM factory/provider name (e.g., OpenAI, Anthropic, ZHIPU-AI)

    Returns:
        Response: A JSON response containing the operation result.
            - On success: Returns code 200
            - On error: Returns error response with appropriate error code and message

    Notes:
        - The method removes all models from the specified factory for the tenant.
        - The llm_factory parameter is required in the request body.
        - All models of all types (chat, embedding, rerank, etc.) from the factory are removed.
        - This operation cannot be undone; models must be re-added if needed.
    """
    req: Dict[str, Any] = await get_request_json()

    llm_factory: Optional[str] = req.get("llm_factory")

    if not llm_factory:
        return get_error_argument_result("llm_factory is required")

    if llm_factory == "Builtin":
        return get_error_argument_result("LLM factory Builtin is not allowed")

    TenantLLMService.filter_delete(
        [
            TenantLLM.tenant_id == tenant_id,
            TenantLLM.llm_factory == llm_factory,
        ]
    )
    return get_result()
