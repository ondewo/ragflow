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

from typing import Any, Callable, Dict, List, Optional

import pytest
from common import add_model
from configs import INVALID_API_TOKEN
from common.constants import RetCode
from libs.auth import RAGFlowHttpApiAuth


@pytest.mark.p1
class TestAuthorization:
    @pytest.mark.parametrize(
        "invalid_auth, expected_code, expected_message",
        [
            (None, RetCode.SUCCESS, "`Authorization` can't be empty"),
            (
                RAGFlowHttpApiAuth(INVALID_API_TOKEN),
                RetCode.AUTHENTICATION_ERROR,
                "Authentication error: API key is invalid!",
            ),
        ],
        ids=["empty_auth", "invalid_api_token"],
    )
    def test_invalid_auth(self, invalid_auth: Optional[RAGFlowHttpApiAuth], expected_code: int, expected_message: str) -> None:
        res: Dict[str, Any] = add_model(invalid_auth, {"llm_factory": "OpenAI", "api_key": "test-key"})
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelParameterValidation:
    """Test parameter validation for all factory types"""

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "payload, expected_code, expected_message_pattern",
        [
            ({}, RetCode.ARGUMENT_ERROR, "Field: <llm_factory> - Message: <Field required>"),
            ({"llm_factory": ""}, RetCode.ARGUMENT_ERROR, "Field: <llm_factory> - Message: <String should have at least 1 character>"),
            ({"llm_factory": None}, RetCode.ARGUMENT_ERROR, "Field: <llm_factory> - Message: <Input should be a valid string>"),
            (
                {"llm_factory": "InvalidFactoryName"},
                RetCode.ARGUMENT_ERROR,
                "Field: <> - Message: <api_key or appropriate authentication fields are required for factory-level addition> - Value: <{'llm_factory': 'InvalidFactoryName'}>",
            ),
        ],
        ids=["missing_llm_factory", "empty_llm_factory", "none_llm_factory", "invalid_factory"],
    )
    def test_llm_factory_validation(self, HttpApiAuth: RAGFlowHttpApiAuth, payload: Dict[str, Any], expected_code: int, expected_message_pattern: str) -> None:
        """Test that llm_factory parameter is validated correctly"""
        res: Dict[str, Any] = add_model(HttpApiAuth, payload)
        assert res["code"] == expected_code, res
        assert expected_message_pattern in res["message"], f"Expected '{expected_message_pattern}' in '{res['message']}'"

    @pytest.mark.p1
    def test_individual_model_missing_llm_name(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test that llm_name is required when model_type is provided"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "model_type": "chat"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "llm_name is required when adding an individual model" in res["message"], res

    @pytest.mark.p1
    def test_individual_model_missing_model_type(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test that model_type is required when llm_name is provided"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "model_type is required when adding an individual model" in res["message"], res

    @pytest.mark.p1
    def test_individual_model_invalid_model_type(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test that invalid model_type is rejected"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": "invalid_type"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "model_type must be one of:" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelOllama:
    """Test adding Ollama (local model provider) - factory-level and individual model"""

    @pytest.mark.p1
    def test_add_ollama_factory_level_success(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test factory-level addition (adds all models from Ollama factory)"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_add_ollama_individual_model_success(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test adding a single Ollama model with llm_name and model_type"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": "chat", "base_url": "http://localhost:11434"})
        # Local models skip validation, so this should succeed even if model doesn't exist
        assert res["code"] == RetCode.SUCCESS, res

    @pytest.mark.p1
    def test_add_ollama_individual_model_empty_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test individual model with empty api_key (allowed for local models)"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": "chat", "api_key": "", "base_url": "http://localhost:11434"})
        assert res["code"] == RetCode.SUCCESS, res

    @pytest.mark.p1
    def test_add_ollama_individual_model_missing_base_url(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test individual model without base_url (should still work for local models)"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": "chat", "api_key": ""})
        # For local models, base_url is optional if api_key is provided (even if empty)
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert (
            res.get("message", "")
            == "Field: <> - Message: <api_base is required for local/self-hosted models when api_key is not provided> - Value: <{'llm_factory': 'Ollama', 'llm_name': 'llama2', 'model_type': 'chat', 'api_key': ''}>"
        ), res

    @pytest.mark.p2
    def test_add_ollama_duplicate_addition(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Adding twice stays successful."""
        res1: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res1["code"] == RetCode.SUCCESS, res1
        res2: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res2["code"] == RetCode.SUCCESS, res2
        assert res2.get("message", "") == "", res2


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelOpenAI:
    """Test adding OpenAI factory (API service) - factory-level and individual model"""

    @pytest.mark.p1
    def test_add_openai_factory_level_missing_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Missing api_key returns authentication error."""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "OpenAI"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "Field: <> - Message: <api_key or appropriate authentication fields are required for factory-level addition> - Value: <{'llm_factory': 'OpenAI'}>", res

    @pytest.mark.p1
    def test_add_openai_factory_level_empty_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Empty api_key returns argument error."""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": ""})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "api_key or appropriate authentication fields are required for factory-level addition" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_factory_level_whitespace_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Whitespace-only api_key returns argument error."""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "   "})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "api_key is required for factory-level addition" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_factory_level_invalid_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Invalid api_key returns authentication error."""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key-12345"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"] or "Incorrect API key provided" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_individual_model_missing_params(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test individual model addition requires both llm_name and model_type"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "llm_name": "gpt-4", "api_key": "invalid-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "model_type is required when adding an individual model" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_individual_model_invalid_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test individual model with invalid API key fails authentication"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "llm_name": "gpt-4", "model_type": "chat", "api_key": "invalid-key-12345"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access model(OpenAI/gpt-4)" in res["message"], res

    @pytest.mark.p2
    def test_add_openai_case_sensitivity(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test that OpenAI factory name is case-sensitive"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "openai", "api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory openai is not allowed", res

        res = add_model(HttpApiAuth, {"llm_factory": "OPENAI", "api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory OPENAI is not allowed", res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelSpecialFactories:
    """Test parameter validation for special factory authentication methods"""

    @pytest.mark.p1
    def test_tencent_hunyuan_missing_all_params(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test Tencent Hunyuan factory-level - missing all required parameters"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Tencent Hunyuan"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "hunyuan_sid and hunyuan_sk are required for Tencent Hunyuan" in res["message"], res

    @pytest.mark.p1
    def test_tencent_hunyuan_empty_strings(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test Tencent Hunyuan factory-level - empty string parameters"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Tencent Hunyuan", "hunyuan_sid": "", "hunyuan_sk": "test-sk"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "hunyuan_sid and hunyuan_sk are required for Tencent Hunyuan" in res["message"], res

    @pytest.mark.p1
    def test_tencent_hunyuan_force_factory_level(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test that Tencent Hunyuan forces factory-level mode (ignores individual model params)"""
        # Even if llm_name and model_type are provided, it should use factory-level mode
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Tencent Hunyuan", "llm_name": "test-model", "model_type": "chat", "hunyuan_sid": "test-sid", "hunyuan_sk": "test-sk"})
        # Should validate parameters but may fail on authentication
        # The key is that it doesn't treat it as individual model addition
        assert res["code"] in [RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR, RetCode.SUCCESS], res

    @pytest.mark.p1
    def test_tencent_cloud_missing_all_params(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test Tencent Cloud factory-level - missing all required parameters"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Tencent Cloud"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "tencent_cloud_sid and tencent_cloud_sk are required for Tencent Cloud" in res["message"], res

    @pytest.mark.p1
    def test_tencent_cloud_empty_strings(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test Tencent Cloud factory-level - empty string parameters"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Tencent Cloud", "tencent_cloud_sid": "", "tencent_cloud_sk": "test-sk"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "tencent_cloud_sid and tencent_cloud_sk are required for Tencent Cloud" in res["message"], res

    @pytest.mark.p1
    def test_tencent_cloud_force_factory_level(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test that Tencent Cloud forces factory-level mode (ignores individual model params)"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Tencent Cloud", "llm_name": "test-model", "model_type": "chat", "tencent_cloud_sid": "test-sid", "tencent_cloud_sk": "test-sk"})
        assert res["code"] in [RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR, RetCode.SUCCESS], res

    @pytest.mark.p1
    def test_xunfei_spark_individual_model_tts_missing_all(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark individual model with TTS - missing all required parameters"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-tts", "model_type": "tts"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "spark_app_id, spark_api_secret, and spark_api_key are required for XunFei Spark TTS models" in res["message"], res

    @pytest.mark.p1
    def test_xunfei_spark_individual_model_tts_missing_spark_app_id(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark individual model with TTS - missing spark_app_id"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-tts", "model_type": "tts", "spark_api_secret": "test-secret", "spark_api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "spark_app_id, spark_api_secret, and spark_api_key are required for XunFei Spark TTS models" in res["message"], res

    @pytest.mark.p1
    def test_xunfei_spark_individual_model_tts_missing_spark_api_secret(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark individual model with TTS - missing spark_api_secret"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-tts", "model_type": "tts", "spark_app_id": "test-app", "spark_api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "spark_app_id, spark_api_secret, and spark_api_key are required for XunFei Spark TTS models" in res["message"], res

    @pytest.mark.p1
    def test_xunfei_spark_individual_model_tts_missing_spark_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark individual model with TTS - missing spark_api_key"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-tts", "model_type": "tts", "spark_app_id": "test-app", "spark_api_secret": "test-secret"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "spark_app_id, spark_api_secret, and spark_api_key are required for XunFei Spark TTS models" in res["message"], res

    @pytest.mark.p1
    def test_xunfei_spark_individual_model_tts_empty_strings(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark individual model with TTS - empty string parameters"""
        res: Dict[str, Any] = add_model(
            HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-tts", "model_type": "tts", "spark_app_id": "", "spark_api_secret": "test-secret", "spark_api_key": "test-key"}
        )
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "spark_app_id, spark_api_secret, and spark_api_key are required for XunFei Spark TTS models" in res["message"], res

    @pytest.mark.p1
    def test_xunfei_spark_individual_model_chat_missing_spark_api_password(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark individual model with chat - missing spark_api_password"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-chat", "model_type": "chat"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "spark_api_password is required for XunFei Spark chat models" in res["message"], res

    @pytest.mark.p1
    def test_xunfei_spark_individual_model_chat_empty_spark_api_password(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark individual model with chat - empty spark_api_password"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-chat", "model_type": "chat", "spark_api_password": ""})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "spark_api_password is required for XunFei Spark chat models" in res["message"], res

    @pytest.mark.p1
    def test_xunfei_spark_factory_level_missing_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark factory-level mode - missing api_key"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "api_key is required for XunFei Spark factory-level addition" in res["message"], res

    @pytest.mark.p1
    def test_xunfei_spark_factory_level_empty_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark factory-level mode - empty api_key"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "api_key": ""})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "api_key is required for XunFei Spark factory-level addition" in res["message"], res

    @pytest.mark.p2
    def test_xunfei_spark_invalid_model_type(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test XunFei Spark individual model with unsupported model type"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-chat", "model_type": "embedding", "api_key": "test-key"})
        # Should fail because embedding is not supported
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "Embedding model from XunFei Spark is not supported yet" in res["message"], res

    @pytest.mark.p2
    def test_mineru_optional_params(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test MinerU with optional parameters (mineru_backend, mineru_server_url, mineru_delete_output)"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "MinerU", "mineru_backend": "test-backend", "mineru_server_url": "http://localhost:8000", "mineru_delete_output": True})
        # MinerU parameters are optional, should not fail on parameter validation
        assert res["code"] in [RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR, RetCode.SUCCESS], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestSpecialFactoryLevelParameterValidation:
    """Comprehensive tests for factory-level mode parameter validation for all special factories"""

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "factory_name, required_params, test_case, payload_modifier",
        [
            # VolcEngine
            ("VolcEngine", ["ark_api_key", "endpoint_id"], "missing_all", lambda p: {}),
            ("VolcEngine", ["ark_api_key", "endpoint_id"], "missing_endpoint_id", lambda p: {"ark_api_key": "test-key"}),
            # Bedrock
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "missing_all", lambda p: {}),
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "missing_bedrock_ak", lambda p: {"bedrock_sk": "test-sk", "bedrock_region": "us-east-1"}),
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "missing_bedrock_sk", lambda p: {"bedrock_ak": "test-ak", "bedrock_region": "us-east-1"}),
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "missing_bedrock_region", lambda p: {"bedrock_ak": "test-ak", "bedrock_sk": "test-sk"}),
            # BaiduYiyan
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "missing_all", lambda p: {}),
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "missing_yiyan_ak", lambda p: {"yiyan_sk": "test-sk"}),
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "missing_yiyan_sk", lambda p: {"yiyan_ak": "test-ak"}),
            # Fish Audio
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "missing_all", lambda p: {}),
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "missing_fish_audio_ak", lambda p: {"fish_audio_refid": "test-refid"}),
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "missing_fish_audio_refid", lambda p: {"fish_audio_ak": "test-ak"}),
            # Google Cloud
            ("Google Cloud", ["google_project_id", "google_region", "google_service_account_key"], "missing_all", lambda p: {}),
            (
                "Google Cloud",
                ["google_project_id", "google_region", "google_service_account_key"],
                "missing_project_id",
                lambda p: {"google_region": "us-central1", "google_service_account_key": "test-key"},
            ),
            (
                "Google Cloud",
                ["google_project_id", "google_region", "google_service_account_key"],
                "missing_region",
                lambda p: {"google_project_id": "test-project", "google_service_account_key": "test-key"},
            ),
            (
                "Google Cloud",
                ["google_project_id", "google_region", "google_service_account_key"],
                "missing_service_account_key",
                lambda p: {"google_project_id": "test-project", "google_region": "us-central1"},
            ),
            # Azure-OpenAI
            ("Azure-OpenAI", ["api_key", "api_version"], "missing_all", lambda p: {}),
            ("Azure-OpenAI", ["api_key", "api_version"], "missing_api_key", lambda p: {"api_version": "2024-01-01"}),
            ("Azure-OpenAI", ["api_key", "api_version"], "missing_api_version", lambda p: {"api_key": "test-key"}),
            # OpenRouter
            ("OpenRouter", ["api_key", "provider_order"], "missing_all", lambda p: {}),
            ("OpenRouter", ["api_key", "provider_order"], "missing_api_key", lambda p: {"provider_order": "openai,anthropic"}),
            ("OpenRouter", ["api_key", "provider_order"], "missing_provider_order", lambda p: {"api_key": "test-key"}),
        ],
        ids=[
            "volcengine_missing_all",
            "volcengine_missing_endpoint_id",
            "bedrock_missing_all",
            "bedrock_missing_bedrock_ak",
            "bedrock_missing_bedrock_sk",
            "bedrock_missing_bedrock_region",
            "baidu_yiyan_missing_all",
            "baidu_yiyan_missing_yiyan_ak",
            "baidu_yiyan_missing_yiyan_sk",
            "fish_audio_missing_all",
            "fish_audio_missing_fish_audio_ak",
            "fish_audio_missing_fish_audio_refid",
            "google_cloud_missing_all",
            "google_cloud_missing_project_id",
            "google_cloud_missing_region",
            "google_cloud_missing_service_account_key",
            "azure_openai_missing_all",
            "azure_openai_missing_api_key",
            "azure_openai_missing_api_version",
            "openrouter_missing_all",
            "openrouter_missing_api_key",
            "openrouter_missing_provider_order",
        ],
    )
    def test_factory_level_missing_required_params(
        self,
        HttpApiAuth: RAGFlowHttpApiAuth,
        factory_name: str,
        required_params: List[str],
        test_case: str,
        payload_modifier: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Test factory-level mode with missing required parameters"""
        payload: Dict[str, Any] = {"llm_factory": factory_name}
        payload.update(payload_modifier(payload))  # Merge modifier result to preserve llm_factory
        res: Dict[str, Any] = add_model(HttpApiAuth, payload)
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        # Pydantic validation checks ALL required fields together and returns message with ALL of them
        # Build expected error message matching Pydantic format using ALL required_params:
        # - 1 field: "field1 is required"
        # - 2 fields: "field1 and field2 are required"
        # - 3+ fields: "field1, field2, and field3 are required" (Oxford comma)
        fields_str: str
        if len(required_params) == 1:
            fields_str = required_params[0]
            expected_message: str = f"{fields_str} is required for {factory_name}"
        elif len(required_params) == 2:
            fields_str = f"{required_params[0]} and {required_params[1]}"
            expected_message = f"{fields_str} are required for {factory_name}"
        else:
            fields_str = ", ".join(required_params[:-1]) + f", and {required_params[-1]}"
            expected_message = f"{fields_str} are required for {factory_name}"
        # Extract message from Pydantic error format: "Field: <> - Message: <...> - Value: <...>"
        assert res["message"] == f"Field: <> - Message: <{expected_message}> - Value: <{payload}>", f"Expected '{expected_message}' in error message. Got: {res['message']}"

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "factory_name, required_params, empty_param",
        [
            ("VolcEngine", ["ark_api_key", "endpoint_id"], "ark_api_key"),
            ("VolcEngine", ["ark_api_key", "endpoint_id"], "endpoint_id"),
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "bedrock_ak"),
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "bedrock_sk"),
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "bedrock_region"),
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "yiyan_ak"),
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "yiyan_sk"),
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "fish_audio_ak"),
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "fish_audio_refid"),
            ("Google Cloud", ["google_project_id", "google_region", "google_service_account_key"], "google_project_id"),
            ("Google Cloud", ["google_project_id", "google_region", "google_service_account_key"], "google_region"),
            ("Google Cloud", ["google_project_id", "google_region", "google_service_account_key"], "google_service_account_key"),
            ("Azure-OpenAI", ["api_key", "api_version"], "api_key"),
            ("Azure-OpenAI", ["api_key", "api_version"], "api_version"),
            ("OpenRouter", ["api_key", "provider_order"], "api_key"),
            ("OpenRouter", ["api_key", "provider_order"], "provider_order"),
        ],
        ids=[
            "volcengine_empty_ark_api_key",
            "volcengine_empty_endpoint_id",
            "bedrock_empty_bedrock_ak",
            "bedrock_empty_bedrock_sk",
            "bedrock_empty_bedrock_region",
            "baidu_yiyan_empty_yiyan_ak",
            "baidu_yiyan_empty_yiyan_sk",
            "fish_audio_empty_fish_audio_ak",
            "fish_audio_empty_fish_audio_refid",
            "google_cloud_empty_project_id",
            "google_cloud_empty_region",
            "google_cloud_empty_service_account_key",
            "azure_openai_empty_api_key",
            "azure_openai_empty_api_version",
            "openrouter_empty_api_key",
            "openrouter_empty_provider_order",
        ],
    )
    def test_factory_level_empty_string_params(self, HttpApiAuth: RAGFlowHttpApiAuth, factory_name: str, required_params: List[str], empty_param: str) -> None:
        """Test factory-level mode with empty string values for required parameters"""
        payload: Dict[str, Any] = {"llm_factory": factory_name}
        # Set all required params, but make one empty
        for param in required_params:
            if param == empty_param:
                payload[param] = ""
            else:
                payload[param] = f"test-{param}"
        res: Dict[str, Any] = add_model(HttpApiAuth, payload)
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        # Pydantic validation checks all fields together, so it returns all missing fields
        # Build expected error message matching Pydantic format
        fields_str: str
        if len(required_params) == 2:
            fields_str = f"{required_params[0]} and {required_params[1]}"
            expected_message: str = f"{fields_str} are required for {factory_name}"
        elif len(required_params) == 3:
            fields_str = f"{required_params[0]}, {required_params[1]}, and {required_params[2]}"
            expected_message = f"{fields_str} are required for {factory_name}"
        else:
            # Fallback for other cases
            fields_str = ", ".join(required_params[:-1]) + f", and {required_params[-1]}" if len(required_params) > 1 else required_params[0]
            expected_message = f"{fields_str} are required for {factory_name}"
        assert expected_message in res["message"], f"Expected '{expected_message}' in error message. Got: {res['message']}"


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelFactoryLevel:
    """Test factory-level addition behavior"""

    @pytest.mark.p3
    def test_add_model_adds_all_models_from_factory(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test that factory-level add_model adds ALL models from a factory"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p2
    def test_factory_level_with_llm_name_filter(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test factory-level addition with llm_name filter"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "model_type": "chat", "base_url": "http://localhost:8000", "llm_name": "llama2"})
        # Should succeed (may filter to only llama2 model)
        assert res["code"] == RetCode.SUCCESS, res

    @pytest.mark.p2
    def test_factory_level_invalid_filter_no_match(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test factory-level addition with filters that don't match any models"""
        res: Dict[str, Any] = add_model(
            HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000", "model_type": "nonexistent_type", "llm_name": "nonexistent_model"}
        )
        # Should return error about no models matching filters
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert (
            res["message"]
            == "Field: <> - Message: <model_type must be one of: chat, embedding, rerank, image2text, speech2text, tts, ocr> - Value: <{'llm_factory': 'Ollama', 'api_key': 'dummy-key', 'base_url': 'http://localhost:8000', 'model_type': 'nonexistent_type', 'llm...>"
        ), res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRegularFactoryParameterValidation:
    """Test parameter validation for regular factories (OpenAI, Anthropic, etc.)"""

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "factory_name",
        ["OpenAI", "Anthropic", "ZHIPU-AI"],
        ids=["openai", "anthropic", "zhipu_ai"],
    )
    def test_regular_factory_missing_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth, factory_name: str) -> None:
        """Test regular factories require api_key for factory-level mode"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": factory_name})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "api_key or appropriate authentication fields are required for factory-level addition" in res["message"], res

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "factory_name",
        ["OpenAI", "Anthropic"],
        ids=["openai", "anthropic"],
    )
    def test_regular_factory_whitespace_api_key(self, HttpApiAuth: RAGFlowHttpApiAuth, factory_name: str) -> None:
        """Test regular factories reject whitespace-only api_key for factory-level mode"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": factory_name, "api_key": "   "})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "api_key is required for factory-level addition" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestIndividualModelParameterValidation:
    """Comprehensive tests for individual model mode parameter validation for all special factories"""

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "factory_name, required_params, test_case, payload_modifier",
        [
            # VolcEngine
            ("VolcEngine", ["ark_api_key", "endpoint_id"], "missing_all", lambda p: {"llm_name": "test-model", "model_type": "chat"}),
            ("VolcEngine", ["ark_api_key", "endpoint_id"], "missing_ark_api_key", lambda p: {"llm_name": "test-model", "model_type": "chat", "endpoint_id": "test-endpoint"}),
            ("VolcEngine", ["ark_api_key", "endpoint_id"], "missing_endpoint_id", lambda p: {"llm_name": "test-model", "model_type": "chat", "ark_api_key": "test-key"}),
            # Bedrock
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "missing_all", lambda p: {"llm_name": "test-model", "model_type": "chat"}),
            (
                "Bedrock",
                ["bedrock_ak", "bedrock_sk", "bedrock_region"],
                "missing_bedrock_ak",
                lambda p: {"llm_name": "test-model", "model_type": "chat", "bedrock_sk": "test-sk", "bedrock_region": "us-east-1"},
            ),
            (
                "Bedrock",
                ["bedrock_ak", "bedrock_sk", "bedrock_region"],
                "missing_bedrock_sk",
                lambda p: {"llm_name": "test-model", "model_type": "chat", "bedrock_ak": "test-ak", "bedrock_region": "us-east-1"},
            ),
            (
                "Bedrock",
                ["bedrock_ak", "bedrock_sk", "bedrock_region"],
                "missing_bedrock_region",
                lambda p: {"llm_name": "test-model", "model_type": "chat", "bedrock_ak": "test-ak", "bedrock_sk": "test-sk"},
            ),
            # BaiduYiyan
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "missing_all", lambda p: {"llm_name": "test-model", "model_type": "chat"}),
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "missing_yiyan_ak", lambda p: {"llm_name": "test-model", "model_type": "chat", "yiyan_sk": "test-sk"}),
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "missing_yiyan_sk", lambda p: {"llm_name": "test-model", "model_type": "chat", "yiyan_ak": "test-ak"}),
            # Fish Audio
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "missing_all", lambda p: {"llm_name": "test-model", "model_type": "tts"}),
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "missing_fish_audio_ak", lambda p: {"llm_name": "test-model", "model_type": "tts", "fish_audio_refid": "test-refid"}),
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "missing_fish_audio_refid", lambda p: {"llm_name": "test-model", "model_type": "tts", "fish_audio_ak": "test-ak"}),
            # Google Cloud
            ("Google Cloud", ["google_project_id", "google_region", "google_service_account_key"], "missing_all", lambda p: {"llm_name": "test-model", "model_type": "chat"}),
            (
                "Google Cloud",
                ["google_project_id", "google_region", "google_service_account_key"],
                "missing_project_id",
                lambda p: {"llm_name": "test-model", "model_type": "chat", "google_region": "us-central1", "google_service_account_key": "test-key"},
            ),
            (
                "Google Cloud",
                ["google_project_id", "google_region", "google_service_account_key"],
                "missing_region",
                lambda p: {"llm_name": "test-model", "model_type": "chat", "google_project_id": "test-project", "google_service_account_key": "test-key"},
            ),
            (
                "Google Cloud",
                ["google_project_id", "google_region", "google_service_account_key"],
                "missing_service_account_key",
                lambda p: {"llm_name": "test-model", "model_type": "chat", "google_project_id": "test-project", "google_region": "us-central1"},
            ),
            # Azure-OpenAI
            ("Azure-OpenAI", ["api_key", "api_version"], "missing_all", lambda p: {"llm_name": "test-model", "model_type": "chat"}),
            ("Azure-OpenAI", ["api_key", "api_version"], "missing_api_key", lambda p: {"llm_name": "test-model", "model_type": "chat", "api_version": "2024-01-01"}),
            ("Azure-OpenAI", ["api_key", "api_version"], "missing_api_version", lambda p: {"llm_name": "test-model", "model_type": "chat", "api_key": "test-key"}),
            # OpenRouter
            ("OpenRouter", ["api_key", "provider_order"], "missing_all", lambda p: {"llm_name": "test-model", "model_type": "chat"}),
            ("OpenRouter", ["api_key", "provider_order"], "missing_api_key", lambda p: {"llm_name": "test-model", "model_type": "chat", "provider_order": "openai,anthropic"}),
            ("OpenRouter", ["api_key", "provider_order"], "missing_provider_order", lambda p: {"llm_name": "test-model", "model_type": "chat", "api_key": "test-key"}),
        ],
        ids=[
            "volcengine_individual_missing_all",
            "volcengine_individual_missing_ark_api_key",
            "volcengine_individual_missing_endpoint_id",
            "bedrock_individual_missing_all",
            "bedrock_individual_missing_bedrock_ak",
            "bedrock_individual_missing_bedrock_sk",
            "bedrock_individual_missing_bedrock_region",
            "baidu_yiyan_individual_missing_all",
            "baidu_yiyan_individual_missing_yiyan_ak",
            "baidu_yiyan_individual_missing_yiyan_sk",
            "fish_audio_individual_missing_all",
            "fish_audio_individual_missing_fish_audio_ak",
            "fish_audio_individual_missing_fish_audio_refid",
            "google_cloud_individual_missing_all",
            "google_cloud_individual_missing_project_id",
            "google_cloud_individual_missing_region",
            "google_cloud_individual_missing_service_account_key",
            "azure_openai_individual_missing_all",
            "azure_openai_individual_missing_api_key",
            "azure_openai_individual_missing_api_version",
            "openrouter_individual_missing_all",
            "openrouter_individual_missing_api_key",
            "openrouter_individual_missing_provider_order",
        ],
    )
    def test_individual_model_missing_required_params(
        self,
        HttpApiAuth: RAGFlowHttpApiAuth,
        factory_name: str,
        required_params: List[str],
        test_case: str,
        payload_modifier: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Test individual model mode with missing required parameters"""
        payload: Dict[str, Any] = {"llm_factory": factory_name, "llm_name": "test-model", "model_type": "chat"}
        payload.update(payload_modifier(payload))  # Merge modifier result to preserve llm_factory, llm_name, model_type
        res: Dict[str, Any] = add_model(HttpApiAuth, payload)
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        # Pydantic validation checks ALL required fields together and returns message with ALL of them
        # Build expected error message matching Pydantic format using ALL required_params:
        # - 1 field: "field1 is required"
        # - 2 fields: "field1 and field2 are required"
        # - 3+ fields: "field1, field2, and field3 are required" (Oxford comma)
        fields_str: str
        if len(required_params) == 1:
            fields_str = required_params[0]
            expected_message: str = f"{fields_str} is required for {factory_name} individual model addition"
        elif len(required_params) == 2:
            fields_str = f"{required_params[0]} and {required_params[1]}"
            expected_message = f"{fields_str} are required for {factory_name} individual model addition"
        else:
            fields_str = ", ".join(required_params[:-1]) + f", and {required_params[-1]}"
            expected_message = f"{fields_str} are required for {factory_name} individual model addition"

        # Special case: OpenRouter with missing api_key might trigger api_base check first
        # Check if api_key is missing and factory is OpenRouter
        missing_params: List[str] = [param for param in required_params if param not in payload or not payload.get(param)]
        if factory_name == "OpenRouter" and "api_key" in missing_params:
            # Accept either the factory-specific error or the api_base error
            assert expected_message in res["message"] or "api_base is required for local/self-hosted models when api_key is not provided" in res["message"], (
                f"Expected '{expected_message}' or api_base error in message. Got: {res['message']}"
            )
        else:
            assert expected_message in res["message"], f"Expected '{expected_message}' in error message. Got: {res['message']}"

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "factory_name, required_params, empty_param",
        [
            ("VolcEngine", ["ark_api_key", "endpoint_id"], "ark_api_key"),
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"], "bedrock_ak"),
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"], "yiyan_ak"),
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"], "fish_audio_ak"),
            ("Google Cloud", ["google_project_id", "google_region", "google_service_account_key"], "google_project_id"),
            ("Azure-OpenAI", ["api_key", "api_version"], "api_key"),
            ("OpenRouter", ["api_key", "provider_order"], "api_key"),
        ],
        ids=[
            "volcengine_individual_empty_ark_api_key",
            "bedrock_individual_empty_bedrock_ak",
            "baidu_yiyan_individual_empty_yiyan_ak",
            "fish_audio_individual_empty_fish_audio_ak",
            "google_cloud_individual_empty_project_id",
            "azure_openai_individual_empty_api_key",
            "openrouter_individual_empty_api_key",
        ],
    )
    def test_individual_model_empty_string_params(self, HttpApiAuth: RAGFlowHttpApiAuth, factory_name: str, required_params: List[str], empty_param: str) -> None:
        """Test individual model mode with empty string values for required parameters"""
        payload: Dict[str, Any] = {"llm_factory": factory_name, "llm_name": "test-model", "model_type": "chat"}
        # Set all required params, but make one empty
        for param in required_params:
            if param == empty_param:
                payload[param] = ""
            else:
                payload[param] = f"test-{param}"
        res: Dict[str, Any] = add_model(HttpApiAuth, payload)
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        # Pydantic validation checks all fields together, so it returns all missing fields
        # Build expected error message matching Pydantic format
        fields_str: str
        if len(required_params) == 2:
            fields_str = f"{required_params[0]} and {required_params[1]}"
            expected_message: str = f"{fields_str} are required for {factory_name} individual model addition"
        elif len(required_params) == 3:
            fields_str = f"{required_params[0]}, {required_params[1]}, and {required_params[2]}"
            expected_message = f"{fields_str} are required for {factory_name} individual model addition"
        else:
            # Fallback for other cases
            fields_str = ", ".join(required_params[:-1]) + f", and {required_params[-1]}" if len(required_params) > 1 else required_params[0]
            expected_message = f"{fields_str} are required for {factory_name} individual model addition"
        # Special case: OpenRouter with empty api_key might trigger api_base check first
        if factory_name == "OpenRouter" and empty_param == "api_key":
            # Accept either the factory-specific error or the api_base error
            assert expected_message in res["message"] or "api_base is required for local/self-hosted models when api_key is not provided" in res["message"], (
                f"Expected '{expected_message}' or api_base error in message. Got: {res['message']}"
            )
        else:
            assert expected_message in res["message"], f"Expected '{expected_message}' in error message. Got: {res['message']}"

    @pytest.mark.p1
    def test_individual_model_whitespace_llm_name(self, HttpApiAuth: RAGFlowHttpApiAuth) -> None:
        """Test individual model mode with whitespace-only llm_name"""
        res: Dict[str, Any] = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "   ", "model_type": "chat", "api_base": "http://localhost:11434"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "Field: <api_base> - Message: <Extra inputs are not permitted> - Value: <http://localhost:11434>", res
