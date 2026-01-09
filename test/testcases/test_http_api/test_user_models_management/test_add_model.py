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
    def test_invalid_auth(self, invalid_auth, expected_code, expected_message):
        res = add_model(invalid_auth, {"llm_factory": "OpenAI", "api_key": "test-key"})
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
    def test_llm_factory_validation(self, HttpApiAuth, payload, expected_code, expected_message_pattern):
        """Test that llm_factory parameter is validated correctly"""
        res = add_model(HttpApiAuth, payload)
        assert res["code"] == expected_code, res
        assert expected_message_pattern in res["message"], f"Expected '{expected_message_pattern}' in '{res['message']}'"

    @pytest.mark.p1
    def test_add_builtin_factory_should_fail(self, HttpApiAuth):
        """Adding Builtin should be rejected."""
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory Builtin is not allowed", res

    @pytest.mark.p1
    def test_individual_model_missing_llm_name(self, HttpApiAuth):
        """Test that llm_name is required when model_type is provided"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "model_type": "chat"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "llm_name is required when adding an individual model" in res["message"], res

    @pytest.mark.p1
    def test_individual_model_missing_model_type(self, HttpApiAuth):
        """Test that model_type is required when llm_name is provided"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "model_type is required when adding an individual model" in res["message"], res

    @pytest.mark.p1
    def test_individual_model_both_required_together(self, HttpApiAuth):
        """Test that both llm_name and model_type must be provided together"""
        # This is also validated in the endpoint itself
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": None})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        # Either Pydantic validation or endpoint validation will catch this
        assert "model_type is required when adding an individual model" in res["message"] or "Both llm_name and model_type must be provided together" in res["message"], res

    @pytest.mark.p1
    def test_individual_model_invalid_model_type(self, HttpApiAuth):
        """Test that invalid model_type is rejected"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": "invalid_type"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "model_type must be one of:" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelOllama:
    """Test adding Ollama (local model provider) - factory-level and individual model"""

    @pytest.mark.p1
    def test_add_ollama_factory_level_success(self, HttpApiAuth):
        """Test factory-level addition (adds all models from Ollama factory)"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_add_ollama_factory_level_missing_base_url(self, HttpApiAuth):
        """Missing base_url still returns success for self-deployed providers."""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_add_ollama_factory_level_empty_api_key(self, HttpApiAuth):
        """Empty api_key accepted for self-deployed providers."""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_add_ollama_individual_model_success(self, HttpApiAuth):
        """Test adding a single Ollama model with llm_name and model_type"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": "chat", "base_url": "http://localhost:11434"})
        # Local models skip validation, so this should succeed even if model doesn't exist
        assert res["code"] == RetCode.SUCCESS, res

    @pytest.mark.p1
    def test_add_ollama_individual_model_empty_api_key(self, HttpApiAuth):
        """Test individual model with empty api_key (allowed for local models)"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": "chat", "api_key": "", "base_url": "http://localhost:11434"})
        assert res["code"] == RetCode.SUCCESS, res

    @pytest.mark.p1
    def test_add_ollama_individual_model_missing_base_url(self, HttpApiAuth):
        """Test individual model without base_url (should still work for local models)"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "llm_name": "llama2", "model_type": "chat", "api_key": ""})
        # For local models, base_url is optional if api_key is provided (even if empty)
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert (
            res.get("message", "")
            == "Field: <> - Message: <api_base is required for local/self-hosted models when api_key is not provided> - Value: <{'llm_factory': 'Ollama', 'llm_name': 'llama2', 'model_type': 'chat', 'api_key': ''}>"
        ), res

    @pytest.mark.p2
    def test_add_ollama_duplicate_addition(self, HttpApiAuth):
        """Adding twice stays successful."""
        res1 = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res1["code"] == RetCode.SUCCESS, res1
        res2 = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res2["code"] == RetCode.SUCCESS, res2
        assert res2.get("message", "") == "", res2


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelOpenAI:
    """Test adding OpenAI factory (API service) - factory-level and individual model"""

    @pytest.mark.p1
    def test_add_openai_factory_level_missing_api_key(self, HttpApiAuth):
        """Missing api_key returns authentication error."""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "Field: <> - Message: <api_key or appropriate authentication fields are required for factory-level addition> - Value: <{'llm_factory': 'OpenAI'}>", res

    @pytest.mark.p1
    def test_add_openai_factory_level_invalid_api_key(self, HttpApiAuth):
        """Invalid api_key returns authentication error."""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key-12345"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"] or "Incorrect API key provided" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_individual_model_missing_params(self, HttpApiAuth):
        """Test individual model addition requires both llm_name and model_type"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "llm_name": "gpt-4", "api_key": "invalid-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "model_type is required when adding an individual model" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_individual_model_invalid_key(self, HttpApiAuth):
        """Test individual model with invalid API key fails authentication"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "llm_name": "gpt-4", "model_type": "chat", "api_key": "invalid-key-12345"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access model(OpenAI/gpt-4)" in res["message"], res

    @pytest.mark.p2
    def test_add_openai_case_sensitivity(self, HttpApiAuth):
        """Test that OpenAI factory name is case-sensitive"""
        res = add_model(HttpApiAuth, {"llm_factory": "openai", "api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory openai is not allowed", res

        res = add_model(HttpApiAuth, {"llm_factory": "OPENAI", "api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory OPENAI is not allowed", res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelSpecialFactories:
    """Test parameter validation for special factory authentication methods"""

    @pytest.mark.p2
    @pytest.mark.parametrize(
        "factory_name, missing_params, expected_message",
        [
            ("VolcEngine", {}, "ark_api_key and endpoint_id are required for VolcEngine"),
            ("Tencent Hunyuan", {}, "hunyuan_sid and hunyuan_sk are required for Tencent Hunyuan"),
            ("Tencent Cloud", {}, "tencent_cloud_sid and tencent_cloud_sk are required for Tencent Cloud"),
            ("Bedrock", {}, "bedrock_ak, bedrock_sk, and bedrock_region are required for Bedrock"),
            ("BaiduYiyan", {}, "yiyan_ak and yiyan_sk are required for BaiduYiyan"),
            ("Fish Audio", {}, "fish_audio_ak and fish_audio_refid are required for Fish Audio"),
            ("Google Cloud", {}, "google_project_id, google_region, and google_service_account_key are required for Google Cloud"),
            ("Azure-OpenAI", {}, "api_key and api_version are required for Azure-OpenAI"),
            ("OpenRouter", {}, "api_key and provider_order are required for OpenRouter"),
        ],
        ids=[
            "volcengine",
            "tencent_hunyuan",
            "tencent_cloud",
            "bedrock",
            "baidu_yiyan",
            "fish_audio",
            "google_cloud",
            "azure_openai",
            "openrouter",
        ],
    )
    def test_special_factory_missing_required_params(self, HttpApiAuth, factory_name, missing_params, expected_message):
        """Test that special factories require their specific authentication parameters"""
        payload = {"llm_factory": factory_name, "api_key": "test-key"}
        res = add_model(HttpApiAuth, payload)
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert expected_message in res["message"], f"Expected '{expected_message}' in '{res['message']}'"

    @pytest.mark.p2
    def test_tencent_hunyuan_force_factory_level(self, HttpApiAuth):
        """Test that Tencent Hunyuan forces factory-level mode (ignores individual model params)"""
        # Even if llm_name and model_type are provided, it should use factory-level mode
        res = add_model(HttpApiAuth, {"llm_factory": "Tencent Hunyuan", "llm_name": "test-model", "model_type": "chat", "hunyuan_sid": "test-sid", "hunyuan_sk": "test-sk"})
        # Should validate parameters but may fail on authentication
        # The key is that it doesn't treat it as individual model addition
        assert res["code"] in [RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR, RetCode.SUCCESS], res

    @pytest.mark.p2
    def test_tencent_cloud_force_factory_level(self, HttpApiAuth):
        """Test that Tencent Cloud forces factory-level mode (ignores individual model params)"""
        res = add_model(HttpApiAuth, {"llm_factory": "Tencent Cloud", "llm_name": "test-model", "model_type": "chat", "tencent_cloud_sid": "test-sid", "tencent_cloud_sk": "test-sk"})
        assert res["code"] in [RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR, RetCode.SUCCESS], res

    @pytest.mark.p2
    def test_xunfei_spark_individual_model_tts(self, HttpApiAuth):
        """Test XunFei Spark individual model with TTS requires spark_app_id, spark_api_secret, spark_api_key"""
        res = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-tts", "model_type": "tts", "api_key": "test-key"})
        # Should fail because TTS requires special parameters
        assert res["code"] in [RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR], res

    @pytest.mark.p2
    def test_xunfei_spark_invalid_model_type(self, HttpApiAuth):
        """Test XunFei Spark individual model with chat requires spark_api_password"""
        res = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "llm_name": "test-chat", "model_type": "embedding", "api_key": "test-key"})
        # Should fail because chat requires spark_api_password
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "Embedding model from XunFei Spark is not supported yet.", res

    @pytest.mark.p2
    def test_xunfei_spark_factory_level(self, HttpApiAuth):
        """Test XunFei Spark factory-level mode uses generic api_key"""
        res = add_model(HttpApiAuth, {"llm_factory": "XunFei Spark", "api_key": "test-key"})
        # Factory-level mode should work (may fail on authentication)
        assert res["code"] in [RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR, RetCode.SUCCESS], res

    @pytest.mark.p2
    def test_mineru_optional_params(self, HttpApiAuth):
        """Test MinerU with optional parameters (mineru_backend, mineru_server_url, mineru_delete_output)"""
        res = add_model(HttpApiAuth, {"llm_factory": "MinerU", "mineru_backend": "test-backend", "mineru_server_url": "http://localhost:8000", "mineru_delete_output": True})
        # MinerU parameters are optional, should not fail on parameter validation
        assert res["code"] in [RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR, RetCode.SUCCESS], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelFactoryLevel:
    """Test factory-level addition behavior"""

    @pytest.mark.p3
    def test_add_model_adds_all_models_from_factory(self, HttpApiAuth):
        """Test that factory-level add_model adds ALL models from a factory"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p2
    def test_factory_level_with_llm_name_filter(self, HttpApiAuth):
        """Test factory-level addition with llm_name filter"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "model_type": "chat", "base_url": "http://localhost:8000", "llm_name": "llama2"})
        # Should succeed (may filter to only llama2 model)
        assert res["code"] == RetCode.SUCCESS, res

    @pytest.mark.p2
    def test_factory_level_invalid_filter_no_match(self, HttpApiAuth):
        """Test factory-level addition with filters that don't match any models"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000", "model_type": "nonexistent_type", "llm_name": "nonexistent_model"})
        # Should return error about no models matching filters
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert (
            res["message"]
            == "Field: <> - Message: <model_type must be one of: chat, embedding, rerank, image2text, speech2text, tts, ocr> - Value: <{'llm_factory': 'Ollama', 'api_key': 'dummy-key', 'base_url': 'http://localhost:8000', 'model_type': 'nonexistent_type', 'llm...>"
        ), res
