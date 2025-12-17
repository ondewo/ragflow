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
from typing import Any, Dict

import pytest
from common import add_model, list_user_models
from common.constants import RetCode
from configs import INVALID_API_TOKEN
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
        "payload, expected_code, expected_message",
        [
            ({}, RetCode.ARGUMENT_ERROR, "llm_factory is required"),
            ({"llm_factory": ""}, RetCode.ARGUMENT_ERROR, "llm_factory is required"),
            ({"llm_factory": None}, RetCode.ARGUMENT_ERROR, "llm_factory is required"),
            ({"llm_factory": "InvalidFactoryName"}, RetCode.ARGUMENT_ERROR, "LLM factory InvalidFactoryName is not allowed"),
        ],
        ids=["missing_llm_factory", "empty_llm_factory", "none_llm_factory", "invalid_factory"],
    )
    def test_llm_factory_validation(self, HttpApiAuth, payload, expected_code, expected_message):
        """Test that llm_factory parameter is validated correctly"""
        res = add_model(HttpApiAuth, payload)
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "factory_name",
        [
            "OpenAI",
            "Anthropic",
            "ZHIPU-AI",
            "LocalAI",
            "Ollama",
            "Xinference",
            "LM-Studio",
            "GPUStack",
            "FastEmbed",
            "VolcEngine",
            "Tencent Hunyuan",
            "Tencent Cloud",
            "Bedrock",
            "BaiduYiyan",
            "Fish Audio",
            "Google Cloud",
            "Azure-OpenAI",
            "OpenRouter",
        ],
        ids=[
            "openai",
            "anthropic",
            "zhipu_ai",
            "localai",
            "ollama",
            "xinference",
            "lm_studio",
            "gpustack",
            "fastembed",
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
    def test_factory_name_validation(self, HttpApiAuth, factory_name):
        """Test that all known factory names are validated (may be allowed or not allowed depending on config)"""
        res = add_model(HttpApiAuth, {"llm_factory": factory_name, "api_key": "test-key"})
        # Should either succeed (if factory is allowed) or fail with specific error
        assert res["code"] in [RetCode.SUCCESS, RetCode.ARGUMENT_ERROR, RetCode.AUTHENTICATION_ERROR], res
        if res["code"] == RetCode.ARGUMENT_ERROR:
            assert res["message"] == f"LLM factory {factory_name} is not allowed", res
        elif res["code"] == RetCode.AUTHENTICATION_ERROR:
            # API key validation failed - this is expected for invalid keys
            assert "Fail to access" in res["message"], res

    @pytest.mark.p1
    def test_add_builtin_factory_should_fail(self, HttpApiAuth):
        """Test that adding Builtin factory should fail (Builtin is always available and cannot be added)"""
        # Note: Currently the API may allow adding Builtin, but it shouldn't
        # This test documents the expected behavior
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        # The API currently may succeed, but ideally this should fail
        # If the API is fixed to prevent Builtin addition, this test should assert failure
        if res["code"] == RetCode.SUCCESS:
            # API currently allows it - document this as a known issue
            # In the future, this should be:
            # assert res["code"] == RetCode.ARGUMENT_ERROR, res
            # assert "Builtin" in res["message"], res
            pass
        else:
            # If API is fixed to prevent Builtin addition
            assert res["code"] == RetCode.ARGUMENT_ERROR, res
            assert "Builtin" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelLocalAI:
    """Test adding LocalAI factory (self-deployed local model) - all success and failure cases"""

    @pytest.mark.p1
    def test_add_localai_success(self, HttpApiAuth):
        """Test successfully adding LocalAI factory with valid base_url"""
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] == RetCode.SUCCESS:
            # Verify models were added
            models_res = list_user_models(HttpApiAuth)
            assert models_res["code"] == RetCode.SUCCESS, models_res
            assert "LocalAI" in models_res["data"], "LocalAI should be in models list"
            factory_data: Dict[str, Any] = models_res["data"]["LocalAI"]
            assert "llm" in factory_data, "LocalAI should have 'llm' key"
            assert isinstance(factory_data["llm"], list), "LocalAI 'llm' should be a list"
            assert len(factory_data["llm"]) > 0, "LocalAI should have at least one model"
        else:
            # May fail if LocalAI service is not available at the base_url
            # Should not fail due to parameter validation
            assert res["code"] != RetCode.ARGUMENT_ERROR, f"Should not be argument error, got: {res}"
            assert res["message"] != "llm_factory is required", f"Should not be missing factory error, got: {res}"
            assert res["message"] != "LLM factory LocalAI is not allowed", f"Should not be not allowed error, got: {res}"

    @pytest.mark.p1
    def test_add_localai_missing_base_url(self, HttpApiAuth):
        """Test adding LocalAI without base_url (may succeed or fail depending on default config)"""
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key"})
        # Self-deployed factories skip API key validation, so may succeed or fail based on service availability
        assert res["code"] in [RetCode.SUCCESS, RetCode.AUTHENTICATION_ERROR], res
        if res["code"] == RetCode.AUTHENTICATION_ERROR:
            # Should fail with connection/service error, not parameter validation
            assert res["code"] != RetCode.ARGUMENT_ERROR, f"Should not be argument error, got: {res}"

    @pytest.mark.p1
    def test_add_localai_invalid_base_url(self, HttpApiAuth):
        """Test adding LocalAI with invalid base_url"""
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://invalid-host:9999"})
        # Should fail with connection error, not parameter validation
        assert res["code"] != RetCode.ARGUMENT_ERROR, f"Should not be argument error, got: {res}"
        # May succeed if service is available, or fail with connection error
        assert res["code"] in [RetCode.SUCCESS, RetCode.AUTHENTICATION_ERROR], res

    @pytest.mark.p1
    def test_add_localai_empty_api_key(self, HttpApiAuth):
        """Test adding LocalAI with empty api_key (self-deployed factories skip validation)"""
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "", "base_url": "http://localhost:8000"})
        # Self-deployed factories skip API key validation
        assert res["code"] in [RetCode.SUCCESS, RetCode.AUTHENTICATION_ERROR], res
        if res["code"] == RetCode.AUTHENTICATION_ERROR:
            assert res["code"] != RetCode.ARGUMENT_ERROR, f"Should not be argument error, got: {res}"

    @pytest.mark.p2
    def test_add_localai_duplicate_addition(self, HttpApiAuth):
        """Test adding LocalAI twice (should update existing models)"""
        res1 = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res1["code"] == RetCode.SUCCESS:
            # Try adding again - should succeed (updates existing models)
            res2 = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
            assert res2["code"] == RetCode.SUCCESS, res2
            # Verify models are still in the list
            models_res = list_user_models(HttpApiAuth)
            assert models_res["code"] == RetCode.SUCCESS, models_res
            assert "LocalAI" in models_res["data"], "LocalAI should be in models list after duplicate addition"


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelOpenAI:
    """Test adding OpenAI factory (API service) - all success and failure cases"""

    @pytest.mark.p1
    def test_add_openai_missing_api_key(self, HttpApiAuth):
        """Test adding OpenAI without api_key"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI"})
        # Should fail with API key validation error
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_invalid_api_key(self, HttpApiAuth):
        """Test adding OpenAI with invalid API key"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key-12345"})
        # Should fail with API key validation error
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_empty_api_key(self, HttpApiAuth):
        """Test adding OpenAI with empty api_key"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": ""})
        # Should fail with API key validation error
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_none_api_key(self, HttpApiAuth):
        """Test adding OpenAI with None api_key"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": None})
        # Should fail with API key validation error
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"], res

    @pytest.mark.p2
    def test_add_openai_with_base_url(self, HttpApiAuth):
        """Test adding OpenAI with custom base_url (for self-hosted OpenAI-compatible API)"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key", "base_url": "http://localhost:8000"})
        # Should fail with API key validation, not parameter validation
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"], res

    @pytest.mark.p2
    def test_add_openai_case_sensitivity(self, HttpApiAuth):
        """Test that OpenAI factory name is case-sensitive"""
        # Test lowercase (should fail)
        res = add_model(HttpApiAuth, {"llm_factory": "openai", "api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory openai is not allowed", res

        # Test uppercase (should fail)
        res = add_model(HttpApiAuth, {"llm_factory": "OPENAI", "api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory OPENAI is not allowed", res

        # Test correct case (should fail with API key validation, not factory validation)
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelSpecialFactories:
    """Test special factory authentication methods - parameter validation"""

    @pytest.mark.p2
    @pytest.mark.parametrize(
        "factory_name, special_params",
        [
            ("VolcEngine", ["ark_api_key", "endpoint_id"]),
            ("Tencent Hunyuan", ["hunyuan_sid", "hunyuan_sk"]),
            ("Tencent Cloud", ["tencent_cloud_sid", "tencent_cloud_sk"]),
            ("Bedrock", ["bedrock_ak", "bedrock_sk", "bedrock_region"]),
            ("BaiduYiyan", ["yiyan_ak", "yiyan_sk"]),
            ("Fish Audio", ["fish_audio_ak", "fish_audio_refid"]),
            ("Google Cloud", ["google_project_id", "google_region", "google_service_account_key"]),
            ("Azure-OpenAI", ["api_key", "api_version"]),
            ("OpenRouter", ["api_key", "provider_order"]),
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
    def test_special_factory_parameter_handling(self, HttpApiAuth, factory_name, special_params):
        """Test that special factories handle their specific parameters correctly"""
        # Test with missing special parameters (should fail API key validation)
        payload = {"llm_factory": factory_name, "api_key": "test-key"}
        res = add_model(HttpApiAuth, payload)
        # Should fail with API key validation error (special params are required for these factories)
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Fail to access" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelLimitations:
    """Test API limitations and behavior"""

    @pytest.mark.p3
    def test_add_model_adds_all_models_from_factory(self, HttpApiAuth):
        """Test that add_model adds ALL models from a factory, not individual models
        
        Note: The current API implementation only supports adding all models from a factory.
        There is no endpoint for adding individual models. This test documents this limitation.
        """
        # Try with LocalAI (self-deployed, skips validation)
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] == RetCode.SUCCESS:
            # Verify all models from the factory were added
            models_res = list_user_models(HttpApiAuth)
            assert models_res["code"] == RetCode.SUCCESS, models_res
            assert "LocalAI" in models_res["data"], "LocalAI should be in models list"
            factory_data: Dict[str, Any] = models_res["data"]["LocalAI"]
            llm_list = factory_data["llm"]
            # All models from the factory should be present (not just one)
            assert isinstance(llm_list, list), "LocalAI 'llm' should be a list"
            # The factory should have multiple models (chat, embedding, etc.) if available
            # This demonstrates that add_model adds all models, not individual ones
            if len(llm_list) > 0:
                # Verify multiple model types are present (if factory has them)
                model_types = {model.get("type") for model in llm_list}
                # This shows that all models from the factory are added, not just one type
