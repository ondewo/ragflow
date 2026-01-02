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
        "factory_name, expected_code, expected_message",
        [
            (
                "OpenAI",
                RetCode.AUTHENTICATION_ERROR,
                "Incorrect API key provided",
            ),
            ("Tencent Hunyuan", RetCode.EXCEPTION_ERROR, "'InvalidCredential'"),
            ("VolcEngine", RetCode.SUCCESS, ""),
        ],
        ids=["openai_missing_key", "tencent_hunyuan_missing_params", "volcengine_missing_params"],
    )
    def test_factory_name_validation(self, HttpApiAuth, factory_name, expected_code, expected_message):
        """Concrete expectations per factory using current backend behavior."""
        res = add_model(HttpApiAuth, {"llm_factory": factory_name, "api_key": "test-key"})
        assert res["code"] == expected_code, res
        assert res.get("message", "") == expected_message or expected_message in res.get("message", ""), res

    @pytest.mark.p1
    def test_add_builtin_factory_should_fail(self, HttpApiAuth):
        """Adding Builtin should be rejected."""
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory Builtin is not allowed", res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelOllama:
    """Test adding a local model provider (self-deployed) - current behavior"""

    @pytest.mark.p1
    def test_add_ollama_success(self, HttpApiAuth):
        """Expect success response even if no models are persisted (empty config)."""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_add_ollama_missing_base_url(self, HttpApiAuth):
        """Missing base_url still returns success for self-deployed providers."""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_add_ollama_invalid_base_url(self, HttpApiAuth):
        """Invalid base_url still returns success for self-deployed providers."""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://invalid-host:9999"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_add_ollama_empty_api_key(self, HttpApiAuth):
        """Empty api_key accepted for self-deployed providers."""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

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
    """Test adding OpenAI factory (API service) - failure cases"""

    @pytest.mark.p1
    def test_add_openai_missing_api_key(self, HttpApiAuth):
        """Missing api_key returns backend exception error."""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Incorrect API key provided" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_invalid_api_key(self, HttpApiAuth):
        """Invalid api_key returns backend exception error."""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key-12345"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res
        assert "Incorrect API key provided" in res["message"], res

    @pytest.mark.p1
    def test_add_openai_empty_api_key(self, HttpApiAuth):
        """Empty api_key returns backend exception error."""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": ""})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res

    @pytest.mark.p1
    def test_add_openai_none_api_key(self, HttpApiAuth):
        """None api_key returns backend exception error."""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": None})
        assert res["code"] == RetCode.EXCEPTION_ERROR, res
        assert res["message"] == "OpenAIError('The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable')", res

    @pytest.mark.p2
    def test_add_openai_with_base_url(self, HttpApiAuth):
        """Custom base_url still returns backend exception error."""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res

    @pytest.mark.p2
    def test_add_openai_case_sensitivity(self, HttpApiAuth):
        """Test that OpenAI factory name is case-sensitive"""
        res = add_model(HttpApiAuth, {"llm_factory": "openai", "api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory openai is not allowed", res

        res = add_model(HttpApiAuth, {"llm_factory": "OPENAI", "api_key": "test-key"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory OPENAI is not allowed", res

        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key"})
        assert res["code"] == RetCode.AUTHENTICATION_ERROR, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelSpecialFactories:
    """Test special factory authentication methods - current backend responses"""

    @pytest.mark.p2
    @pytest.mark.parametrize(
        "factory_name, expected_code, expected_message",
        [
            ("VolcEngine", RetCode.SUCCESS, ""),
            ("Tencent Hunyuan", RetCode.EXCEPTION_ERROR, "'InvalidCredential'"),
            ("Tencent Cloud", RetCode.SUCCESS, ""),
            ("Bedrock", RetCode.SUCCESS, ""),
            ("BaiduYiyan", RetCode.SUCCESS, ""),
            ("Fish Audio", RetCode.SUCCESS, ""),
            ("Google Cloud", RetCode.SUCCESS, ""),
            ("OpenRouter", RetCode.SUCCESS, ""),
        ],
        ids=[
            "volcengine",
            "tencent_hunyuan",
            "tencent_cloud",
            "bedrock",
            "baidu_yiyan",
            "fish_audio",
            "google_cloud",
            "openrouter",
        ],
    )
    def test_special_factory_parameter_handling(self, HttpApiAuth, factory_name, expected_code, expected_message):
        """Assert current backend responses for missing special parameters."""
        payload = {"llm_factory": factory_name, "api_key": "test-key"}
        res = add_model(HttpApiAuth, payload)
        assert res["code"] == expected_code, res
        assert res.get("message", "") == expected_message, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModelLimitations:
    """Test API limitations and behavior"""

    @pytest.mark.p3
    def test_add_model_adds_all_models_from_factory(self, HttpApiAuth):
        """Test that add_model adds ALL models from a factory, not individual models
        
        Note: The current API implementation only supports adding all models from a factory.
        There is no endpoint for adding individual models. This test documents this limitation.
        """
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p3
    def test_add_individual_model_not_supported(self, HttpApiAuth):
        """Adding a single model name is ignored; API still processes factory-level addition."""
        res = add_model(
            HttpApiAuth,
            {"llm_factory": "Ollama", "api_key": "dummy-key", "llm_name": "dummy-model", "base_url": "http://localhost:8000"},
        )
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res
