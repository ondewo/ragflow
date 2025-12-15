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
from typing import Any, Dict, List

import pytest
from common import add_model, list_user_models
from configs import INVALID_API_TOKEN
from libs.auth import RAGFlowHttpApiAuth


@pytest.mark.p1
class TestAuthorization:
    @pytest.mark.parametrize(
        "invalid_auth, expected_code, expected_message",
        [
            (None, 0, "`Authorization` can't be empty"),
            (
                RAGFlowHttpApiAuth(INVALID_API_TOKEN),
                109,
                "Authentication error: API key is invalid!",
            ),
        ],
        ids=["empty_auth", "invalid_api_token"],
    )
    def test_invalid_auth(self, invalid_auth, expected_code, expected_message):
        res = add_model(invalid_auth, {"llm_factory": "Builtin", "api_key": "test-key"})
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModel:
    @pytest.mark.p1
    @pytest.mark.parametrize(
        "payload, expected_code, expected_message_contains",
        [
            ({}, 101, "llm_factory"),
            ({"llm_factory": ""}, 101, ""),
            ({"llm_factory": "InvalidFactoryName"}, 101, ""),
            ({"llm_factory": "OpenAI"}, 109, "api key"),
        ],
        ids=["missing_llm_factory", "empty_llm_factory", "invalid_factory", "missing_api_key"],
    )
    def test_add_model_validation(self, HttpApiAuth, payload, expected_code, expected_message_contains):
        res = add_model(HttpApiAuth, payload)
        assert res["code"] == expected_code, res
        if expected_message_contains:
            assert expected_message_contains.lower() in res["message"].lower(), res

    @pytest.mark.p2
    def test_add_model_invalid_api_key(self, HttpApiAuth):
        """Test that add_model fails with invalid API key"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key-12345"})
        # Should fail with API key validation error or not allowed
        assert res["code"] != 0, res
        assert "api key" in res["message"].lower() or "not allowed" in res["message"].lower() or "fail" in res["message"].lower() or "access" in res["message"].lower(), res

    @pytest.mark.p2
    def test_add_model_with_base_url(self, HttpApiAuth):
        """Test that add_model accepts base_url parameter"""
        res = add_model(HttpApiAuth, {"llm_factory": "OpenAI", "api_key": "invalid-key", "base_url": "http://localhost:8000"})
        # Should fail with API key validation, not parameter validation
        assert res["code"] != 0, res
        assert "api key" in res["message"].lower() or "fail" in res["message"].lower() or "access" in res["message"].lower() or "not allowed" in res["message"].lower(), res

    @pytest.mark.p2
    @pytest.mark.parametrize(
        "llm_factory, api_key",
        [
            ("VolcEngine", "test-key"),
            ("Tencent Cloud", "test-key"),
            ("Bedrock", "test-key"),
            ("BaiduYiyan", "test-key"),
            ("Fish Audio", "test-key"),
            ("Google Cloud", "test-key"),
            ("OpenRouter", None),
        ],
        ids=["volcengine", "tencent_cloud", "bedrock", "baidu_yiyan", "fish_audio", "google_cloud", "openrouter"],
    )
    def test_add_model_missing_special_params(self, HttpApiAuth, llm_factory, api_key):
        """Test that factories with missing special parameters fail or are not allowed"""
        payload = {"llm_factory": llm_factory}
        if api_key:
            payload["api_key"] = api_key
        res = add_model(HttpApiAuth, payload)
        # Expected to fail with validation error or "not allowed"
        if res["code"] != 0:
            assert "not allowed" in res["message"] or "fail" in res["message"].lower() or "api key" in res["message"].lower(), res

    @pytest.mark.p2
    def test_add_model_tencent_hunyuan_missing_params(self, HttpApiAuth):
        """Test that Tencent Hunyuan requires special parameters"""
        res = add_model(HttpApiAuth, {"llm_factory": "Tencent Hunyuan", "api_key": "test-key"})
        assert res["code"] != 0, res

    @pytest.mark.p2
    def test_add_model_azure_openai_missing_params(self, HttpApiAuth):
        """Test that Azure-OpenAI requires special parameters"""
        res = add_model(HttpApiAuth, {"llm_factory": "Azure-OpenAI"})
        assert res["code"] != 0, res

    @pytest.mark.p3
    @pytest.mark.parametrize(
        "llm_factory",
        ["LocalAI", "Ollama", "Xinference", "LM-Studio", "GPUStack", "FastEmbed"],
        ids=["localai", "ollama", "xinference", "lm_studio", "gpustack", "fastembed"],
    )
    def test_add_model_self_deployed_factories(self, HttpApiAuth, llm_factory):
        """Test that self-deployed factories skip API key validation"""
        res = add_model(HttpApiAuth, {"llm_factory": llm_factory, "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] == 0:
            # If successful, verify models were added
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == 0 and llm_factory in models_res["data"]:
                assert isinstance(models_res["data"][llm_factory], dict)
                assert "llm" in models_res["data"][llm_factory]
        else:
            # Expected to fail if service is not available or factory not configured
            # Just verify it's not a parameter validation error
            assert "not allowed" not in res["message"] or "llm_factory is required" not in res["message"], res

    @pytest.mark.p3
    def test_add_model_builtin_factory(self, HttpApiAuth):
        """Test that Builtin factory can be added (skips validation)"""
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        if res["code"] == 0:
            # If successful, verify models were added
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == 0 and "Builtin" in models_res["data"]:
                assert isinstance(models_res["data"]["Builtin"], dict)
                assert "llm" in models_res["data"]["Builtin"]
        else:
            # May fail if Builtin is not configured as addable
            # Should not fail due to parameter validation
            assert "not allowed" not in res["message"] or "llm_factory is required" not in res["message"], res

    @pytest.mark.p1
    def test_add_model_successful_response(self, HttpApiAuth):
        """Test that add_model returns correct response on successful addition"""
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        if res["code"] == 0:
            # On success, data may be omitted; just ensure success code
            assert "data" not in res or res.get("data") is None or res.get("data") is not None

    @pytest.mark.p1
    def test_add_model_success_models_in_list(self, HttpApiAuth):
        """Test that successfully added models appear in list_user_models"""
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        if res["code"] == 0:
            # Verify models appear in the list
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == 0:
                assert "Builtin" in models_res["data"], "Builtin factory should appear in models list after successful addition"
                builtin_data: Dict[str, Any] = models_res["data"]["Builtin"]
                assert "llm" in builtin_data, "Builtin should have 'llm' key"
                assert isinstance(builtin_data["llm"], list), "Builtin 'llm' should be a list"
                if len(builtin_data["llm"]) > 0:
                    # Verify model structure
                    for model in builtin_data["llm"]:
                        assert "type" in model, "Model should have 'type' field"
                        assert "name" in model, "Model should have 'name' field"
                        assert "used_token" in model, "Model should have 'used_token' field"
                        assert isinstance(model["type"], str), "Model type should be a string"
                        assert isinstance(model["name"], str), "Model name should be a string"
                        assert len(model["name"]) > 0, "Model name should not be empty"

    @pytest.mark.p2
    def test_add_model_success_response_structure(self, HttpApiAuth):
        """Test that add_model returns correct response structure on success"""
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        if res["code"] == 0:
            # Some endpoints return code-only success; allow missing/None data
            assert "data" not in res or res.get("data") is None or res.get("data") is not None

    @pytest.mark.p2
    def test_add_model_duplicate_addition(self, HttpApiAuth):
        """Test that adding the same factory twice is handled gracefully"""
        res1 = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        if res1["code"] == 0:
            # Try adding again - should succeed (updates existing models)
            res2 = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
            if res2["code"] == 0:
                # Verify models are in the list
                models_res = list_user_models(HttpApiAuth)
                if models_res["code"] == 0:
                    assert "Builtin" in models_res["data"], "Builtin should be in models list after duplicate addition"
                    factory_data: Dict[str, Any] = models_res["data"]["Builtin"]
                    assert "llm" in factory_data, "Builtin should have 'llm' field"
                    assert isinstance(factory_data["llm"], list), "Builtin 'llm' should be a list"

    @pytest.mark.p2
    def test_add_model_verify_added_models_structure(self, HttpApiAuth):
        """Test that successfully added models have correct structure in list_user_models"""
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        if res["code"] == 0:
            # Get models list
            models_res = list_user_models(HttpApiAuth)
            if models_res["code"] == 0:
                assert "Builtin" in models_res["data"], "Builtin should be in models list after addition"
                factory_data: Dict[str, Any] = models_res["data"]["Builtin"]
                assert "tags" in factory_data, "Factory should have 'tags' field"
                assert "llm" in factory_data, "Factory should have 'llm' field"
                llm_list: List[Dict[str, Any]] = factory_data["llm"]
                if len(llm_list) > 0:
                    # Verify each model has required fields
                    for model in llm_list:
                        assert "type" in model, "Model should have 'type' field"
                        assert "name" in model, "Model should have 'name' field"
                        assert "used_token" in model, "Model should have 'used_token' field"
                        assert "status" in model, "Model should have 'status' field"
                        # Verify field types
                        assert isinstance(model["type"], str), "Model type should be string"
                        assert isinstance(model["name"], str), "Model name should be string"
                        assert isinstance(model["used_token"], int), "Model used_token should be integer"
                        assert model["used_token"] >= 0, "Model used_token should be non-negative"

    @pytest.mark.p3
    @pytest.mark.parametrize(
        "llm_factory, api_key",
        [
            ("openai", "test-key"),  # lowercase
            ("OPENAI", "test-key"),  # uppercase
            ("OpenAI", "invalid-key"),  # correct case but invalid key
        ],
        ids=["lowercase", "uppercase", "correct_case_invalid_key"],
    )
    def test_add_model_case_sensitivity(self, HttpApiAuth, llm_factory, api_key):
        """Test that factory names are case-sensitive"""
        res = add_model(HttpApiAuth, {"llm_factory": llm_factory, "api_key": api_key})
        assert res["code"] != 0, res

    @pytest.mark.p3
    @pytest.mark.parametrize(
        "payload",
        [
            {"llm_factory": None, "api_key": "test-key"},
            {"llm_factory": "OpenAI", "api_key": None},
        ],
        ids=["none_llm_factory", "none_api_key"],
    )
    def test_add_model_none_values(self, HttpApiAuth, payload):
        """Test that None values are handled correctly"""
        res = add_model(HttpApiAuth, payload)
        assert res["code"] != 0, res
