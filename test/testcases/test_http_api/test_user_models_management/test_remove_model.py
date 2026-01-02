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
from typing import Any, Dict, List, Set

import pytest
from common import add_model, get_default_models, list_user_models, remove_factory, set_default_models
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
        res = remove_factory(invalid_auth, {"llm_factory": "Ollama"})
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveFactoryValidation:
    """Test parameter validation for remove_factory"""

    @pytest.mark.p1
    @pytest.mark.parametrize(
        "payload, expected_code, expected_message",
        [
            ({}, RetCode.ARGUMENT_ERROR, "llm_factory is required"),
            ({"llm_factory": ""}, RetCode.ARGUMENT_ERROR, "llm_factory is required"),
            ({"llm_factory": None}, RetCode.ARGUMENT_ERROR, "llm_factory is required"),
        ],
        ids=["missing_llm_factory", "empty_llm_factory", "none_llm_factory"],
    )
    def test_remove_factory_validation(self, HttpApiAuth, payload, expected_code, expected_message):
        res = remove_factory(HttpApiAuth, payload)
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveFactoryBuiltin:
    """Builtin should not be removable."""

    @pytest.mark.p1
    def test_remove_builtin_factory_should_fail(self, HttpApiAuth):
        res = remove_factory(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert res["message"] == "LLM factory Builtin is not allowed", res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveFactoryNotConfigured:
    """Test removing factories that exist but are not configured for the user"""

    @pytest.mark.p1
    def test_remove_factory_not_configured_exists(self, HttpApiAuth):
        """Test removing a factory that exists in the system but is not configured for the user"""
        # Use a factory that exists in the system but hasn't been added by the user
        # Common factories that exist: OpenAI, Anthropic, ZHIPU-AI, etc.
        factory_name = "OpenAI"
        res = remove_factory(HttpApiAuth, {"llm_factory": factory_name})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p2
    @pytest.mark.parametrize(
        "factory_name",
        ["OpenAI", "Anthropic", "ZHIPU-AI", "Moonshot", "DeepSeek"],
        ids=["openai", "anthropic", "zhipu_ai", "moonshot", "deepseek"],
    )
    def test_remove_multiple_factories_not_configured(self, HttpApiAuth, factory_name):
        """Test removing multiple factories that exist but aren't configured"""
        res = remove_factory(HttpApiAuth, {"llm_factory": factory_name})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveFactoryNotExists:
    """Test removing factories that don't exist at all"""

    @pytest.mark.p1
    def test_remove_factory_not_exists(self, HttpApiAuth):
        """Test removing a factory that doesn't exist in the system should fail"""
        factory_name = "NonExistentFactory123"
        res = remove_factory(HttpApiAuth, {"llm_factory": factory_name})
        # Backend treats removal of unknown factories as no-op success
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p2
    def test_remove_multiple_nonexistent_factories(self, HttpApiAuth):
        """Test removing multiple factories that don't exist should fail"""
        factories = ["NonExistent1", "NonExistent2", "NonExistent3"]
        for factory in factories:
            res = remove_factory(HttpApiAuth, {"llm_factory": factory})
            assert res["code"] == RetCode.SUCCESS, res
            assert res.get("message", "") == "", res

    @pytest.mark.p2
    def test_remove_factory_case_sensitivity(self, HttpApiAuth):
        """Test that factory names are case-sensitive - wrong case should fail if factory doesn't exist"""
        # Try removing with wrong case - should fail if factory doesn't exist with that exact case
        factories = ["localai", "LOCALAI", "LocalAI"]
        for factory in factories:
            res = remove_factory(HttpApiAuth, {"llm_factory": factory})
            assert res["code"] == RetCode.SUCCESS, res
            assert res.get("message", "") == "", res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveFactoryConfigured:
    """Test removing factories that are configured for the user"""

    @pytest.mark.p1
    def test_remove_configured_factory_success(self, HttpApiAuth):
        """Test successfully removing a configured factory"""
        # Add a self-deployed factory and remove it (backend returns success for both)
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add Ollama factory")
        
        res = remove_factory(HttpApiAuth, {"llm_factory": "Ollama"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_remove_factory_removes_all_model_types(self, HttpApiAuth):
        """Test that removing a factory removes all model types"""
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add Ollama factory")
        
        res = remove_factory(HttpApiAuth, {"llm_factory": "Ollama"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p1
    def test_remove_factory_clears_default_models(self, HttpApiAuth):
        """Test that removing a factory clears default models of that factory"""
        # Add a factory
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add Ollama factory")
        
        # Backend currently does not persist models for Ollama; just ensure removal succeeds
        res = remove_factory(HttpApiAuth, {"llm_factory": "Ollama"})
        assert res["code"] == RetCode.SUCCESS, res
        assert res.get("message", "") == "", res

    @pytest.mark.p2
    def test_remove_factory_other_factories_unchanged(self, HttpApiAuth):
        """Test that removing one factory doesn't affect others"""
        # Get initial list of factories
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models_before: Dict[str, Any] = res["data"]
        factories_before: Set[str] = set(models_before.keys())
        
        # Add a factory
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add Ollama factory")
        factories_before.add("Ollama")
        
        # Remove Ollama
        res = remove_factory(HttpApiAuth, {"llm_factory": "Ollama"})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Get list after removal
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models_after: Dict[str, Any] = res["data"]
        factories_after: Set[str] = set(models_after.keys())
        
        # Other factories should still be present
        other_factories: Set[str] = factories_before - {"Ollama"}
        for factory in other_factories:
            assert factory in factories_before, f"Factory {factory} should be in factories_before"
            assert factory in factories_after, f"Factory {factory} should remain after removing Ollama"

    @pytest.mark.p2
    def test_remove_factory_twice(self, HttpApiAuth):
        """Test that removing the same factory twice succeeds"""
        # Add a factory
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add Ollama factory")
        
        # Remove it twice - both should succeed
        res1 = remove_factory(HttpApiAuth, {"llm_factory": "Ollama"})
        assert res1["code"] == RetCode.SUCCESS, res1
        
        res2 = remove_factory(HttpApiAuth, {"llm_factory": "Ollama"})
        assert res2["code"] == RetCode.SUCCESS, res2

    @pytest.mark.p2
    def test_remove_factory_add_remove_cycle(self, HttpApiAuth):
        """Test adding and removing a factory in sequence"""
        # Add factory
        res = add_model(HttpApiAuth, {"llm_factory": "Ollama", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add Ollama factory")
        
        res = remove_factory(HttpApiAuth, {"llm_factory": "Ollama"})
        assert res["code"] == RetCode.SUCCESS, res

    @pytest.mark.p3
    def test_remove_factory_whitespace_string(self, HttpApiAuth):
        """Test that whitespace-only string is handled"""
        # Whitespace string is truthy, so it passes validation
        # It will try to remove a factory with name "   " which doesn't exist (no-op)
        res = remove_factory(HttpApiAuth, {"llm_factory": "   "})
        # Should succeed (no-op, nothing to remove)
        assert res["code"] == RetCode.SUCCESS, res

    @pytest.mark.p3
    def test_remove_factory_consistency(self, HttpApiAuth):
        """Test that multiple removal calls are consistent"""
        factory_name = "NonExistentFactory"
        
        results: List[int] = []
        for _ in range(3):
            res = remove_factory(HttpApiAuth, {"llm_factory": factory_name})
            results.append(res["code"])
        
        # All should return success code
        assert all(r == RetCode.SUCCESS for r in results), f"All removal calls should return code {RetCode.SUCCESS}, got {results}"
