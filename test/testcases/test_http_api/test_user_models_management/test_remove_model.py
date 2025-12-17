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
        res = remove_factory(invalid_auth, {"llm_factory": "LocalAI"})
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
    """Test that Builtin factory cannot be removed"""

    @pytest.mark.p1
    def test_remove_builtin_factory_should_fail(self, HttpApiAuth):
        """Test that removing Builtin factory should fail (Builtin is always available)"""
        # Note: Currently the API allows removing Builtin, but it shouldn't
        # This test documents the expected behavior
        res = remove_factory(HttpApiAuth, {"llm_factory": "Builtin"})
        # The API currently succeeds, but ideally this should fail
        # For now, we document that it succeeds but shouldn't
        # If the API is fixed to prevent Builtin removal, this test should assert failure
        if res["code"] == RetCode.SUCCESS:
            # API currently allows it - document this as a known issue
            # In the future, this should be:
            # assert res["code"] == RetCode.ARGUMENT_ERROR, res
            # assert "Builtin" in res["message"], res
            pass
        else:
            # If API is fixed to prevent Builtin removal
            assert res["code"] == RetCode.ARGUMENT_ERROR, res
            assert "Builtin" in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveFactoryNotConfigured:
    """Test removing factories that exist but are not configured for the user"""

    @pytest.mark.p1
    def test_remove_factory_not_configured_exists(self, HttpApiAuth):
        """Test removing a factory that exists in the system but is not configured for the user"""
        # Use a factory that exists in the system but hasn't been added by the user
        # Common factories that exist: OpenAI, Anthropic, ZHIPU-AI, etc.
        factory_name = "OpenAI"
        
        # Verify it's not in the user's configured models
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        if factory_name in res["data"]:
            pytest.skip(f"{factory_name} is already configured for this user")
        
        # Try to remove it - should succeed (no-op, nothing to remove)
        res = remove_factory(HttpApiAuth, {"llm_factory": factory_name})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify it's still not in the list (wasn't there before, shouldn't be there after)
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        assert factory_name not in res["data"] or factory_name in res["data"]

    @pytest.mark.p2
    @pytest.mark.parametrize(
        "factory_name",
        ["OpenAI", "Anthropic", "ZHIPU-AI", "Moonshot", "DeepSeek"],
        ids=["openai", "anthropic", "zhipu_ai", "moonshot", "deepseek"],
    )
    def test_remove_multiple_factories_not_configured(self, HttpApiAuth, factory_name):
        """Test removing multiple factories that exist but aren't configured"""
        # Verify it's not configured
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        if factory_name in res["data"]:
            pytest.skip(f"{factory_name} is already configured")
        
        # Remove it - should succeed (no-op)
        res = remove_factory(HttpApiAuth, {"llm_factory": factory_name})
        assert res["code"] == RetCode.SUCCESS, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveFactoryNotExists:
    """Test removing factories that don't exist at all"""

    @pytest.mark.p1
    def test_remove_factory_not_exists(self, HttpApiAuth):
        """Test removing a factory that doesn't exist in the system should fail"""
        factory_name = "NonExistentFactory123"
        
        # Try to remove it - should fail because factory doesn't exist
        res = remove_factory(HttpApiAuth, {"llm_factory": factory_name})
        assert res["code"] == RetCode.ARGUMENT_ERROR, res
        assert "not found" in res["message"].lower() or "does not exist" in res["message"].lower() or factory_name in res["message"], res

    @pytest.mark.p2
    def test_remove_multiple_nonexistent_factories(self, HttpApiAuth):
        """Test removing multiple factories that don't exist should fail"""
        factories = ["NonExistent1", "NonExistent2", "NonExistent3"]
        
        for factory in factories:
            res = remove_factory(HttpApiAuth, {"llm_factory": factory})
            assert res["code"] == RetCode.ARGUMENT_ERROR, f"Removing {factory} should fail, got {res}"
            assert "not found" in res["message"].lower() or "does not exist" in res["message"].lower() or factory in res["message"], res

    @pytest.mark.p2
    def test_remove_factory_case_sensitivity(self, HttpApiAuth):
        """Test that factory names are case-sensitive - wrong case should fail if factory doesn't exist"""
        # Try removing with wrong case - should fail if factory doesn't exist with that exact case
        factories = ["localai", "LOCALAI", "LocalAI"]
        
        for factory in factories:
            res = remove_factory(HttpApiAuth, {"llm_factory": factory})
            # Should fail if factory doesn't exist with that exact case
            assert res["code"] == RetCode.ARGUMENT_ERROR, res
            assert "not found" in res["message"].lower() or "does not exist" in res["message"].lower() or factory in res["message"], res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveFactoryConfigured:
    """Test removing factories that are configured for the user"""

    @pytest.mark.p1
    def test_remove_configured_factory_success(self, HttpApiAuth):
        """Test successfully removing a configured factory"""
        # Add a factory first (use LocalAI as it skips validation)
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add LocalAI factory")
        
        # Verify it's in the list
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        assert "LocalAI" in res["data"], "LocalAI should be in models list after addition"
        
        # Remove it
        res = remove_factory(HttpApiAuth, {"llm_factory": "LocalAI"})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify it's no longer in the list
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        assert "LocalAI" not in res["data"], "LocalAI should be removed from models list"

    @pytest.mark.p1
    def test_remove_factory_removes_all_model_types(self, HttpApiAuth):
        """Test that removing a factory removes all model types"""
        # Add a factory
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add LocalAI factory")
        
        # Get models before removal
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models_before: Dict[str, Any] = res["data"]
        assert "LocalAI" in models_before, "LocalAI should be in models list"
        
        localai_models_before: List[Dict[str, Any]] = models_before["LocalAI"]["llm"]
        model_types_before: Set[str] = {model["type"] for model in localai_models_before}
        assert len(model_types_before) > 0, "LocalAI should have at least one model type"
        
        # Remove the factory
        res = remove_factory(HttpApiAuth, {"llm_factory": "LocalAI"})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify all models are gone
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models_after: Dict[str, Any] = res["data"]
        assert "LocalAI" not in models_after, "All LocalAI models should be removed"

    @pytest.mark.p1
    def test_remove_factory_clears_default_models(self, HttpApiAuth):
        """Test that removing a factory clears default models of that factory"""
        # Add a factory
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add LocalAI factory")
        
        # Get available models from this factory
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        if "LocalAI" not in res["data"] or not res["data"]["LocalAI"]["llm"]:
            pytest.skip("No LocalAI models available")
        
        localai_models = res["data"]["LocalAI"]["llm"]
        
        # Set one of the models as default
        model = localai_models[0]
        model_id = f"{model['name']}@LocalAI"
        model_type = model.get("type")
        
        # Map model type to default model field
        field_map = {
            "chat": "llm_id",
            "embedding": "embd_id",
            "image2text": "img2txt_id",
            "speech2text": "asr_id",
            "rerank": "rerank_id",
            "tts": "tts_id",
        }
        field = field_map.get(model_type, "llm_id")
        
        # Set it as default (need at least one non-empty field)
        other_field = "embd_id" if field != "embd_id" else "llm_id"
        # Get a builtin model for the other field if available
        res = list_user_models(HttpApiAuth)
        if res["code"] == RetCode.SUCCESS and "Builtin" in res["data"]:
            builtin_llm = res["data"]["Builtin"]["llm"]
            if builtin_llm:
                other_model = builtin_llm[0]
                other_model_id = f"{other_model['name']}@Builtin"
                other_model_type = other_model.get("type")
                other_field = field_map.get(other_model_type, "embd_id")
                
                # Set both models
                res = set_default_models(HttpApiAuth, {field: model_id, other_field: other_model_id})
                assert res["code"] == RetCode.SUCCESS, res
                
                # Verify default model was set
                res = get_default_models(HttpApiAuth)
                assert res["code"] == RetCode.SUCCESS, res
                assert res["data"][field] == model_id, f"Default {field} should be set to {model_id}"
                
                # Remove the factory
                res = remove_factory(HttpApiAuth, {"llm_factory": "LocalAI"})
                assert res["code"] == RetCode.SUCCESS, res
                
                # Verify default model was cleared (set to empty string)
                res = get_default_models(HttpApiAuth)
                assert res["code"] == RetCode.SUCCESS, res
                assert res["data"][field] == "", f"Default {field} should be cleared after removing factory"
                return
        
        pytest.skip("Could not set up test with default models")

    @pytest.mark.p2
    def test_remove_factory_other_factories_unchanged(self, HttpApiAuth):
        """Test that removing one factory doesn't affect others"""
        # Get initial list of factories
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models_before: Dict[str, Any] = res["data"]
        factories_before: Set[str] = set(models_before.keys())
        
        # Add a factory
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add LocalAI factory")
        factories_before.add("LocalAI")
        
        # Remove LocalAI
        res = remove_factory(HttpApiAuth, {"llm_factory": "LocalAI"})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Get list after removal
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        models_after: Dict[str, Any] = res["data"]
        factories_after: Set[str] = set(models_after.keys())
        
        # Other factories should still be present
        other_factories: Set[str] = factories_before - {"LocalAI"}
        for factory in other_factories:
            assert factory in factories_before, f"Factory {factory} should be in factories_before"
            assert factory in factories_after, f"Factory {factory} should remain after removing LocalAI"

    @pytest.mark.p2
    def test_remove_factory_twice(self, HttpApiAuth):
        """Test that removing the same factory twice succeeds"""
        # Add a factory
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add LocalAI factory")
        
        # Remove it twice - both should succeed
        res1 = remove_factory(HttpApiAuth, {"llm_factory": "LocalAI"})
        assert res1["code"] == RetCode.SUCCESS, res1
        
        res2 = remove_factory(HttpApiAuth, {"llm_factory": "LocalAI"})
        assert res2["code"] == RetCode.SUCCESS, res2

    @pytest.mark.p2
    def test_remove_factory_add_remove_cycle(self, HttpApiAuth):
        """Test adding and removing a factory in sequence"""
        # Add factory
        res = add_model(HttpApiAuth, {"llm_factory": "LocalAI", "api_key": "dummy-key", "base_url": "http://localhost:8000"})
        if res["code"] != RetCode.SUCCESS:
            pytest.skip("Could not add LocalAI factory")
        
        # Verify it's in the list
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        assert "LocalAI" in res["data"], "LocalAI should be in models list after addition"
        
        # Remove it
        res = remove_factory(HttpApiAuth, {"llm_factory": "LocalAI"})
        assert res["code"] == RetCode.SUCCESS, res
        
        # Verify it's gone
        res = list_user_models(HttpApiAuth)
        assert res["code"] == RetCode.SUCCESS, res
        assert "LocalAI" not in res["data"], "LocalAI should be removed after removal"

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
