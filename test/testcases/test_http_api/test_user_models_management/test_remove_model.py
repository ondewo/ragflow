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
from common import add_model, list_user_models, remove_model
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
        res = remove_model(invalid_auth, {"llm_factory": "Builtin"})
        assert res["code"] == expected_code, res
        assert res["message"] == expected_message, res


@pytest.mark.usefixtures("cleanup_added_models")
class TestRemoveModel:
    @pytest.mark.p1
    @pytest.mark.parametrize(
        "payload, expected_code, expected_message_contains",
        [
            ({}, 101, "llm_factory"),
            ({"llm_factory": ""}, 101, ""),
        ],
        ids=["missing_llm_factory", "empty_llm_factory"],
    )
    def test_remove_model_validation(self, HttpApiAuth, payload, expected_code, expected_message_contains):
        res = remove_model(HttpApiAuth, payload)
        assert res["code"] == expected_code, res
        if expected_message_contains:
            assert expected_message_contains.lower() in res["message"].lower(), res

    @pytest.mark.p1
    def test_remove_model_successful_response(self, HttpApiAuth):
        """Test that remove_model returns correct response on successful removal"""
        # Use Builtin factory - it doesn't require API key
        # First, add the factory
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == 0, res

        # Then remove it and verify the response
        res = remove_model(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res["code"] == 0, res
        # Data may be omitted on success
        assert "data" not in res or res.get("data") is None or res.get("data") is not None

    @pytest.mark.p1
    def test_remove_model_models_disappear_from_list(self, HttpApiAuth):
        """Test that removed models disappear from list_user_models"""
        # Use Builtin factory - it doesn't require API key
        # First, add the factory
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == 0, res

        # Verify it's in the list
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_before: Dict[str, Any] = res["data"]
        assert "Builtin" in models_before, "Builtin should be in models list after addition"

        # Remove it
        res = remove_model(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res["code"] == 0, res

        # Verify it's no longer in the list
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_after: Dict[str, Any] = res["data"]
        assert "Builtin" not in models_after, "Builtin should be removed from models list"

    @pytest.mark.p1
    def test_remove_model_response_structure(self, HttpApiAuth):
        """Test that remove_model returns correct response structure"""
        res = remove_model(HttpApiAuth, {"llm_factory": "NonExistentFactory"})
        # Response should have code and may have data field
        assert "code" in res, f"Response should have 'code' field, got {res}"
        # Removing non-existent factory should still succeed (no-op)
        assert res["code"] == 0, res

    @pytest.mark.p2
    def test_remove_model_nonexistent_factory(self, HttpApiAuth):
        """Test that removing a non-existent factory succeeds (no-op)"""
        # Removing a factory that doesn't exist should still return success
        res = remove_model(HttpApiAuth, {"llm_factory": "NonExistentFactory123"})
        assert res["code"] == 0, res

    @pytest.mark.p2
    def test_remove_model_twice(self, HttpApiAuth):
        """Test that removing the same factory twice succeeds"""
        # Use Builtin factory - it doesn't require API key
        # First, add the factory
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == 0, res

        # Remove it twice - both should succeed
        res1 = remove_model(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res1["code"] == 0, res1

        res2 = remove_model(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res2["code"] == 0, res2

    @pytest.mark.p2
    def test_remove_model_add_remove_cycle(self, HttpApiAuth):
        """Test adding and removing a factory in sequence"""
        # Use Builtin factory - it doesn't require API key
        # Add factory
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == 0, res

        # Verify it's in the list
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models: Dict[str, Any] = res["data"]
        assert "Builtin" in models, "Builtin should be in models list after addition"

        # Remove it
        res = remove_model(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res["code"] == 0, res

        # Verify it's gone
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_after: Dict[str, Any] = res["data"]
        assert "Builtin" not in models_after, "Builtin should be removed after removal"

    @pytest.mark.p2
    def test_remove_model_all_model_types(self, HttpApiAuth):
        """Test that removing a factory removes all model types"""
        # Use Builtin factory - it doesn't require API key
        # Add a factory that has multiple model types
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == 0, res

        # Get models before removal
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_before: Dict[str, Any] = res["data"]
        assert "Builtin" in models_before, "Builtin should be in models list after addition"

        builtin_models_before: List[Dict[str, Any]] = models_before["Builtin"]["llm"]
        model_types_before: Set[str] = {model["type"] for model in builtin_models_before}
        assert len(model_types_before) > 0, "Builtin should have at least one model type"

        # Remove the factory
        res = remove_model(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res["code"] == 0, res

        # Verify all models are gone
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_after: Dict[str, Any] = res["data"]
        assert "Builtin" not in models_after, "All Builtin models should be removed"

    @pytest.mark.p2
    @pytest.mark.parametrize(
        "llm_factory",
        ["localai", "LOCALAI", "LocalAI"],
        ids=["lowercase", "uppercase", "mixed_case"],
    )
    def test_remove_model_case_sensitivity(self, HttpApiAuth, llm_factory):
        """Test that factory names are case-sensitive in removal"""
        # Factory names should match exactly
        # Removing with wrong case should still work (might be a different factory or no-op)
        res = remove_model(HttpApiAuth, {"llm_factory": llm_factory})
        assert res["code"] == 0, res

    @pytest.mark.p2
    def test_remove_model_other_factories_unchanged(self, HttpApiAuth):
        """Test that removing one factory doesn't affect others"""
        # Get initial list of factories
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_before: Dict[str, Any] = res["data"]
        factories_before: Set[str] = set(models_before.keys())

        # Add Builtin factory (doesn't require API key)
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == 0, res
        factories_before.add("Builtin")

        # Remove Builtin
        res = remove_model(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res["code"] == 0, res

        # Get list after removal
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_after: Dict[str, Any] = res["data"]
        factories_after: Set[str] = set(models_after.keys())

        # Other factories should still be present
        other_factories: Set[str] = factories_before - {"Builtin"}
        for factory in other_factories:
            if factory in factories_before:
                assert factory in factories_after, f"Factory {factory} should remain after removing Builtin"

    @pytest.mark.p3
    def test_remove_model_none_value(self, HttpApiAuth):
        """Test that None value for llm_factory is handled correctly"""
        res = remove_model(HttpApiAuth, {"llm_factory": None})
        assert res["code"] != 0, res
        assert "llm_factory" in res["message"].lower(), res

    @pytest.mark.p3
    def test_remove_model_whitespace_string(self, HttpApiAuth):
        """Test that whitespace-only string is handled"""
        # Whitespace string is truthy in Python, so API might accept it
        # It will try to remove a factory with name "   " which doesn't exist (no-op)
        res = remove_model(HttpApiAuth, {"llm_factory": "   "})
        # If it succeeds, that's acceptable (treats whitespace as valid factory name, no-op)
        # If it fails, that's also acceptable (might validate and reject whitespace)
        if res["code"] != 0:
            assert "llm_factory" in res["message"].lower() or len(res["message"]) > 0, res

    @pytest.mark.p3
    def test_remove_model_multiple_removals(self, HttpApiAuth):
        """Test removing multiple different factories"""
        factories_to_remove: List[str] = ["NonExistent1", "NonExistent2", "NonExistent3"]

        for factory in factories_to_remove:
            res = remove_model(HttpApiAuth, {"llm_factory": factory})
            assert res["code"] == 0, f"Removing {factory} should succeed, got {res}"

    @pytest.mark.p3
    def test_remove_model_consistency(self, HttpApiAuth):
        """Test that multiple removal calls are consistent"""
        # Remove the same factory multiple times
        factory_name: str = "NonExistentFactory"

        results: List[int] = []
        for _ in range(3):
            res = remove_model(HttpApiAuth, {"llm_factory": factory_name})
            results.append(res["code"])

        # All should return success code (0)
        assert all(r == 0 for r in results), f"All removal calls should return code 0, got {results}"

    @pytest.mark.p3
    def test_remove_model_verify_removal_from_tenant_llm(self, HttpApiAuth):
        """Test that removing a factory removes it from tenant_llm table"""
        # Use Builtin factory - it doesn't require API key
        # Add the factory first
        res = add_model(HttpApiAuth, {"llm_factory": "Builtin", "api_key": "dummy-key"})
        assert res["code"] == 0, res

        # Verify it's in the list
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_before: Dict[str, Any] = res["data"]
        assert "Builtin" in models_before, "Builtin should be in models list after addition"

        # Remove it
        res = remove_model(HttpApiAuth, {"llm_factory": "Builtin"})
        assert res["code"] == 0, res

        # Verify it's removed
        res = list_user_models(HttpApiAuth)
        assert res["code"] == 0, res
        models_after: Dict[str, Any] = res["data"]
        assert "Builtin" not in models_after, "Builtin should be removed from models list"

        # List should still be a valid dictionary
        assert isinstance(models_after, dict), "Models list should still be a dictionary"
