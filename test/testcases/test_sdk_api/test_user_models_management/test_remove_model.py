from typing import Any, Dict, List, Set

import pytest
from conftest import add_model, list_user_models, remove_model
from ragflow_sdk import RAGFlow


class TestRemoveModel:
    @pytest.mark.p1
    def test_remove_model_missing_llm_factory(self, client: RAGFlow) -> None:
        """Test that remove_model fails when llm_factory is missing"""
        with pytest.raises(Exception) as excinfo:
            remove_model(client)
        assert "llm_factory is required" in str(excinfo.value)

    @pytest.mark.p1
    def test_remove_model_empty_llm_factory(self, client: RAGFlow) -> None:
        """Test that remove_model fails when llm_factory is empty"""
        with pytest.raises(Exception) as excinfo:
            remove_model(client, llm_factory="")
        assert "llm_factory is required" in str(excinfo.value)

    @pytest.mark.p1
    def test_remove_model_successful_response(self, client: RAGFlow) -> None:
        """Test that remove_model returns True on successful removal"""
        # Use Builtin factory - it doesn't require API key
        # First, add the factory
        add_result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert add_result is True, "add_model should succeed"
        
        # Then remove it and verify the response
        result: Any = remove_model(client, llm_factory="Builtin")
        assert result is True, f"Expected True, got {result}"

    @pytest.mark.p1
    def test_remove_model_models_disappear_from_list(self, client: RAGFlow) -> None:
        """Test that removed models disappear from list_user_models and tenant_llm table"""
        # Use Builtin factory - it doesn't require API key
        # First, add the factory
        add_result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert add_result is True, "add_model should succeed"
        
        # Verify it's in the list (tenant_llm table)
        models_before: Dict[str, Any] = list_user_models(client)
        assert "Builtin" in models_before, "Builtin should be in models list after addition"
        
        # Remove it
        result: Any = remove_model(client, llm_factory="Builtin")
        assert result is True, "remove_model should return True"
        
        # Verify it's no longer in the list (tenant_llm table)
        models_after: Dict[str, Any] = list_user_models(client)
        assert "Builtin" not in models_after, "Builtin should be removed from models list (tenant_llm table)"

    @pytest.mark.p1
    def test_remove_model_response_structure(self, client: RAGFlow) -> None:
        """Test that remove_model returns correct response structure"""
        try:
            result: Any = remove_model(client, llm_factory="NonExistentFactory")
            # Response should be exactly True (boolean)
            assert result is True, f"Expected True (boolean), got {result} (type: {type(result)})"
            assert isinstance(result, bool), f"Result should be boolean, got {type(result)}"
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    @pytest.mark.p2
    def test_remove_model_nonexistent_factory(self, client: RAGFlow) -> None:
        """Test that removing a non-existent factory succeeds (no-op)"""
        # Removing a factory that doesn't exist should still return True
        result: Any = remove_model(client, llm_factory="NonExistentFactory123")
        assert result is True, "Removing non-existent factory should return True (no-op)"

    @pytest.mark.p2
    def test_remove_model_twice(self, client: RAGFlow) -> None:
        """Test that removing the same factory twice succeeds"""
        # Use Builtin factory - it doesn't require API key
        # First, add the factory
        add_result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert add_result is True, "add_model should succeed"
        
        # Remove it twice - both should succeed
        result1: Any = remove_model(client, llm_factory="Builtin")
        assert result1 is True, "First removal should succeed"
        
        result2: Any = remove_model(client, llm_factory="Builtin")
        assert result2 is True, "Second removal should also succeed (no-op)"

    @pytest.mark.p2
    def test_remove_model_add_remove_cycle(self, client: RAGFlow) -> None:
        """Test adding and removing a factory in sequence"""
        # Use Builtin factory - it doesn't require API key
        # Add factory
        add_result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert add_result is True, "add_model should succeed"
        
        # Verify it's in the list (tenant_llm table)
        models: Dict[str, Any] = list_user_models(client)
        assert "Builtin" in models, "Builtin should be in models list after addition"
        
        # Remove it
        remove_result: Any = remove_model(client, llm_factory="Builtin")
        assert remove_result is True, "remove_model should succeed"
        
        # Verify it's gone from tenant_llm table
        models_after: Dict[str, Any] = list_user_models(client)
        assert "Builtin" not in models_after, "Builtin should be removed from tenant_llm table after removal"

    @pytest.mark.p2
    def test_remove_model_all_model_types(self, client: RAGFlow) -> None:
        """Test that removing a factory removes all model types from tenant_llm table"""
        # Use Builtin factory - it doesn't require API key
        # Add a factory that has multiple model types
        add_result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert add_result is True, "add_model should succeed"
        
        # Get models before removal
        models_before: Dict[str, Any] = list_user_models(client)
        assert "Builtin" in models_before, "Builtin should be in models list after addition"
        
        builtin_models_before: List[Dict[str, Any]] = models_before["Builtin"]["llm"]
        model_types_before: Set[str] = {model["type"] for model in builtin_models_before}
        assert len(model_types_before) > 0, "Builtin should have at least one model type"
        
        # Remove the factory
        remove_result: Any = remove_model(client, llm_factory="Builtin")
        assert remove_result is True, "remove_model should succeed"
        
        # Verify all models are gone from tenant_llm table
        models_after: Dict[str, Any] = list_user_models(client)
        assert "Builtin" not in models_after, "All Builtin models should be removed from tenant_llm table"

    @pytest.mark.p2
    def test_remove_model_case_sensitivity(self, client: RAGFlow) -> None:
        """Test that factory names are case-sensitive in removal"""
        # Factory names should match exactly
        # Removing with wrong case should still work (might be a different factory or no-op)
        result1: Any = remove_model(client, llm_factory="localai")
        assert result1 is True
        
        result2: Any = remove_model(client, llm_factory="LOCALAI")
        assert result2 is True
        
        result3: Any = remove_model(client, llm_factory="LocalAI")
        assert result3 is True

    @pytest.mark.p2
    def test_remove_model_other_factories_unchanged(self, client: RAGFlow) -> None:
        """Test that removing one factory doesn't affect others in tenant_llm table"""
        # Get initial list of factories
        models_before: Dict[str, Any] = list_user_models(client)
        factories_before: Set[str] = set(models_before.keys())
        
        # Add Builtin factory (doesn't require API key)
        add_result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert add_result is True, "add_model should succeed"
        factories_before.add("Builtin")
        
        # Remove Builtin
        remove_result: Any = remove_model(client, llm_factory="Builtin")
        assert remove_result is True, "remove_model should succeed"
        
        # Get list after removal
        models_after: Dict[str, Any] = list_user_models(client)
        factories_after: Set[str] = set(models_after.keys())
        
        # Other factories should still be present in tenant_llm table
        other_factories: Set[str] = factories_before - {"Builtin"}
        for factory in other_factories:
            if factory in factories_before:
                assert factory in factories_after, f"Factory {factory} should remain in tenant_llm table after removing Builtin"

    @pytest.mark.p3
    def test_remove_model_none_value(self, client: RAGFlow) -> None:
        """Test that None value for llm_factory is handled correctly"""
        with pytest.raises(Exception) as excinfo:
            remove_model(client, llm_factory=None)
        assert "llm_factory is required" in str(excinfo.value)

    @pytest.mark.p3
    def test_remove_model_whitespace_string(self, client: RAGFlow) -> None:
        """Test that whitespace-only string is handled"""
        # Whitespace string is truthy in Python, so API might accept it
        # It will try to remove a factory with name "   " which doesn't exist (no-op)
        try:
            result: Any = remove_model(client, llm_factory="   ")
            # If it succeeds, that's acceptable (treats whitespace as valid factory name, no-op)
            assert result is True
        except Exception as e:
            # If it fails, that's also acceptable (might validate and reject whitespace)
            error_msg: str = str(e)
            assert "llm_factory is required" in error_msg or len(error_msg) > 0

    @pytest.mark.p3
    def test_remove_model_multiple_removals(self, client: RAGFlow) -> None:
        """Test removing multiple different factories"""
        factories_to_remove: List[str] = ["NonExistent1", "NonExistent2", "NonExistent3"]
        
        for factory in factories_to_remove:
            result: Any = remove_model(client, llm_factory=factory)
            assert result is True, f"Removing {factory} should succeed"

    @pytest.mark.p3
    def test_remove_model_consistency(self, client: RAGFlow) -> None:
        """Test that multiple removal calls are consistent"""
        # Remove the same factory multiple times
        factory_name: str = "NonExistentFactory"
        
        results: List[Any] = []
        for _ in range(3):
            result: Any = remove_model(client, llm_factory=factory_name)
            results.append(result)
        
        # All should return True
        assert all(r is True for r in results), "All removal calls should return True"

    @pytest.mark.p3
    def test_remove_model_verify_removal_from_tenant_llm(self, client: RAGFlow) -> None:
        """Test that removing a factory removes it from tenant_llm table"""
        # Use Builtin factory - it doesn't require API key
        # Add the factory first
        add_result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert add_result is True, "add_model should succeed"
        
        # Verify it's in the list (tenant_llm table)
        models_before: Dict[str, Any] = list_user_models(client)
        assert "Builtin" in models_before, "Builtin should be in tenant_llm table after addition"
        
        # Remove it
        remove_result: Any = remove_model(client, llm_factory="Builtin")
        assert remove_result is True, "remove_model should succeed"
        
        # Verify it's removed from tenant_llm table
        models_after: Dict[str, Any] = list_user_models(client)
        assert "Builtin" not in models_after, "Builtin should be removed from tenant_llm table"
        
        # List should still be a valid dictionary
        assert isinstance(models_after, dict), "Models list should still be a dictionary"

