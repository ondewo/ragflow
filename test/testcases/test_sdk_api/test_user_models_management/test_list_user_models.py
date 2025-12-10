from typing import Any, Dict, List

import pytest
from conftest import list_user_models
from ragflow_sdk import RAGFlow


class TestListUserModels:
    @pytest.mark.p1
    def test_list_user_models_structure(self, client: RAGFlow) -> None:
        """Test that list_user_models returns a valid dictionary structure"""
        models: Dict[str, Any] = list_user_models(client)
        
        # Should return a dictionary
        assert isinstance(models, dict)
        
        # Each key should be a factory name (string)
        for factory_name in models.keys():
            assert isinstance(factory_name, str)
            
            # Each factory entry should have 'tags' and 'llm' keys
            factory_data: Dict[str, Any] = models[factory_name]
            assert "tags" in factory_data
            assert "llm" in factory_data
            assert isinstance(factory_data["llm"], list)

    @pytest.mark.p1
    def test_list_user_models_basic_fields(self, client: RAGFlow) -> None:
        """Test that list_user_models returns models with basic fields when include_details=false"""
        models: Dict[str, Any] = list_user_models(client, include_details=False)
        
        # Check structure for each factory
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            for model in llm_list:
                # Basic fields should be present
                assert "type" in model
                assert "name" in model
                assert "used_token" in model
                assert "status" in model
                
                # Should NOT have detailed fields when include_details=false
                assert "api_base" not in model
                assert "max_tokens" not in model

    @pytest.mark.p1
    def test_list_user_models_with_details(self, client: RAGFlow) -> None:
        """Test that list_user_models returns models with detailed fields when include_details=true"""
        models: Dict[str, Any] = list_user_models(client, include_details=True)
        
        # Check structure for each factory
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            for model in llm_list:
                # Basic fields should be present
                assert "type" in model
                assert "name" in model
                assert "used_token" in model
                assert "status" in model
                
                # Detailed fields should be present when include_details=true
                assert "api_base" in model
                assert "max_tokens" in model

    @pytest.mark.p1
    def test_list_user_models_default_include_details(self, client: RAGFlow) -> None:
        """Test that list_user_models defaults to include_details=false"""
        models_basic: Dict[str, Any] = list_user_models(client)
        models_explicit: Dict[str, Any] = list_user_models(client, include_details=False)
        
        # Both should return the same structure (no detailed fields)
        for factory_name in models_basic.keys():
            if factory_name in models_explicit:
                llm_basic: List[Dict[str, Any]] = models_basic[factory_name]["llm"]
                llm_explicit: List[Dict[str, Any]] = models_explicit[factory_name]["llm"]
                
                # Check that both have the same number of models
                assert len(llm_basic) == len(llm_explicit)
                
                # Check that basic models don't have detailed fields
                for model in llm_basic:
                    assert "api_base" not in model
                    assert "max_tokens" not in model

    @pytest.mark.p1
    def test_list_user_models_model_types(self, client: RAGFlow) -> None:
        """Test that list_user_models returns models with valid types"""
        models: Dict[str, Any] = list_user_models(client)
                
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            for model in llm_list:
                model_type: str = model.get("type")
                # Type should be a string (may be one of the valid types or a custom type)
                assert isinstance(model_type, str)

    @pytest.mark.p1
    def test_list_user_models_model_names(self, client: RAGFlow) -> None:
        """Test that list_user_models returns models with valid names"""
        models: Dict[str, Any] = list_user_models(client)
        
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            for model in llm_list:
                model_name: str = model.get("name")
                # Name should be a non-empty string
                assert isinstance(model_name, str)
                assert len(model_name) > 0

    @pytest.mark.p1
    def test_list_user_models_used_token(self, client: RAGFlow) -> None:
        """Test that list_user_models returns models with used_token field"""
        models: Dict[str, Any] = list_user_models(client)
        
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            for model in llm_list:
                used_token: Any = model.get("used_token")
                # used_token should be an integer (may be 0 or higher)
                assert isinstance(used_token, int)
                assert used_token >= 0

    @pytest.mark.p1
    def test_list_user_models_status(self, client: RAGFlow) -> None:
        """Test that list_user_models returns models with status field"""
        models: Dict[str, Any] = list_user_models(client)
        
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            for model in llm_list:
                status: Any = model.get("status")
                # status should be a string
                assert isinstance(status, str)

    @pytest.mark.p2
    def test_list_user_models_api_base_with_details(self, client: RAGFlow) -> None:
        """Test that list_user_models returns api_base when include_details=true"""
        models: Dict[str, Any] = list_user_models(client, include_details=True)
        
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            for model in llm_list:
                api_base: Any = model.get("api_base")
                # api_base should be a string (may be empty)
                assert isinstance(api_base, str)

    @pytest.mark.p2
    def test_list_user_models_max_tokens_with_details(self, client: RAGFlow) -> None:
        """Test that list_user_models returns max_tokens when include_details=true"""
        models: Dict[str, Any] = list_user_models(client, include_details=True)
        
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            for model in llm_list:
                max_tokens: Any = model.get("max_tokens")
                # max_tokens should be an integer (default is 8192)
                assert isinstance(max_tokens, int)
                assert max_tokens > 0

    @pytest.mark.p2
    def test_list_user_models_consistency(self, client: RAGFlow) -> None:
        """Test that multiple calls to list_user_models return consistent results"""
        models1: Dict[str, Any] = list_user_models(client)
        models2: Dict[str, Any] = list_user_models(client)
        
        # Should return the same factories
        assert set(models1.keys()) == set(models2.keys())
        
        # Each factory should have the same number of models
        for factory_name in models1.keys():
            assert len(models1[factory_name]["llm"]) == len(models2[factory_name]["llm"])

    @pytest.mark.p2
    def test_list_user_models_tags(self, client: RAGFlow) -> None:
        """Test that list_user_models returns factory tags"""
        models: Dict[str, Any] = list_user_models(client)
        
        for factory_name, factory_data in models.items():
            factory_data.get("tags")
            # tags may be None, dict, or other types depending on factory
            # Just verify the key exists
            assert "tags" in factory_data

    @pytest.mark.p2
    def test_list_user_models_builtin_factory(self, client: RAGFlow) -> None:
        """Test that list_user_models includes Builtin factory models"""
        models: Dict[str, Any] = list_user_models(client)
        
        # Builtin factory should be present (it's always available)
        # Note: This may not always be true if no builtin models are configured
        # So we just check that if Builtin exists, it has the correct structure
        if "Builtin" in models:
            builtin_data: Dict[str, Any] = models["Builtin"]
            assert "tags" in builtin_data
            assert "llm" in builtin_data
            assert isinstance(builtin_data["llm"], list)

    @pytest.mark.p2
    def test_list_user_models_empty_factory(self, client: RAGFlow) -> None:
        """Test that list_user_models handles empty factory lists gracefully"""
        models: Dict[str, Any] = list_user_models(client)
        
        # Each factory should have an 'llm' list (even if empty)
        for factory_name, factory_data in models.items():
            llm_list: List[Dict[str, Any]] = factory_data["llm"]
            assert isinstance(llm_list, list)

    @pytest.mark.p3
    def test_list_user_models_compare_with_without_details(self, client: RAGFlow) -> None:
        """Test comparing results with and without include_details"""
        models_basic: Dict[str, Any] = list_user_models(client, include_details=False)
        models_detailed: Dict[str, Any] = list_user_models(client, include_details=True)
        
        # Both should have the same factories
        assert set(models_basic.keys()) == set(models_detailed.keys())
        
        # Each factory should have the same number of models
        for factory_name in models_basic.keys():
            basic_llm: List[Dict[str, Any]] = models_basic[factory_name]["llm"]
            detailed_llm: List[Dict[str, Any]] = models_detailed[factory_name]["llm"]
            assert len(basic_llm) == len(detailed_llm)
            
            # Basic info should match
            for i, basic_model in enumerate(basic_llm):
                detailed_model: Dict[str, Any] = detailed_llm[i]
                assert basic_model["type"] == detailed_model["type"]
                assert basic_model["name"] == detailed_model["name"]
                assert basic_model["used_token"] == detailed_model["used_token"]
                assert basic_model["status"] == detailed_model["status"]

    @pytest.mark.p3
    def test_list_user_models_response_format(self, client: RAGFlow) -> None:
        """Test that list_user_models returns data in the correct format"""
        models: Dict[str, Any] = list_user_models(client)
        
        # Top level should be a dictionary
        assert isinstance(models, dict)
        
        # Each factory entry should be a dictionary with specific structure
        for factory_name, factory_data in models.items():
            assert isinstance(factory_name, str)
            assert isinstance(factory_data, dict)
            assert "tags" in factory_data
            assert "llm" in factory_data
            assert isinstance(factory_data["llm"], list)
            
            # Each model in llm should be a dictionary
            for model in factory_data["llm"]:
                assert isinstance(model, dict)

    @pytest.mark.p3
    def test_list_user_models_multiple_calls(self, client: RAGFlow) -> None:
        """Test that multiple calls to list_user_models work correctly"""
        # Make multiple calls
        for _ in range(3):
            models: Dict[str, Any] = list_user_models(client)
            assert isinstance(models, dict)
            # Verify structure is consistent
            for factory_name, factory_data in models.items():
                assert "llm" in factory_data
                assert isinstance(factory_data["llm"], list)

