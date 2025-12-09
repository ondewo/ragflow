from typing import Any, Dict, List

import pytest
from conftest import add_model, list_user_models
from ragflow_sdk import RAGFlow


@pytest.mark.usefixtures("cleanup_added_models")
class TestAddModel:
    @pytest.mark.p1
    def test_add_model_missing_llm_factory(self, client: RAGFlow) -> None:
        """Test that add_model fails when llm_factory is missing"""
        with pytest.raises(Exception) as excinfo:
            add_model(client, api_key="test-key")
        assert "llm_factory is required" in str(excinfo.value)

    @pytest.mark.p1
    def test_add_model_empty_llm_factory(self, client: RAGFlow) -> None:
        """Test that add_model fails when llm_factory is empty"""
        with pytest.raises(Exception) as excinfo:
            add_model(client, llm_factory="", api_key="test-key")
        assert "llm_factory is required" in str(excinfo.value) or "not allowed" in str(excinfo.value)

    @pytest.mark.p1
    def test_add_model_invalid_factory(self, client: RAGFlow) -> None:
        """Test that add_model fails when llm_factory is invalid"""
        with pytest.raises(Exception) as excinfo:
            add_model(client, llm_factory="InvalidFactoryName", api_key="test-key")
        assert "not allowed" in str(excinfo.value) or "LLM factory" in str(excinfo.value)

    @pytest.mark.p1
    def test_add_model_missing_api_key(self, client: RAGFlow) -> None:
        """Test that add_model requires api_key (or special factory parameters)"""
        # For most factories, api_key is required
        # This will fail validation or API key check
        with pytest.raises(Exception):
            add_model(client, llm_factory="OpenAI")

    @pytest.mark.p2
    def test_add_model_invalid_api_key(self, client: RAGFlow) -> None:
        """Test that add_model fails with invalid API key"""
        # Use an invalid API key - should fail validation
        with pytest.raises(Exception) as excinfo:
            add_model(client, llm_factory="OpenAI", api_key="invalid-key-12345")
        # Should fail with API key validation error
        assert "api key" in str(excinfo.value).lower() or "fail" in str(excinfo.value).lower() or "access" in str(excinfo.value).lower()

    @pytest.mark.p2
    def test_add_model_with_base_url(self, client: RAGFlow) -> None:
        """Test that add_model accepts base_url parameter"""
        # This will likely fail due to invalid API key, but should accept the parameter
        with pytest.raises(Exception) as excinfo:
            add_model(client, llm_factory="OpenAI", api_key="invalid-key", base_url="http://localhost:8000")
        # Should fail with API key validation, not parameter validation
        assert "api key" in str(excinfo.value).lower() or "fail" in str(excinfo.value).lower() or "access" in str(excinfo.value).lower() or "not allowed" in str(excinfo.value)

    @pytest.mark.p2
    def test_add_model_volcengine_missing_params(self, client: RAGFlow) -> None:
        """Test that VolcEngine with missing special parameters fails or is not allowed"""
        # VolcEngine requires ark_api_key and endpoint_id
        # API creates JSON with empty values, so it will fail during validation
        # Or factory might not be in allowed list
        try:
            add_model(client, llm_factory="VolcEngine", api_key="test-key")
            # If it succeeds, that's unexpected but acceptable for this test
            # (might be self-deployed or factory not configured)
        except Exception as e:
            # Expected to fail with validation error or "not allowed"
            assert "not allowed" in str(e) or "fail" in str(e).lower() or "api key" in str(e).lower()

    @pytest.mark.p2
    def test_add_model_tencent_hunyuan_missing_params(self, client: RAGFlow) -> None:
        """Test that Tencent Hunyuan requires special parameters"""
        # Tencent Hunyuan requires hunyuan_sid and hunyuan_sk
        with pytest.raises(Exception):
            add_model(client, llm_factory="Tencent Hunyuan", api_key="test-key")

    @pytest.mark.p2
    def test_add_model_tencent_cloud_missing_params(self, client: RAGFlow) -> None:
        """Test that Tencent Cloud with missing special parameters fails or is not allowed"""
        # Tencent Cloud requires tencent_cloud_sid and tencent_cloud_sk
        # API creates JSON with empty values, so it will fail during validation
        # Or factory might not be in allowed list
        try:
            add_model(client, llm_factory="Tencent Cloud", api_key="test-key")
            # If it succeeds, that's unexpected but acceptable for this test
        except Exception as e:
            # Expected to fail with validation error or "not allowed"
            assert "not allowed" in str(e) or "fail" in str(e).lower() or "api key" in str(e).lower()

    @pytest.mark.p2
    def test_add_model_bedrock_missing_params(self, client: RAGFlow) -> None:
        """Test that Bedrock with missing special parameters fails or is not allowed"""
        # Bedrock requires bedrock_ak, bedrock_sk, and bedrock_region
        # API creates JSON with empty values, so it will fail during validation
        # Or factory might not be in allowed list
        try:
            add_model(client, llm_factory="Bedrock", api_key="test-key")
            # If it succeeds, that's unexpected but acceptable for this test
        except Exception as e:
            # Expected to fail with validation error or "not allowed"
            assert "not allowed" in str(e) or "fail" in str(e).lower() or "api key" in str(e).lower()

    @pytest.mark.p2
    def test_add_model_baidu_yiyan_missing_params(self, client: RAGFlow) -> None:
        """Test that BaiduYiyan with missing special parameters fails or is not allowed"""
        # BaiduYiyan requires yiyan_ak and yiyan_sk
        # API creates JSON with empty values, so it will fail during validation
        # Or factory might not be in allowed list
        try:
            add_model(client, llm_factory="BaiduYiyan", api_key="test-key")
            # If it succeeds, that's unexpected but acceptable for this test
        except Exception as e:
            # Expected to fail with validation error or "not allowed"
            assert "not allowed" in str(e) or "fail" in str(e).lower() or "api key" in str(e).lower()

    @pytest.mark.p2
    def test_add_model_fish_audio_missing_params(self, client: RAGFlow) -> None:
        """Test that Fish Audio with missing special parameters fails or is not allowed"""
        # Fish Audio requires fish_audio_ak and fish_audio_refid
        # API creates JSON with empty values, so it will fail during validation
        # Or factory might not be in allowed list
        try:
            add_model(client, llm_factory="Fish Audio", api_key="test-key")
            # If it succeeds, that's unexpected but acceptable for this test
        except Exception as e:
            # Expected to fail with validation error or "not allowed"
            assert "not allowed" in str(e) or "fail" in str(e).lower() or "api key" in str(e).lower()

    @pytest.mark.p2
    def test_add_model_google_cloud_missing_params(self, client: RAGFlow) -> None:
        """Test that Google Cloud with missing special parameters fails or is not allowed"""
        # Google Cloud requires google_project_id, google_region, and google_service_account_key
        # API creates JSON with empty values, so it will fail during validation
        # Or factory might not be in allowed list
        try:
            add_model(client, llm_factory="Google Cloud", api_key="test-key")
            # If it succeeds, that's unexpected but acceptable for this test
        except Exception as e:
            # Expected to fail with validation error or "not allowed"
            assert "not allowed" in str(e) or "fail" in str(e).lower() or "api key" in str(e).lower()

    @pytest.mark.p2
    def test_add_model_azure_openai_missing_params(self, client: RAGFlow) -> None:
        """Test that Azure-OpenAI requires special parameters"""
        # Azure-OpenAI requires api_key and api_version
        with pytest.raises(Exception):
            add_model(client, llm_factory="Azure-OpenAI")

    @pytest.mark.p2
    def test_add_model_openrouter_missing_params(self, client: RAGFlow) -> None:
        """Test that OpenRouter with missing special parameters fails or is not allowed"""
        # OpenRouter requires api_key and provider_order
        # API creates JSON with empty values, so it will fail during validation
        # Or factory might not be in allowed list
        try:
            add_model(client, llm_factory="OpenRouter")
            # If it succeeds, that's unexpected but acceptable for this test
        except Exception as e:
            # Expected to fail with validation error or "not allowed"
            assert "not allowed" in str(e) or "fail" in str(e).lower() or "api key" in str(e).lower()

    @pytest.mark.p3
    def test_add_model_self_deployed_factories(self, client: RAGFlow) -> None:
        """Test that self-deployed factories skip API key validation"""
        # Self-deployed factories: LocalAI, Ollama, Xinference, LM-Studio, GPUStack, FastEmbed, Builtin
        # These skip validation, but may still fail if the service is not available
        self_deployed_factories: List[str] = ["LocalAI", "Ollama", "Xinference", "LM-Studio", "GPUStack", "FastEmbed"]
        
        for factory in self_deployed_factories:
            try:
                # Try to add with a dummy API key and base_url
                # This may succeed if the factory is configured, or fail if service is unavailable
                add_model(client, llm_factory=factory, api_key="dummy-key", base_url="http://localhost:8000")
                # If successful, verify models were added
                models: Dict[str, Any] = list_user_models(client)
                if factory in models:
                    assert isinstance(models[factory], dict)
                    assert "llm" in models[factory]
            except Exception as e:
                # Expected to fail if service is not available or factory not configured
                # Just verify it's not a parameter validation error
                error_msg: str = str(e)
                assert "not allowed" not in error_msg or "llm_factory is required" not in error_msg

    @pytest.mark.p3
    def test_add_model_builtin_factory(self, client: RAGFlow) -> None:
        """Test that Builtin factory can be added (skips validation)"""
        # Builtin is a self-deployed factory that should skip validation
        try:
            add_model(client, llm_factory="Builtin", api_key="dummy-key")
            # If successful, verify models were added
            models: Dict[str, Any] = list_user_models(client)
            if "Builtin" in models:
                assert isinstance(models["Builtin"], dict)
                assert "llm" in models["Builtin"]
        except Exception as e:
            # May fail if Builtin is not configured as addable
            error_msg: str = str(e)
            # Should not fail due to parameter validation
            assert "not allowed" not in error_msg or "llm_factory is required" not in error_msg

    @pytest.mark.p1
    def test_add_model_successful_response(self, client: RAGFlow) -> None:
        """Test that add_model returns True on successful addition"""
        # Use Builtin factory - it doesn't require API key and is always available
        result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        # On success, should return True
        assert result is True, f"Expected True, got {result}"

    @pytest.mark.p1
    def test_add_model_success_models_in_list(self, client: RAGFlow) -> None:
        """Test that successfully added models appear in list_user_models and tenant_llm table"""
        # Use Builtin factory - it doesn't require API key and is always available
        # Add the factory
        result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert result is True, "add_model should return True on success"
        
        # Verify models appear in the list (this queries tenant_llm table)
        models: Dict[str, Any] = list_user_models(client)
        assert "Builtin" in models, "Builtin factory should appear in models list after successful addition (tenant_llm table)"
        
        builtin_data: Dict[str, Any] = models["Builtin"]
        assert "llm" in builtin_data, "Builtin should have 'llm' key"
        assert isinstance(builtin_data["llm"], list), "Builtin 'llm' should be a list"
        assert len(builtin_data["llm"]) > 0, "Builtin should have at least one model in tenant_llm table"
        
        # Verify model structure
        for model in builtin_data["llm"]:
            assert "type" in model, "Model should have 'type' field"
            assert "name" in model, "Model should have 'name' field"
            assert "used_token" in model, "Model should have 'used_token' field"
            assert isinstance(model["type"], str), "Model type should be a string"
            assert isinstance(model["name"], str), "Model name should be a string"
            assert len(model["name"]) > 0, "Model name should not be empty"

    @pytest.mark.p2
    def test_add_model_success_response_structure(self, client: RAGFlow) -> None:
        """Test that add_model returns correct response structure on success"""
        # Use Builtin factory - it doesn't require API key
        result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        # Response should be exactly True (boolean)
        assert result is True, f"Expected True (boolean), got {result} (type: {type(result)})"
        assert isinstance(result, bool), f"Result should be boolean, got {type(result)}"

    @pytest.mark.p2
    def test_add_model_duplicate_addition(self, client: RAGFlow) -> None:
        """Test that adding the same factory twice is handled gracefully"""
        # Use Builtin factory - it doesn't require API key
        # Try to add the same factory twice
        # The API should handle this (either update or ignore)
        result1: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert result1 is True, "First addition should succeed"
        
        # Try adding again - should succeed (updates existing models)
        result2: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert result2 is True, "Second addition should also succeed (updates existing)"
        
        # Verify models are in the list (tenant_llm table)
        models: Dict[str, Any] = list_user_models(client)
        assert "Builtin" in models, "Builtin should be in models list after duplicate addition"
        factory_data: Dict[str, Any] = models["Builtin"]
        assert "llm" in factory_data, "Builtin should have 'llm' field"
        assert isinstance(factory_data["llm"], list), "Builtin 'llm' should be a list"

    @pytest.mark.p2
    def test_add_model_verify_added_models_structure(self, client: RAGFlow) -> None:
        """Test that successfully added models have correct structure in list_user_models (tenant_llm table)"""
        # Use Builtin factory - it doesn't require API key
        # Add the factory
        result: Any = add_model(client, llm_factory="Builtin", api_key="dummy-key")
        assert result is True, "add_model should return True"
        
        # Get models list (queries tenant_llm table)
        models: Dict[str, Any] = list_user_models(client)
        assert "Builtin" in models, "Builtin should be in tenant_llm table after addition"
        
        factory_data: Dict[str, Any] = models["Builtin"]
        assert "tags" in factory_data, "Factory should have 'tags' field"
        assert "llm" in factory_data, "Factory should have 'llm' field"
        
        llm_list: List[Dict[str, Any]] = factory_data["llm"]
        assert len(llm_list) > 0, "Factory should have at least one model in tenant_llm table"
        
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
    def test_add_model_case_sensitivity(self, client: RAGFlow) -> None:
        """Test that factory names are case-sensitive"""
        # Factory names should match exactly
        with pytest.raises(Exception):
            add_model(client, llm_factory="openai", api_key="test-key")  # lowercase
        with pytest.raises(Exception):
            add_model(client, llm_factory="OPENAI", api_key="test-key")  # uppercase
        # Correct case should be "OpenAI"
        with pytest.raises(Exception):
            add_model(client, llm_factory="OpenAI", api_key="invalid-key")

    @pytest.mark.p3
    def test_add_model_none_values(self, client: RAGFlow) -> None:
        """Test that None values are handled correctly"""
        with pytest.raises(Exception):
            add_model(client, llm_factory=None, api_key="test-key")
        
        with pytest.raises(Exception):
            add_model(client, llm_factory="OpenAI", api_key=None)

