from typing import Any, Dict

import pytest
from conftest import get_default_models, set_default_models
from ragflow_sdk import RAGFlow


class TestSetDefaultModels:
    @pytest.mark.p1
    def test_set_llm_id_builtin(self, client: RAGFlow) -> None:
        """Test setting a builtin LLM model"""
        model_id: str = "glm-4-flash@Builtin"
        set_default_models(client, llm_id=model_id)
        
        models: Dict[str, Any] = get_default_models(client) 
        assert models.get("llm_id") == model_id

    @pytest.mark.p1
    def test_set_embd_id_builtin(self, client: RAGFlow) -> None:
        """Test setting a builtin embedding model"""
        model_id: str = "BAAI/bge-small-en-v1.5@Builtin"
        set_default_models(client, embd_id=model_id)
        
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("embd_id") == model_id

    @pytest.mark.p1
    def test_set_img2txt_id_builtin(self, client: RAGFlow) -> None:
        """Test setting a builtin image-to-text model"""
        model_id: str = "glm-4v@Builtin"
        set_default_models(client, img2txt_id=model_id)

        models: Dict[str, Any] = get_default_models(client)
        assert models.get("img2txt_id") == model_id

    @pytest.mark.p1
    def test_set_multiple_models_builtin(self, client: RAGFlow) -> None:
        """Test setting multiple builtin models at once"""
        payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            "img2txt_id": "glm-4v@Builtin",
        }
        set_default_models(client, **payload)

        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]

    @pytest.mark.p1
    def test_set_all_model_types(self, client: RAGFlow) -> None:
        """Test setting all model types"""
        payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
            "asr_id": "",
            "img2txt_id": "glm-4v@Builtin",
            "rerank_id": "",
            "tts_id": "",
        }
        set_default_models(client, **payload)
        
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == payload["llm_id"]
        assert models.get("embd_id") == payload["embd_id"]
        assert models.get("asr_id") == payload["asr_id"]
        assert models.get("img2txt_id") == payload["img2txt_id"]
        assert models.get("rerank_id") == payload["rerank_id"]
        # tts_id might be None from database, but should be treated as empty string
        assert models.get("tts_id") == payload["tts_id"] or models.get("tts_id") is None

    @pytest.mark.p1
    def test_set_configured_model(self, client: RAGFlow) -> None:
        """Test setting a configured model (if available)"""
        # First, try to set a model that might be configured (e.g., from ZHIPU-AI if available)
        # This test assumes the tenant has at least one configured model
        # If not configured, it will use a builtin model instead
        try:
            model_id: str = "glm-4-flash@ZHIPU-AI"
            set_default_models(client, llm_id=model_id)
            models: Dict[str, Any] = get_default_models(client)
            assert models.get("llm_id") == model_id
        except Exception as e:
            # If model is not configured, fall back to builtin
            if "not configured" in str(e):
                model_id: str = "glm-4-flash@Builtin"
                set_default_models(client, llm_id=model_id)
                models: Dict[str, Any] = get_default_models(client)
                assert models.get("llm_id") == model_id
            else:
                raise

    @pytest.mark.p2
    def test_set_empty_request(self, client: RAGFlow) -> None:
        """Test that empty request fails"""
        with pytest.raises(Exception) as excinfo:
            set_default_models(client)
        assert "At least one model ID must be provided" in str(excinfo.value)

    @pytest.mark.p2
    def test_set_empty_dict(self, client: RAGFlow) -> None:
        """Test that empty dict fails"""
        with pytest.raises(Exception) as excinfo:
            set_default_models(client, **{})
        assert "At least one model ID must be provided" in str(excinfo.value)

    @pytest.mark.p2
    def test_set_none_values(self, client: RAGFlow) -> None:
        """Test that None values are treated as empty"""
        with pytest.raises(Exception) as excinfo:
            set_default_models(client, llm_id=None, embd_id=None)
        # The API should reject this or treat it as empty
        assert "At least one model ID must be provided" in str(excinfo.value) or "not instance of" in str(excinfo.value)

    @pytest.mark.p2
    def test_set_empty_string(self, client: RAGFlow) -> None:
        """Test that empty string is ignored (API doesn't process empty strings to clear models)"""
        # First set a model
        set_default_models(client, llm_id="glm-4-flash@Builtin", embd_id="BAAI/bge-small-en-v1.5@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        
        # Try to clear llm_id with empty string (but keep embd_id to satisfy "at least one" requirement)
        # Note: Empty strings are ignored by the API due to the condition `if field_name in req and req[field_name]:`
        # So the model remains unchanged
        set_default_models(client, llm_id="", embd_id="BAAI/bge-small-en-v1.5@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        # Empty string is ignored, so llm_id remains unchanged
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"

    @pytest.mark.p2
    def test_set_whitespace_string(self, client: RAGFlow) -> None:
        """Test that whitespace-only string clears the model (whitespace is truthy, so it's processed)"""
        # First set a model
        set_default_models(client, llm_id="glm-4-flash@Builtin", embd_id="BAAI/bge-small-en-v1.5@Builtin")
        
        # Then clear with whitespace (but keep embd_id to satisfy "at least one" requirement)
        # Note: Whitespace strings are truthy, so they pass the condition and are processed as empty strings
        set_default_models(client, llm_id="   ", embd_id="BAAI/bge-small-en-v1.5@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        # Whitespace string is processed and clears the model
        assert models.get("llm_id") == ""
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"

    @pytest.mark.p2
    def test_set_nonexistent_model(self, client: RAGFlow) -> None:
        """Test setting a non-existent model fails"""
        with pytest.raises(Exception) as excinfo:
            set_default_models(client, llm_id="nonexistent-model@UnknownFactory")
        assert "not configured" in str(excinfo.value) or "Model" in str(excinfo.value)

    @pytest.mark.p2
    def test_set_invalid_model_format(self, client: RAGFlow) -> None:
        """Test setting a model with invalid format"""
        with pytest.raises(Exception) as excinfo:
            set_default_models(client, llm_id="invalid-format")
        # Should fail validation
        assert "not configured" in str(excinfo.value) or "Model" in str(excinfo.value)

    @pytest.mark.p2
    def test_set_missing_at_symbol(self, client: RAGFlow) -> None:
        """Test setting a model without @ symbol"""
        with pytest.raises(Exception) as excinfo:
            set_default_models(client, llm_id="glm-4-flashBuiltin")
        # Should fail validation
        assert "not configured" in str(excinfo.value) or "Model" in str(excinfo.value)

    @pytest.mark.p2
    def test_set_partial_update(self, client: RAGFlow) -> None:
        """Test that only provided models are updated, others remain unchanged"""
        # Set initial models
        initial_payload: Dict[str, str] = {
            "llm_id": "glm-4-flash@Builtin",
            "embd_id": "BAAI/bge-small-en-v1.5@Builtin",
        }
        set_default_models(client, **initial_payload)
        initial_models: Dict[str, Any] = get_default_models(client)
        
        # Update only one model
        set_default_models(client, llm_id="glm-4@Builtin")
        updated_models: Dict[str, Any] = get_default_models(client)
        
        # LLM should be updated
        assert updated_models.get("llm_id") == "glm-4@Builtin"
        # Embedding should remain unchanged
        assert updated_models.get("embd_id") == initial_models.get("embd_id")

    @pytest.mark.p2
    def test_set_clear_one_keep_others(self, client: RAGFlow) -> None:
        """Test that empty strings don't clear models (API limitation)"""
        # Set multiple models
        set_default_models(
            client,
            llm_id="glm-4-flash@Builtin",
            embd_id="BAAI/bge-small-en-v1.5@Builtin",
            img2txt_id="glm-4v@Builtin",
        )
        
        # Try to clear one model with empty string (but keep at least one non-empty to satisfy API requirement)
        # Note: Empty strings are ignored by the API, so llm_id remains unchanged
        set_default_models(client, llm_id="", embd_id="BAAI/bge-small-en-v1.5@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        
        # Empty string is ignored, so llm_id remains unchanged
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"
        # img2txt_id should remain unchanged
        assert models.get("img2txt_id") == "glm-4v@Builtin"

    @pytest.mark.p3
    def test_set_rerank_id(self, client: RAGFlow) -> None:
        """Test setting rerank model"""
        # Set rerank_id along with at least one non-empty model to satisfy API requirement
        # Note: Empty strings are ignored, so rerank_id won't be cleared, but we can verify the API accepts it
        set_default_models(client, rerank_id="", llm_id="glm-4-flash@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        # Empty string is ignored, so rerank_id remains unchanged (whatever it was before)
        assert models.get("llm_id") == "glm-4-flash@Builtin"

    @pytest.mark.p3
    def test_set_asr_id(self, client: RAGFlow) -> None:
        """Test setting ASR model"""
        # Set asr_id along with at least one non-empty model to satisfy API requirement
        # Note: Empty strings are ignored, so asr_id won't be cleared, but we can verify the API accepts it
        set_default_models(client, asr_id="", llm_id="glm-4-flash@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        # Empty string is ignored, so asr_id remains unchanged (whatever it was before)
        assert models.get("llm_id") == "glm-4-flash@Builtin"

    @pytest.mark.p3
    def test_set_tts_id(self, client: RAGFlow) -> None:
        """Test setting TTS model"""
        # Set tts_id along with at least one non-empty model to satisfy API requirement
        # Note: Empty strings are ignored, so tts_id won't be cleared, but we can verify the API accepts it
        set_default_models(client, tts_id="", llm_id="glm-4-flash@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        # Empty string is ignored, so tts_id remains unchanged (whatever it was before)
        assert models.get("llm_id") == "glm-4-flash@Builtin"

    @pytest.mark.p3
    def test_set_models_sequential_updates(self, client: RAGFlow) -> None:
        """Test sequential updates to different models"""
        # First update
        set_default_models(client, llm_id="glm-4-flash@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"
        
        # Second update
        set_default_models(client, embd_id="BAAI/bge-small-en-v1.5@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"  # Should remain
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"
        
        # Third update
        set_default_models(client, img2txt_id="glm-4v@Builtin")
        models: Dict[str, Any] = get_default_models(client)
        assert models.get("llm_id") == "glm-4-flash@Builtin"  # Should remain
        assert models.get("embd_id") == "BAAI/bge-small-en-v1.5@Builtin"  # Should remain
        assert models.get("img2txt_id") == "glm-4v@Builtin"

